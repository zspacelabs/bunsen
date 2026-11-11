//! The query language: `{select, prefer, order, cutoff, limit}`, and what it
//! compiles to.
//!
//! ```yaml
//! select:                            # the hard filter
//!   from:    [qwen3]                 # a member of at least one
//!   require: [hybrid]                # a member of every one
//!   exclude: [qwen3-2507]            # a member of none
//!   where:   {thinking: false, base: false, parameters: {max: 8e9}}
//! prefer:                            # soft scores
//!   where:   {language: {is: en, otherwise: 0.5}}
//!   release: latest                  # or {step, floor}
//! order: [-weight, -parameters]      # most significant first; `-` descending
//! cutoff: 0.5                        # the least weight kept; absent, 0
//! limit: 3                           # how many are kept; absent, all
//! ```
//!
//! - **Membership** (`from`, `require`, `exclude`) lists references, found by
//!   lookup, to a group (namespace, contract or release: its membership
//!   closure), a template class (the build's chain includes it), or a build
//!   (itself).
//! - **`where`** keys are declared axes and tags (and keys a derive step
//!   declares), each value typed by its declaration ([`Predicate`]). The facet
//!   keys and `contract` are refused: membership is in the lists.
//! - **`prefer`** mirrors `select.where`, an entry scoring 1 when it holds and
//!   `otherwise` (default 0) when not, a language graded with `truncated`
//!   (default 1); and `release` is [`Latest`].
//! - **`order`** keys are `weight` and numeric tags.
//! - **`cutoff`** drops the candidates whose weight is below it, and
//!   **`limit`** keeps the first so many of the rest, in order. A choice of one
//!   build takes the first.
//!
//! Queries combine as RFC 7386 merge patches ([`merge`]): every level is a
//! map, but the membership lists and `order`, which a patch replaces whole.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{
    Map,
    Value,
};

use crate::{
    document::{
        AxisKind,
        CatalogEntry,
        EntryKind,
        TagType,
    },
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogResult,
    },
    lookup::LeafMatch,
    merge_patch,
    name::{
        CatalogName,
        CatalogRef,
    },
    resolve::CONTRACT_TAG,
    select::{
        core::{
            DerivedKey,
            Order,
            Selection,
            SelectionCandidate,
        },
        latest::Latest,
        predicate::{
            Predicate,
            Subject,
        },
    },
    stack::PretrainedCatalogStack,
};

/// The `order` key that sorts by weight.
pub const WEIGHT: &str = "weight";

/// The kinds a membership reference may name.
pub const MEMBERSHIP_KINDS: [EntryKind; 5] = [
    EntryKind::Namespace,
    EntryKind::Contract,
    EntryKind::Release,
    EntryKind::TemplateClass,
    EntryKind::Build,
];

pub(crate) fn selection_error(message: impl Into<String>) -> CatalogError {
    CatalogError::new(CatalogErrorKind::Selection, message)
}

/// A query, as written: see the [module](self).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Query {
    /// `select.from`.
    pub from: Vec<String>,
    /// `select.require`.
    pub require: Vec<String>,
    /// `select.exclude`.
    pub exclude: Vec<String>,
    /// `select.where`, by key.
    pub filter: BTreeMap<String, Value>,
    /// `prefer.where`, by key.
    pub prefer: BTreeMap<String, Value>,
    /// `prefer.release`.
    pub latest: Option<Latest>,
    /// `order`, most significant first.
    pub order: Vec<(String, Order)>,
    /// `cutoff`.
    pub cutoff: Option<f64>,
    /// `limit`.
    pub limit: Option<usize>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawQuery {
    #[serde(default)]
    select: RawSelect,
    #[serde(default)]
    prefer: RawPrefer,
    #[serde(default)]
    order: Vec<String>,
    #[serde(default)]
    cutoff: Option<f64>,
    #[serde(default)]
    limit: Option<Value>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawSelect {
    #[serde(default)]
    from: Vec<String>,
    #[serde(default)]
    require: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default, rename = "where")]
    filter: BTreeMap<String, Value>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawPrefer {
    #[serde(default, rename = "where")]
    filter: BTreeMap<String, Value>,
    #[serde(default)]
    release: Option<Value>,
}

fn parse_score(
    what: &str,
    value: &Value,
) -> CatalogResult<f64> {
    value
        .as_f64()
        .filter(|v| (0.0..=1.0).contains(v))
        .ok_or_else(|| selection_error(format!("{what} is a score from 0 to 1, not {value}")))
}

impl Query {
    /// The query `value` writes. Its references and keys are read here and
    /// resolved by [`compile`](Self::compile).
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] for an unknown key or a part of the
    /// wrong shape.
    pub fn from_value(value: &Value) -> CatalogResult<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        let raw: RawQuery = serde_json::from_value(value.clone())
            .map_err(|e| selection_error(format!("a query: {e}")))?;
        let latest = match raw.prefer.release {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if s == "latest" => Some(Latest::default()),
            Some(Value::Object(params)) if params.keys().all(|k| k == "step" || k == "floor") => {
                let mut latest = Latest::default();
                if let Some(step) = params.get("step") {
                    latest.step = parse_score("`prefer.release.step`", step)?;
                }
                if let Some(floor) = params.get("floor") {
                    latest.floor = parse_score("`prefer.release.floor`", floor)?;
                }
                Some(latest)
            }
            Some(other) => {
                return Err(selection_error(format!(
                    "`prefer.release` is `latest` or `{{step, floor}}`, not {other}"
                )));
            }
        };
        if let Some(cutoff) = raw.cutoff
            && !(cutoff.is_finite() && cutoff >= 0.0)
        {
            return Err(selection_error(format!(
                "`cutoff` is a weight, 0 or more, not {cutoff}"
            )));
        }
        let limit = match raw.limit {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_u64()
                    .filter(|n| *n >= 1)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        selection_error(format!("`limit` is a count, 1 or more, not {value}"))
                    })?,
            ),
        };
        Ok(Query {
            from: raw.select.from,
            require: raw.select.require,
            exclude: raw.select.exclude,
            filter: raw.select.filter,
            prefer: raw.prefer.filter,
            latest,
            order: raw
                .order
                .iter()
                .map(|key| match key.strip_prefix('-') {
                    Some(key) => (key.to_string(), Order::Descending),
                    None => (key.to_string(), Order::Ascending),
                })
                .collect(),
            cutoff: raw.cutoff,
            limit,
        })
    }

    /// Resolves the query against `keys`'s stack, and builds its steps.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] for a reference or key that resolves
    /// to nothing (or to something it can't be), or a value that doesn't
    /// fit its key's declaration.
    pub fn compile(
        &self,
        keys: &TagKeys<'_>,
    ) -> CatalogResult<CompiledQuery> {
        let refs = |list: &[String], what: &str| -> CatalogResult<Vec<Member>> {
            list.iter().map(|r| keys.member(r, what)).collect()
        };
        let mut filter = Vec::new();
        for (key, value) in &self.filter {
            filter.push(Predicate::compile(keys.subject(key)?, value)?);
        }
        let mut prefer = Vec::new();
        for (key, entry) in &self.prefer {
            let (value, otherwise, truncated) = split_preference(key, entry)?;
            prefer.push(Preference {
                predicate: Predicate::compile(keys.subject(key)?, &value)?,
                otherwise,
                truncated,
            });
        }
        let mut order = Vec::new();
        for (key, direction) in &self.order {
            order.push((keys.order_key(key)?, *direction));
        }
        Ok(CompiledQuery {
            from: refs(&self.from, "select.from")?,
            require: refs(&self.require, "select.require")?,
            exclude: refs(&self.exclude, "select.exclude")?,
            filter,
            prefer,
            latest: self.latest,
            order,
            cutoff: self.cutoff.unwrap_or(0.0),
            limit: self.limit,
        })
    }
}

/// A `prefer.where` entry's value form, `otherwise` and `truncated`.
fn split_preference(
    key: &str,
    entry: &Value,
) -> CatalogResult<(Value, f64, f64)> {
    let Value::Object(fields) = entry else {
        return Ok((entry.clone(), 0.0, 1.0));
    };
    let mut rest = fields.clone();
    let otherwise = rest
        .remove("otherwise")
        .map(|v| parse_score(&format!("`prefer.where.{key}.otherwise`"), &v))
        .transpose()?
        .unwrap_or(0.0);
    let truncated = rest
        .remove("truncated")
        .map(|v| parse_score(&format!("`prefer.where.{key}.truncated`"), &v))
        .transpose()?
        .unwrap_or(1.0);
    let value = match rest.remove("is") {
        Some(is) if rest.is_empty() => is,
        Some(_) => {
            return Err(selection_error(format!(
                "`prefer.where.{key}`: `is` stands alone beside `otherwise` and `truncated`"
            )));
        }
        None => Value::Object(rest),
    };
    Ok((value, otherwise, truncated))
}

/// A resolved membership reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    /// What it named.
    pub name: CatalogName,
    /// Its kind.
    pub kind: EntryKind,
}

impl Member {
    /// Whether `candidate` is in it.
    pub fn contains(
        &self,
        candidate: &SelectionCandidate,
    ) -> bool {
        let build = &candidate.build;
        match self.kind {
            EntryKind::Build => build.name == self.name,
            EntryKind::TemplateClass => build.template_chain.contains(&self.name),
            _ => build.memberships.contains(&self.name),
        }
    }
}

/// A `prefer.where` entry, compiled.
#[derive(Clone, Debug, PartialEq)]
pub struct Preference {
    /// What it tests.
    pub predicate: Predicate,
    /// The score when it doesn't hold.
    pub otherwise: f64,
    /// A language's score through a truncated request.
    pub truncated: f64,
}

/// A query with its references and keys resolved: the steps it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledQuery {
    /// `select.from`.
    pub from: Vec<Member>,
    /// `select.require`.
    pub require: Vec<Member>,
    /// `select.exclude`.
    pub exclude: Vec<Member>,
    /// `select.where`.
    pub filter: Vec<Predicate>,
    /// `prefer.where`.
    pub prefer: Vec<Preference>,
    /// `prefer.release`.
    pub latest: Option<Latest>,
    /// The order, by resolved key.
    pub order: Vec<(String, Order)>,
    /// The least weight kept.
    pub cutoff: f64,
    /// How many are kept; `None`, all.
    pub limit: Option<usize>,
}

impl CompiledQuery {
    /// Whether `candidate` passes `select`.
    pub fn selects(
        &self,
        candidate: &SelectionCandidate,
    ) -> bool {
        (self.from.is_empty() || self.from.iter().any(|m| m.contains(candidate)))
            && self.require.iter().all(|m| m.contains(candidate))
            && !self.exclude.iter().any(|m| m.contains(candidate))
            && self.filter.iter().all(|p| p.holds(candidate))
    }

    /// Ranks `selection`: `select`, then each preference, then `release`,
    /// then `order`. [`cut`](Self::cut) then makes the result.
    pub fn apply(
        &self,
        selection: Selection,
    ) -> Selection {
        let mut selection = selection.weigh_each(|c| if self.selects(c) { 1.0 } else { 0.0 });
        for preference in &self.prefer {
            selection = selection.weigh_each(|c| {
                preference
                    .predicate
                    .grade(c, preference.truncated, preference.otherwise)
            });
        }
        if let Some(latest) = &self.latest {
            selection = selection.weigh(latest);
        }
        // A tuple order is stable sorts, least significant first.
        for (key, direction) in self.order.iter().rev() {
            selection = if key == WEIGHT {
                selection.order(&[*direction], |_, w| [Some(w)])
            } else {
                selection.order(&[*direction], |c, _| [c.number(key)])
            };
        }
        selection
    }

    /// The result of a ranked `selection`: those at or above `cutoff`, the
    /// first `limit` of them.
    pub fn cut(
        &self,
        selection: Selection,
    ) -> Selection {
        let selection = selection.cutoff(self.cutoff);
        match self.limit {
            Some(limit) => selection.limit(limit),
            None => selection,
        }
    }
}

/// What a query's references and keys resolve against: a stack, and the
/// keys derive steps declare.
pub struct TagKeys<'a> {
    stack: &'a PretrainedCatalogStack,
    derived: Vec<DerivedKey>,
}

impl<'a> TagKeys<'a> {
    /// The stack's keys, and `derived`.
    pub fn new(
        stack: &'a PretrainedCatalogStack,
        derived: Vec<DerivedKey>,
    ) -> Self {
        Self { stack, derived }
    }

    /// Resolves a membership reference.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] when it names nothing a build can be
    /// in, or more than one thing.
    pub fn member(
        &self,
        reference: &str,
        what: &str,
    ) -> CatalogResult<Member> {
        let found = match CatalogRef::parse(reference) {
            Ok(r) if r.is_absolute() => self
                .stack
                .declaration(r.path())
                .map(|d| d.name().clone())
                .ok_or_else(|| format!("nothing is declared as `{r}`")),
            _ => self
                .stack
                .find(reference, &MEMBERSHIP_KINDS, LeafMatch::Exact)
                .map_err(|e| e.message),
        }
        .map_err(|why| selection_error(format!("`{what}`: `{reference}`: {why}")))?;
        let kind = self
            .stack
            .declaration(&found)
            .map(|d| d.kind())
            .expect("found");
        if !MEMBERSHIP_KINDS.contains(&kind) {
            return Err(selection_error(format!(
                "`{what}`: `{reference}` is a {kind}; a build can be in a group, a template \
                 class or a build"
            )));
        }
        Ok(Member { name: found, kind })
    }

    /// Resolves a `where` key to what it tests.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] for a key nothing declares, a facet
    /// or the contract (membership is in the lists), or more than one match.
    pub fn subject(
        &self,
        key: &str,
    ) -> CatalogResult<Subject> {
        if self.derived.iter().any(|d| d.name == key) {
            return Ok(Subject::Derived {
                key: key.to_string(),
            });
        }
        if key == CONTRACT_TAG {
            return Err(selection_error(
                "`where.contract`: a build's contract is membership: use \
                 `select.require: [<contract>]` or `select.from`",
            ));
        }
        let kinds = [EntryKind::Tag, EntryKind::Axis, EntryKind::Facet];
        let name = self
            .stack
            .find(key, &kinds, LeafMatch::Exact)
            .map_err(|e| {
                selection_error(format!(
                    "`where.{key}`: {}; every key a query tests is declared",
                    e.message
                ))
            })?;
        let decl = self.stack.declaration(&name).expect("found");
        match decl.body() {
            CatalogEntry::Facet(_) => Err(selection_error(format!(
                "`where.{key}`: `{name}` is a facet, which is membership: use \
                 `select.require: [<group>]` or `select.from`"
            ))),
            CatalogEntry::Axis(a) => Ok(match a.axis {
                AxisKind::Bool => Subject::BoolAxis { axis: name },
                AxisKind::Categorical => Subject::CategoricalAxis {
                    axis: name,
                    members: a.member_names(),
                },
            }),
            CatalogEntry::Tag(t) => Ok(Subject::Tag {
                key: name.to_string(),
                tag_type: t.tag_type.clone(),
            }),
            _ => unreachable!("found among tags, axes and facets"),
        }
    }

    /// Resolves an `order` key: `weight`, a numeric tag, or a derived key.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] for anything else.
    pub fn order_key(
        &self,
        key: &str,
    ) -> CatalogResult<String> {
        if key == WEIGHT {
            return Ok(WEIGHT.to_string());
        }
        match self.subject(key)? {
            Subject::Derived { key } => Ok(key),
            Subject::Tag {
                key,
                tag_type: TagType::Count,
            } => Ok(key),
            other => Err(selection_error(format!(
                "`order`: `{}` is not a number to sort by; order by `weight`, a count tag, or a \
                 derived key",
                other.key()
            ))),
        }
    }
}

/// `objects` merged in order, each an RFC 7386 merge patch over those before
/// it; a `null` is skipped.
pub fn merge<'v>(objects: impl IntoIterator<Item = &'v Value>) -> Value {
    let mut merged = Value::Object(Map::new());
    for object in objects {
        if !object.is_null() {
            merge_patch::apply(&mut merged, object);
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        fixtures,
        select::testing::{
            all_builds,
            shown,
        },
    };

    fn compile(value: Value) -> CatalogResult<CompiledQuery> {
        let stack = fixtures::builtin_stack().unwrap();
        Query::from_value(&value)?.compile(&TagKeys::new(&stack, vec![]))
    }

    fn run(value: Value) -> Vec<String> {
        shown(&compile(value).unwrap().apply(all_builds()))
    }

    #[test]
    fn test_membership_lists() {
        assert_eq!(
            run(json!({"select": {"from": ["whisper"]}})),
            ["large-v1", "large-v2", "large-v3", "large-v3-turbo"]
        );
        assert_eq!(
            run(json!({"select": {"from": ["qwen3-0.6b", "large-v3"]}})),
            ["qwen3-0.6b", "large-v3"]
        );
        assert_eq!(
            run(json!({"select": {"require": ["whisper-asr", "release"]}})),
            ["large-v1", "large-v2", "large-v3"]
        );
        assert_eq!(
            run(json!({"select": {"require": ["hybrid"], "exclude": ["thinking-only"]}})),
            ["qwen3-0.6b", "qwen3-0.6b-base"]
        );
        // Two contracts can't both be required.
        assert!(run(json!({"select": {"require": ["qwen3-lm", "whisper-asr"]}})).is_empty());
    }

    #[test]
    fn test_where_is_typed() {
        assert_eq!(
            run(json!({"select": {"where": {"thinking": "switch"}}})),
            ["qwen3-0.6b"]
        );
        assert_eq!(
            run(json!({"select": {"where": {"task": ["transcribe", "translate"]}}})),
            ["large-v1", "large-v2", "large-v3"]
        );
        assert_eq!(
            run(json!({"select": {"where": {"timestamps": true}}})),
            ["large-v3"]
        );
        let refused = |value: Value| compile(value).unwrap_err().message;
        assert!(
            refused(json!({"select": {"where": {"base": "yes"}}}))
                .contains("is not `true` or `false`")
        );
        assert!(refused(json!({"select": {"where": {"family": "qwen3"}}})).contains("is a facet"));
        assert!(
            refused(json!({"select": {"where": {"contract": "qwen3-lm"}}}))
                .contains("is membership")
        );
        assert!(
            refused(json!({"select": {"where": {"kit": "qwen3"}}}))
                .contains("every key a query tests is declared")
        );
        assert!(refused(json!({"select": {"from": ["parameters"]}})).contains("nothing matches"));
    }

    #[test]
    fn test_prefer_release_order_cut() {
        let q = compile(json!({
            "select": {"from": ["whisper"]},
            "prefer": {"release": {"step": 0.25}},
            "order": ["-weight", "-parameters"],
        }))
        .unwrap();
        assert_eq!(
            shown(&q.apply(all_builds()))[..2],
            ["large-v3", "large-v3-turbo"]
        );
        let lang = compile(json!({
            "prefer": {"where": {"language": {"is": "en", "otherwise": 0.5}}},
            "order": ["-weight"],
        }))
        .unwrap();
        let s = lang.apply(all_builds());
        assert_eq!(s.iter().next().unwrap().1, 1.0);
        assert_eq!(s.iter().last().unwrap().1, 0.5);
        assert!(compile(json!({"order": ["thinking"]})).is_err());
        let cut = compile(json!({"order": ["-weight"], "cutoff": 0.75, "limit": 1}))
            .unwrap()
            .cut(s);
        assert_eq!(cut.len(), 1);
        assert_eq!(cut.first().unwrap().1, 1.0);
        assert!(compile(json!({"cutoff": -1})).is_err());
        for limit in [json!(0), json!(1.5), json!("all")] {
            assert!(compile(json!({ "limit": limit })).is_err(), "{limit}");
        }
        assert!(compile(json!({"prefer": {"release": "newest"}})).is_err());
        assert!(compile(json!({"selects": {}})).is_err());
    }

    #[test]
    fn test_merge_is_a_merge_patch() {
        let merged = merge(&[
            json!({"select": {"where": {"base": false, "thinking": "switch"}}, "cutoff": 0.5}),
            Value::Null,
            json!({"select": {"where": {"thinking": null}}, "order": ["-weight"]}),
        ]);
        assert_eq!(
            merged,
            json!({"select": {"where": {"base": false}}, "cutoff": 0.5, "order": ["-weight"]})
        );
    }
}
