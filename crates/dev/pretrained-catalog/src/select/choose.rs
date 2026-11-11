//! Choosing a build: a command's [`SelectionContext`], what the command
//! line asks for ([`ChoiceRequest`]), the [`Ranking`] it makes, and the
//! [`Choice`]s in its result.
//!
//! A query's result is a list: the ranked builds at or above its `cutoff`,
//! the first `limit` of them ([`SelectionContext::choices`]). Loading a model
//! takes one build, an implicit limit of 1: the first
//! ([`SelectionContext::choose`]).
//!
//! Each command-line flag fills the query key of the same name: a bare
//! `--select NAME` adds to `select.from` (or names a selection the command
//! defines), `--select {..}` patches `select`, `--prefer {..}` patches
//! `prefer`, `--order`, `--cutoff` and `--limit` set theirs, and `--query {..}`
//! patches the whole query. An axis flag such as `--off thinking` is a
//! shortcut, not a mode of its own: it is `select.where.thinking: false`, and
//! it selects the member it names as a preset.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{
    Map,
    Value,
    json,
};

use crate::{
    document::{
        AxisKind,
        CatalogEntry,
        EntryKind,
    },
    error::{
        CatalogError,
        CatalogResult,
    },
    lookup::LeafMatch,
    merge_patch,
    name::CatalogName,
    presets::PresetSelection,
    select::{
        core::{
            DeriveOp,
            DerivedKey,
            Selection,
            SelectionCandidate,
        },
        expr::{
            Query,
            TagKeys,
            merge,
            selection_error,
        },
    },
    stack::PretrainedCatalogStack,
};

/// A selection a command names: `--select en`.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedSelection {
    /// Its name.
    pub name: String,
    /// A line for a listing.
    pub description: String,
    /// The query it stands for, merged over the defaults.
    pub query: Value,
}

/// What a command chooses with: the default query every choice merges
/// over, and the selections it names.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectionContext {
    /// The default query: applied unless a choice patches it, and not at
    /// all when the only name is one build.
    pub defaults: Value,
    /// The named selections.
    pub named: Vec<NamedSelection>,
}

/// A flag that names a member of a registered bool axis: `--off thinking`
/// is `{axis: thinking, on: false}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisFlag {
    /// The flag as written, for messages.
    pub flag: String,
    /// The axis, as a lookup names it.
    pub axis: String,
    /// Whether it names the positive member.
    pub on: bool,
}

/// What a command line asks of a choice.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChoiceRequest {
    /// Each bare `--select NAME`, in order.
    pub names: Vec<String>,
    /// Query patches, in order (`--query`, `--select {..}`, `--prefer`,
    /// `--order`, `--cutoff`, `--limit`).
    pub patches: Vec<Value>,
    /// Each `--preset`.
    pub presets: Vec<String>,
    /// The axis flags.
    pub flags: Vec<AxisFlag>,
}

impl ChoiceRequest {
    /// `--select NAME`.
    pub fn name(
        mut self,
        name: impl Into<String>,
    ) -> Self {
        self.names.push(name.into());
        self
    }

    /// `--query {..}`: a patch of the whole query.
    pub fn query(
        mut self,
        patch: Value,
    ) -> Self {
        self.patches.push(patch);
        self
    }

    /// `--select {..}`: a patch of `select`.
    pub fn select(
        self,
        patch: Value,
    ) -> Self {
        self.query(json!({ "select": patch }))
    }

    /// `--prefer {..}`: a patch of `prefer`.
    pub fn prefer(
        self,
        patch: Value,
    ) -> Self {
        self.query(json!({ "prefer": patch }))
    }

    /// `--order KEY...`: the order, most significant first.
    pub fn order<S: AsRef<str>>(
        self,
        keys: &[S],
    ) -> Self {
        let keys: Vec<&str> = keys.iter().map(AsRef::as_ref).collect();
        self.query(json!({ "order": keys }))
    }

    /// `--cutoff W`: the least weight kept.
    pub fn cutoff(
        self,
        weight: f64,
    ) -> Self {
        self.query(json!({ "cutoff": weight }))
    }

    /// `--limit N`: how many are kept.
    pub fn limit(
        self,
        count: usize,
    ) -> Self {
        self.query(json!({ "limit": count }))
    }

    /// `--preset P`.
    pub fn preset(
        mut self,
        preset: impl Into<String>,
    ) -> Self {
        self.presets.push(preset.into());
        self
    }

    /// `--on AXIS` (or `--off AXIS`, when `on` is false).
    pub fn flag(
        mut self,
        axis: impl Into<String>,
        on: bool,
    ) -> Self {
        let axis = axis.into();
        self.flags.push(AxisFlag {
            flag: format!("--{} {axis}", if on { "on" } else { "off" }),
            axis,
            on,
        });
        self
    }
}

/// A ranking: every candidate a query ranks, in order, with its weight,
/// before `cutoff` and `limit` make the [result](Self::result).
#[derive(Clone, Debug, PartialEq)]
pub struct Ranking {
    /// The candidates, in order, weighted.
    pub selection: Selection,
    /// The least weight the result keeps.
    pub cutoff: f64,
    /// How many the result keeps; `None`, all.
    pub limit: Option<usize>,
    /// The query the choice ran, merged.
    pub query: Value,
    /// How the candidates were found, a line per step.
    pub why: Vec<String>,
    /// The builds the presets or flags ruled out, with the reasons.
    pub dropped: Vec<String>,
    /// Each flag, and the bool axis it names.
    pub flags: Vec<(AxisFlag, CatalogName)>,
    /// The `--preset`s.
    pub presets: Vec<String>,
}

impl Ranking {
    /// The result: the candidates at or above the cutoff, the first `limit`
    /// of them.
    pub fn result(&self) -> Selection {
        let selection = self.selection.clone().cutoff(self.cutoff);
        match self.limit {
            Some(limit) => selection.limit(limit),
            None => selection,
        }
    }

    /// Why a ranked candidate, the `index`th, is not in the result: `below
    /// cutoff`, `past limit`, or `None` when it is.
    pub fn excluded(
        &self,
        index: usize,
        weight: f64,
    ) -> Option<&'static str> {
        if weight < self.cutoff {
            return Some("below cutoff");
        }
        let kept_before = self
            .selection
            .iter()
            .take(index)
            .filter(|(_, w)| *w >= self.cutoff)
            .count();
        match self.limit {
            Some(limit) if kept_before >= limit => Some("past limit"),
            _ => None,
        }
    }

    /// The result as choices, in order: each with its presets and why.
    pub fn choices(&self) -> Vec<Choice> {
        self.result()
            .iter()
            .map(|(candidate, weight)| self.choice(candidate.clone(), weight))
            .collect()
    }

    /// `candidate`, a member of the result, as a choice.
    fn choice(
        &self,
        candidate: SelectionCandidate,
        weight: f64,
    ) -> Choice {
        let presets = self
            .presets_for(&candidate)
            .expect("kept only when its presets apply");
        let mut why = self.why.clone();
        for (i, (flag, _)) in self.flags.iter().enumerate() {
            let member = &presets[self.presets.len() + i];
            why.push(format!("`{}` implies `--preset {member}`", flag.flag));
        }
        why.push(format!("took `{}` at weight {weight}", candidate.name()));
        Choice {
            candidate,
            weight,
            presets: PresetSelection::of(presets),
            why,
        }
    }

    /// The ranking as a report: the merged query, why, what was dropped,
    /// and the result. With `every`, every ranked candidate is listed, and
    /// those outside the result carry the reason
    /// ([`excluded`](Self::excluded)).
    pub fn report(
        &self,
        every: bool,
    ) -> SelectionReport {
        let results = if every {
            self.selection
                .iter()
                .enumerate()
                .map(|(i, (c, w))| {
                    let mut record = self.choice(c.clone(), w).record();
                    record.excluded = self.excluded(i, w).map(String::from);
                    record
                })
                .collect()
        } else {
            self.choices().iter().map(Choice::record).collect()
        };
        SelectionReport {
            query: self.query.clone(),
            why: self.why.clone(),
            dropped: self.dropped.clone(),
            results,
        }
    }

    /// The error for an empty result, with the reasons builds were dropped.
    pub fn empty_error(&self) -> CatalogError {
        let mut message = "no build is left to choose".to_string();
        if !self.dropped.is_empty() {
            message.push_str(&format!(" (dropped: {})", self.dropped.join("; ")));
        }
        selection_error(message)
    }

    /// The presets `candidate` takes: the `--preset`s, then the member each
    /// flag names on it.
    ///
    /// # Errors
    /// The reason, when a flag's member isn't offered or the presets don't
    /// go together.
    pub fn presets_for(
        &self,
        candidate: &SelectionCandidate,
    ) -> Result<Vec<String>, String> {
        let mut presets = self.presets.clone();
        for (flag, axis) in &self.flags {
            let member = candidate
                .build
                .presets
                .bool_member(axis, flag.on)
                .map_err(|e| format!("`{}` doesn't apply: {}", flag.flag, e.message))?;
            presets.push(member);
        }
        candidate
            .build
            .presets
            .check_selection(&PresetSelection::of(presets.iter().cloned()))
            .map_err(|e| e.message)
            .map(|_| presets)
    }
}

/// A build in a query's result, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// The candidate.
    pub candidate: SelectionCandidate,
    /// Its weight.
    pub weight: f64,
    /// The presets to apply: each `--preset`, then what each flag implies.
    pub presets: PresetSelection,
    /// How it was chosen, a line per step.
    pub why: Vec<String>,
}

impl Choice {
    /// The choice as a record: its build, weight and presets.
    pub fn record(&self) -> ChoiceRecord {
        let build = &self.candidate.build;
        let names = |names: &mut dyn Iterator<Item = &CatalogName>| -> Vec<String> {
            names.map(ToString::to_string).collect()
        };
        ChoiceRecord {
            name: build.name.to_string(),
            weight: self.weight,
            excluded: None,
            presets: self.presets.presets.clone(),
            description: build.description.clone(),
            license: build.license.clone(),
            origin: build.origin.clone(),
            template_chain: names(&mut build.template_chain.iter()),
            memberships: names(&mut build.memberships.iter()),
            facets: build
                .facets
                .iter()
                .map(|(facet, group)| (facet.to_string(), group.to_string()))
                .collect(),
            contract: build.contract.as_ref().map(ToString::to_string),
            release: build.release.as_ref().map(|r| ReleaseRecord {
                group: r.group.to_string(),
                released: r.released.to_string(),
            }),
            tags: build
                .tags
                .iter()
                .map(|(name, tag)| (name.to_string(), tag.value.clone()))
                .collect(),
            derived: self.candidate.derived.clone(),
            dependencies: build
                .dependencies
                .iter()
                .map(|(key, dep)| (key.clone(), dep.resource.to_string()))
                .collect(),
        }
    }
}

/// A ranking, as `select` prints it: see [`Ranking::report`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SelectionReport {
    /// The query the choice ran, merged.
    pub query: Value,
    /// How the candidates were found, a line per step.
    pub why: Vec<String>,
    /// The builds the presets or flags ruled out, with the reasons.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dropped: Vec<String>,
    /// The result, in order (or every ranked candidate, when asked).
    pub results: Vec<ChoiceRecord>,
}

/// A build in a result, as `select` prints it: the resolved build's
/// fields, tags as plain values, with the choice's weight and presets.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChoiceRecord {
    /// The build's full name.
    pub name: String,
    /// Its weight.
    pub weight: f64,
    /// Why it is not in the result (`below cutoff`, `past limit`), when a
    /// report lists every ranked candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excluded: Option<String>,
    /// The presets the choice selects: each `--preset`, then each flag's
    /// member. Axis defaults apply beneath them.
    pub presets: Vec<String>,
    /// What it is, for a person.
    pub description: String,
    /// The checkpoint's license.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Where the checkpoint comes from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// The template-class chain, root first.
    pub template_chain: Vec<String>,
    /// Every group it is in.
    pub memberships: Vec<String>,
    /// Its group in each facet, by facet.
    pub facets: BTreeMap<String, String>,
    /// Its contract.
    pub contract: Option<String>,
    /// Its release membership.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release: Option<ReleaseRecord>,
    /// Its tags, written and implied, by full name.
    pub tags: BTreeMap<String, Value>,
    /// The tags a derive step added, by key.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub derived: BTreeMap<String, Value>,
    /// Its dependencies: each key's resource.
    pub dependencies: BTreeMap<String, String>,
}

/// A build's release membership, in a [`ChoiceRecord`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReleaseRecord {
    /// The release group.
    pub group: String,
    /// When the build was released.
    pub released: String,
}

impl SelectionContext {
    /// The named selection called `name`.
    pub fn named(
        &self,
        name: &str,
    ) -> Option<&NamedSelection> {
        self.named
            .iter()
            .find(|n| n.name.eq_ignore_ascii_case(name))
    }

    /// Chooses one build of `stack` (every build, or those in `universe`),
    /// as loading a model does: the first of the [result](Ranking::result).
    ///
    /// # Errors
    /// As [`rank`](Self::rank), and when the result is empty.
    pub fn choose(
        &self,
        stack: &PretrainedCatalogStack,
        universe: Option<&[CatalogName]>,
        derive: &[&dyn DeriveOp],
        request: &ChoiceRequest,
    ) -> CatalogResult<Choice> {
        let ranking = self.rank(stack, universe, derive, request)?;
        let Some((candidate, weight)) = ranking.result().first().map(|(c, w)| (c.clone(), w))
        else {
            return Err(ranking.empty_error());
        };
        Ok(ranking.choice(candidate, weight))
    }

    /// Every build in the result of `request` over `stack` (every build, or
    /// those in `universe`), in order: as many as its `limit` says, all when
    /// it has none.
    ///
    /// # Errors
    /// As [`rank`](Self::rank), and when the result is empty.
    pub fn choices(
        &self,
        stack: &PretrainedCatalogStack,
        universe: Option<&[CatalogName]>,
        derive: &[&dyn DeriveOp],
        request: &ChoiceRequest,
    ) -> CatalogResult<Vec<Choice>> {
        let ranking = self.rank(stack, universe, derive, request)?;
        let choices = ranking.choices();
        if choices.is_empty() {
            return Err(ranking.empty_error());
        }
        Ok(choices)
    }

    /// Ranks the builds of `stack` (every build, or those in `universe`) as
    /// `request` asks:
    ///
    /// 1. A name that a named selection has is that selection's query, merged
    ///    first; every other name is a reference, added to `select.from`. When
    ///    the only name is one build, the defaults don't apply.
    /// 2. The query is the defaults, the named selections, then the patches,
    ///    merged in order, with each flag's `select.where` entry.
    /// 3. The derive steps run; a build that doesn't offer a flag's member, or
    ///    can't take the `--preset`s and the flags' members together, is
    ///    dropped, with the reason.
    /// 4. The query compiles against the stack and runs.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`](crate::CatalogErrorKind::Selection)
    /// for a reference or key that resolves to nothing, a flag that names no
    /// bool axis, a value that doesn't fit its declaration, or a build named
    /// exactly that the presets or flags rule out.
    pub fn rank(
        &self,
        stack: &PretrainedCatalogStack,
        universe: Option<&[CatalogName]>,
        derive: &[&dyn DeriveOp],
        request: &ChoiceRequest,
    ) -> CatalogResult<Ranking> {
        let mut derived: Vec<DerivedKey> = Vec::new();
        for op in derive {
            derived.extend(op.keys());
        }
        let keys = TagKeys::new(stack, derived);
        let mut why = Vec::new();

        // 1. Names.
        let mut named = Vec::new();
        let mut refs = Vec::new();
        for name in &request.names {
            match self.named(name) {
                Some(selection) => {
                    why.push(format!("the selection `{}`", selection.name));
                    named.push(selection.query.clone());
                }
                None => refs.push(name.clone()),
            }
        }
        let exact = match (refs.as_slice(), named.is_empty()) {
            ([only], true) => keys
                .member(only, "select")
                .ok()
                .filter(|m| m.kind == EntryKind::Build)
                .map(|m| m.name),
            _ => None,
        };

        // 2. The query.
        let mut layers = Vec::new();
        if exact.is_none() {
            layers.push(self.defaults.clone());
        }
        layers.extend(named);
        layers.extend(request.patches.iter().cloned());
        let mut query = merge(&layers);
        if !refs.is_empty() {
            let select = query
                .as_object_mut()
                .expect("a merged query is an object")
                .entry("select")
                .or_insert_with(|| Value::Object(Map::new()));
            let Value::Object(select) = select else {
                return Err(selection_error("`select` is an object"));
            };
            let from = select
                .entry("from")
                .or_insert_with(|| Value::Array(Vec::new()));
            let Value::Array(from) = from else {
                return Err(selection_error("`select.from` is a list"));
            };
            from.extend(refs.iter().cloned().map(Value::String));
        }
        let mut flags = Vec::new();
        for flag in &request.flags {
            let axis = stack
                .find(&flag.axis, &[EntryKind::Axis], LeafMatch::Exact)
                .map_err(|e| selection_error(format!("`{}`: {}", flag.flag, e.message)))?;
            let is_bool = stack.declaration(&axis).is_some_and(
                |d| matches!(d.body(), CatalogEntry::Axis(a) if a.axis == AxisKind::Bool),
            );
            if !is_bool {
                return Err(selection_error(format!(
                    "`{}` names `{axis}`, which is not a bool axis",
                    flag.flag
                )));
            }
            merge_patch::apply(
                &mut query,
                &json!({"select": {"where": {axis.to_string(): flag.on}}}),
            );
            why.push(format!(
                "`{}` is `select.where.{axis}: {}`",
                flag.flag, flag.on
            ));
            flags.push((flag.clone(), axis));
        }
        let compiled = Query::from_value(&query)?.compile(&keys)?;
        if !compiled.from.is_empty() {
            let from: Vec<String> = compiled
                .from
                .iter()
                .map(|m| format!("`{}` ({})", m.name, m.kind))
                .collect();
            why.insert(0, format!("from {}", from.join(", ")));
        }

        // 3. Candidates, derived tags, presets and flags.
        let universe: Vec<CatalogName> = match universe {
            Some(u) => u.to_vec(),
            None => stack
                .of_kind(EntryKind::Build)
                .iter()
                .map(|b| b.name().clone())
                .collect(),
        };
        let mut selection = Selection::new(
            universe
                .iter()
                .filter_map(|b| stack.resolve_build(b).ok().map(SelectionCandidate::new)),
        );
        for op in derive {
            selection = selection.derive(*op);
        }
        let mut ranking = Ranking {
            selection: Selection::new([]),
            cutoff: compiled.cutoff,
            limit: compiled.limit,
            query,
            why: Vec::new(),
            dropped: Vec::new(),
            flags,
            presets: request.presets.clone(),
        };
        if !ranking.presets.is_empty() || !ranking.flags.is_empty() {
            let mut dropped = Vec::new();
            for (c, _) in selection.iter() {
                let in_from =
                    compiled.from.is_empty() || compiled.from.iter().any(|m| m.contains(c));
                if in_from && let Err(reason) = ranking.presets_for(c) {
                    dropped.push(format!("`{}`: {reason}", c.name()));
                }
            }
            selection = selection.weigh_each(|c| {
                if ranking.presets_for(c).is_ok() {
                    1.0
                } else {
                    0.0
                }
            });
            if let Some(build) = &exact
                && !selection.iter().any(|(c, _)| c.name() == build)
            {
                return Err(selection_error(format!(
                    "the build `{build}` can't be chosen: {}",
                    dropped.join("; ")
                )));
            }
            ranking.dropped = dropped;
        }

        // 4. The query.
        ranking.selection = compiled.apply(selection);
        ranking.why = why;
        Ok(ranking)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    fn context() -> SelectionContext {
        SelectionContext {
            defaults: json!({"select": {"where": {"base": false}}, "order": ["-parameters"]}),
            named: vec![NamedSelection {
                name: "en".into(),
                description: "English speech.".into(),
                // Whisper has no `base` tag: the selection removes that
                // default.
                query: json!({
                    "select": {"from": ["whisper"], "where": {"base": null}},
                    "prefer": {"where": {"language": "en"}, "release": "latest"},
                    "order": ["-weight"],
                }),
            }],
        }
    }

    fn leaf(choice: &Choice) -> &str {
        choice.candidate.name().leaf()
    }

    #[test]
    fn test_defaults_apply_to_a_group() {
        let stack = fixtures::builtin_stack().unwrap();
        let choice = context()
            .choose(&stack, None, &[], &ChoiceRequest::default().name("qwen3"))
            .unwrap();
        assert_eq!(leaf(&choice), "qwen3-4b-thinking-2507");
        assert_eq!(choice.why[0], "from `qwen/qwen3` (namespace)");
    }

    #[test]
    fn test_one_build_skips_the_defaults() {
        let stack = fixtures::builtin_stack().unwrap();
        let choice = context()
            .choose(
                &stack,
                None,
                &[],
                &ChoiceRequest::default().name("qwen3-0.6b-base"),
            )
            .unwrap();
        assert_eq!(leaf(&choice), "qwen3-0.6b-base");
    }

    #[test]
    fn test_names_are_a_union() {
        let stack = fixtures::builtin_stack().unwrap();
        let ranking = SelectionContext::default()
            .rank(
                &stack,
                None,
                &[],
                &ChoiceRequest::default().name("qwen3-0.6b").name("large-v3"),
            )
            .unwrap();
        let leaves: Vec<&str> = ranking.selection.names().iter().map(|n| n.leaf()).collect();
        assert_eq!(leaves, ["qwen3-0.6b", "large-v3"]);
    }

    #[test]
    fn test_a_named_selection() {
        let stack = fixtures::builtin_stack().unwrap();
        let choice = context()
            .choose(&stack, None, &[], &ChoiceRequest::default().name("en"))
            .unwrap();
        assert_eq!(leaf(&choice), "large-v3");
        assert!(choice.why.contains(&"the selection `en`".to_string()));
    }

    #[test]
    fn test_flags_are_where_entries_and_presets() {
        let stack = fixtures::builtin_stack().unwrap();
        let choice = context()
            .choose(
                &stack,
                None,
                &[],
                &ChoiceRequest::default()
                    .name("qwen3")
                    .flag("thinking", false),
            )
            .unwrap();
        assert_eq!(leaf(&choice), "qwen3-0.6b");
        assert_eq!(choice.presets, PresetSelection::of(["no-thinking"]));
        assert!(
            choice
                .why
                .contains(&"`--off thinking` is `select.where.thinking: false`".into())
        );
        assert!(
            choice
                .why
                .contains(&"`--off thinking` implies `--preset no-thinking`".into())
        );
        let err = context()
            .choose(
                &stack,
                None,
                &[],
                &ChoiceRequest::default()
                    .name("qwen3-4b-thinking-2507")
                    .flag("thinking", false),
            )
            .unwrap_err();
        assert!(err.message.contains("fixed at `thinking`"), "{err}");
    }

    #[test]
    fn test_presets_narrow_with_reasons() {
        let stack = fixtures::builtin_stack().unwrap();
        let ranking = SelectionContext::default()
            .rank(
                &stack,
                None,
                &[],
                &ChoiceRequest::default().name("whisper").preset("translate"),
            )
            .unwrap();
        assert_eq!(ranking.selection.len(), 3);
        assert_eq!(ranking.dropped.len(), 1);
        assert!(ranking.dropped[0].contains("large-v3-turbo"));
    }

    #[test]
    fn test_choice_errors() {
        let stack = fixtures::builtin_stack().unwrap();
        let ctx = context();
        let err = |r: ChoiceRequest| ctx.choose(&stack, None, &[], &r).unwrap_err().message;
        assert!(err(ChoiceRequest::default().name("gemma")).contains("`gemma`"));
        assert!(
            err(ChoiceRequest::default().select(json!({"where": {"parameters": {"min": 1e12}}})))
                .contains("no build is left")
        );
        assert!(err(ChoiceRequest::default().flag("task", true)).contains("not a bool axis"));
    }

    #[test]
    fn test_the_result_is_cut_and_limited() {
        let stack = fixtures::builtin_stack().unwrap();
        let ctx = SelectionContext::default();
        let request = ChoiceRequest::default()
            .name("whisper")
            .prefer(json!({"release": {"step": 0.25}}))
            .order(&["-weight", "parameters"]);
        let leaves = |choices: Vec<Choice>| -> Vec<String> {
            choices
                .iter()
                .map(|c| c.candidate.name().leaf().to_string())
                .collect()
        };
        let all = ctx.choices(&stack, None, &[], &request).unwrap();
        assert_eq!(
            leaves(all),
            ["large-v3-turbo", "large-v3", "large-v2", "large-v1"]
        );
        let cut = request.clone().cutoff(0.8).limit(1);
        assert_eq!(
            leaves(ctx.choices(&stack, None, &[], &cut).unwrap()),
            ["large-v3-turbo"]
        );
        let ranking = ctx.rank(&stack, None, &[], &cut).unwrap();
        let reasons: Vec<Option<&str>> = ranking
            .selection
            .iter()
            .enumerate()
            .map(|(i, (_, w))| ranking.excluded(i, w))
            .collect();
        assert_eq!(
            reasons,
            [
                None,
                Some("past limit"),
                Some("below cutoff"),
                Some("below cutoff")
            ]
        );
        let every = ranking.report(true);
        assert_eq!(every.results.len(), 4);
        assert_eq!(every.results[1].excluded.as_deref(), Some("past limit"));
        let report = ranking.report(false);
        assert_eq!(report.results.len(), 1);
        let turbo = &report.results[0];
        assert_eq!(turbo.name, "openai/whisper/large/large-v3-turbo");
        assert_eq!(turbo.weight, 1.0);
        assert_eq!(turbo.excluded, None);
        assert_eq!(turbo.contract.as_deref(), Some("whisper-asr"));
        assert_eq!(turbo.tags["family"], json!("openai/whisper"));
        assert_eq!(turbo.dependencies.len(), 2);
        // One build, as loading takes it: the first, whatever the limit.
        let one = ctx.choose(&stack, None, &[], &request).unwrap();
        assert_eq!(leaf(&one), "large-v3-turbo");
    }

    #[test]
    fn test_derived_keys_can_be_tested_and_ordered() {
        let stack = fixtures::builtin_stack().unwrap();
        let size = crate::select::core::DeriveEach {
            keys: vec![DerivedKey {
                name: "est_size".into(),
                description: "An estimate, for the test.".into(),
            }],
            tags: |c: &SelectionCandidate| {
                let mut tags = std::collections::BTreeMap::new();
                if let Some(p) = c.number("parameters") {
                    tags.insert("est_size".to_string(), json!(p * 2.0));
                }
                tags
            },
        };
        let choice = SelectionContext::default()
            .choose(
                &stack,
                None,
                &[&size],
                &ChoiceRequest::default()
                    .select(json!({"where": {"est_size": {"max": 2e9}}}))
                    .order(&["-est_size"]),
            )
            .unwrap();
        assert_eq!(leaf(&choice), "large-v3-turbo");
    }
}
