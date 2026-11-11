//! A query as written: [`SelectQuery`], a kind and a list of stages, made
//! only of YAML particles.
//!
//! ```yaml
//! kind: build
//! run:
//!   - from: [qwen3, whisper]
//!   - keep: {instruct: true, thinking: false, parameters: {le: 8.9e9}}
//!   - order: [{desc: parameters}, {asc: $name}]
//!   - limit: 1
//! ```
//!
//! **No string is ever parsed.** A string is a name (a reference: `qwen3`,
//! `a/b`, `/root`), a `$` word, or a literal value in a value position; a
//! map is an operator from a closed vocabulary, or a test keyed by a name or
//! a `$` word. Wherever a catalog name can appear, anything that isn't one
//! starts with `$`.
//!
//! | stage | argument | does |
//! |---|---|---|
//! | `from` | a name or a list of names | keeps the rows in any of them |
//! | `keep` | a [`Predicate`] | keeps the rows that satisfy it |
//! | `drop` | a predicate | `keep: {$not: P}` |
//! | `weigh` | a [`WeighBlock`] | updates each row's weight |
//! | `order` | a [`SortKey`] or a list of them | sorts, stably |
//! | `limit` | an integer, 1 or more | keeps the first N |
//!
//! The query compiles against a [`SelectEnv`]
//! ([`compile`](SelectQuery::compile)), which checks every name and test
//! against the query's kind, and runs ([`CompiledQuery`]).

use serde::{
    Deserialize,
    Deserializer,
    Serialize,
    Serializer,
    de,
};
use serde_json::{
    Map,
    Value,
};

use crate::{
    document::EntryKind,
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogResult,
    },
    select::{
        run::CompiledQuery,
        state::SelectEnv,
    },
};

pub(crate) fn query_error(message: impl Into<String>) -> CatalogError {
    CatalogError::new(CatalogErrorKind::Selection, message)
}

/// The least weight a row keeps between stages, unless a query sets its
/// `floor`.
pub const DEFAULT_FLOOR: f64 = 1e-9;

/// A query: a kind, an optional floor, and the stages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectQuery {
    /// The kind of entry it selects. The run starts from every entry of the
    /// kind, and the kind decides which tests the stages may use.
    pub kind: EntryKind,

    /// The least weight a row keeps between stages, 0 or more;
    /// [`DEFAULT_FLOOR`] when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor: Option<f64>,

    /// The stages, in order.
    #[serde(default)]
    pub run: Vec<Stage>,
}

impl SelectQuery {
    /// A query of `kind` running `run`.
    pub fn new(
        kind: EntryKind,
        run: Vec<Stage>,
    ) -> Self {
        Self {
            kind,
            floor: None,
            run,
        }
    }

    /// The query `value` writes.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] when it isn't a query.
    pub fn from_value(value: &Value) -> CatalogResult<Self> {
        let query: Self = serde_json::from_value(value.clone())
            .map_err(|e| query_error(format!("a query: {e}")))?;
        if let Some(floor) = query.floor
            && !(floor.is_finite() && floor >= 0.0)
        {
            return Err(query_error(format!(
                "`floor` is a weight, 0 or more, not {floor}"
            )));
        }
        Ok(query)
    }

    /// The query `yaml` writes.
    ///
    /// # Errors
    /// As [`from_value`](Self::from_value), and when it isn't YAML.
    pub fn from_yaml(yaml: &str) -> CatalogResult<Self> {
        let value: Value = serde_saphyr::from_str(yaml)
            .map_err(|e| query_error(format!("a query is YAML: {e}")))?;
        Self::from_value(&value)
    }

    /// The floor the run keeps rows at.
    pub fn floor(&self) -> f64 {
        self.floor.unwrap_or(DEFAULT_FLOOR)
    }

    /// Resolves every name and test against `env`, for the query's kind.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] for a name or key that resolves to
    /// nothing (or to something it can't be), a test the kind doesn't admit,
    /// or a value that doesn't fit its declaration.
    pub fn compile(
        &self,
        env: &SelectEnv<'_>,
    ) -> CatalogResult<CompiledQuery> {
        CompiledQuery::compile(self, env)
    }
}

/// One stage: a single-key map, the operation and its argument.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Stage {
    /// Keeps the rows in any of the names: `keep: {$any: [..]}`.
    From(NameList),
    /// Keeps the rows that satisfy the predicate.
    Keep(Predicate),
    /// Drops the rows that satisfy the predicate: `keep: {$not: P}`.
    Drop(Predicate),
    /// Updates each row's weight.
    Weigh(WeighBlock),
    /// Sorts the rows, stably, by the keys, most significant first.
    Order(SortKeys),
    /// Keeps the first N rows.
    Limit(usize),
}

impl Stage {
    /// The operation's name.
    pub fn op(&self) -> &'static str {
        match self {
            Self::From(_) => "from",
            Self::Keep(_) => "keep",
            Self::Drop(_) => "drop",
            Self::Weigh(_) => "weigh",
            Self::Order(_) => "order",
            Self::Limit(_) => "limit",
        }
    }
}

/// A `weigh` stage: a block of updates, `{by: [..], floor?: f}`.
///
/// 1. It starts from each row's running weight.
/// 2. It applies each `by` entry in order, as signed arithmetic, with nothing
///    clipped in between.
/// 3. It commits the result, clipped to 0 or more.
/// 4. It drops the row if the committed weight is below `floor`, when one is
///    given. The query's floor applies after it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeighBlock {
    /// The updates, in order.
    #[serde(default)]
    pub by: Vec<WeighEntry>,
    /// The block's own floor, 0 or more.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor: Option<f64>,
}

/// One update in a [`WeighBlock`].
///
/// | entry | does to the weight |
/// |---|---|
/// | `{when: P, else: w, then: v}` | × `v` (default 1) if the row satisfies P, else × `w` (default 0) |
/// | `{match: {TAG: L}, truncated: t, else: w}` | RFC 4647 on a `languages` tag: × 1 when `L` filters to one of its languages, × `t` (default 1) when only a truncation of `L` does, else × `w` (default 0) |
/// | `{release: latest, step: s, min: m}` | × 1 for the newest of its release group in the catalog, `s` (default 0.05) less for each newer one, never below `m` (default 0); × 1 in no release group |
/// | `{mul: k}` | × `k` |
/// | `{add: k}` | + `k` |
/// | `{clip: [a, b]}` | clamps to `[a, b]` |
///
/// `when`, `match`, `release` and `mul` multiply; `add` and `clip` read the
/// running weight, so their place matters.
#[derive(Clone, Debug, PartialEq)]
pub enum WeighEntry {
    /// `{when: P, else: w, then: v}`.
    When {
        /// The predicate.
        when: Predicate,
        /// The factor when it holds.
        then: f64,
        /// The factor when it doesn't.
        otherwise: f64,
    },
    /// `{match: {TAG: L}, truncated: t, else: w}`.
    Match {
        /// The `languages` tag.
        tag: String,
        /// The requested language.
        language: String,
        /// The factor for a match through a truncation.
        truncated: f64,
        /// The factor otherwise.
        otherwise: f64,
    },
    /// `{release: latest, step: s, min: m}`.
    Release {
        /// The factor lost per newer release.
        step: f64,
        /// The least factor.
        min: f64,
    },
    /// `{mul: k}`.
    Mul(f64),
    /// `{add: k}`.
    Add(f64),
    /// `{clip: [a, b]}`.
    Clip(f64, f64),
}

/// The default `step` of `{release: latest}`.
pub const RELEASE_STEP: f64 = 0.05;

impl WeighEntry {
    /// Whether it multiplies (and so commutes with other multiplying entries
    /// and with fixed tests).
    pub fn multiplies(&self) -> bool {
        !matches!(self, Self::Add(_) | Self::Clip(..))
    }

    fn from_value(value: &Value) -> CatalogResult<Self> {
        let Value::Object(fields) = value else {
            return Err(query_error(format!("a `by` entry is a map, not {value}")));
        };
        let number = |key: &str, default: f64| -> CatalogResult<f64> {
            match fields.get(key) {
                None => Ok(default),
                Some(v) => v
                    .as_f64()
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| query_error(format!("`{key}` is a number, not {v}"))),
            }
        };
        let only = |allowed: &[&str]| -> CatalogResult<()> {
            match fields.keys().find(|k| !allowed.contains(&k.as_str())) {
                Some(k) => Err(query_error(format!(
                    "this `by` entry takes {allowed:?}, not `{k}`"
                ))),
                None => Ok(()),
            }
        };
        if let Some(when) = fields.get("when") {
            only(&["when", "then", "else"])?;
            return Ok(Self::When {
                when: Predicate::from_value(when)?,
                then: number("then", 1.0)?,
                otherwise: number("else", 0.0)?,
            });
        }
        if let Some(matched) = fields.get("match") {
            only(&["match", "truncated", "else"])?;
            let (tag, language) = match matched {
                Value::Object(m) if m.len() == 1 => {
                    let (tag, language) = m.iter().next().expect("one");
                    let language = language.as_str().ok_or_else(|| {
                        query_error(format!("`match.{tag}` is a language, not {language}"))
                    })?;
                    (tag.clone(), language.to_string())
                }
                other => {
                    return Err(query_error(format!(
                        "`match` is `{{TAG: LANGUAGE}}`, not {other}"
                    )));
                }
            };
            return Ok(Self::Match {
                tag,
                language,
                truncated: number("truncated", 1.0)?,
                otherwise: number("else", 0.0)?,
            });
        }
        if let Some(release) = fields.get("release") {
            only(&["release", "step", "min"])?;
            if release.as_str() != Some("latest") {
                return Err(query_error(format!("`release` is `latest`, not {release}")));
            }
            let step = number("step", RELEASE_STEP)?;
            let min = number("min", 0.0)?;
            if step < 0.0 || min < 0.0 {
                return Err(query_error("`release`'s `step` and `min` are 0 or more"));
            }
            return Ok(Self::Release { step, min });
        }
        if fields.contains_key("mul") {
            only(&["mul"])?;
            return Ok(Self::Mul(number("mul", 1.0)?));
        }
        if fields.contains_key("add") {
            only(&["add"])?;
            return Ok(Self::Add(number("add", 0.0)?));
        }
        if let Some(clip) = fields.get("clip") {
            only(&["clip"])?;
            let bounds: Option<Vec<f64>> = clip
                .as_array()
                .map(|items| items.iter().filter_map(Value::as_f64).collect());
            return match bounds.as_deref() {
                Some([a, b]) if a <= b && a.is_finite() && b.is_finite() => Ok(Self::Clip(*a, *b)),
                _ => Err(query_error(format!(
                    "`clip` is `[a, b]`, with `a` at most `b`, not {clip}"
                ))),
            };
        }
        Err(query_error(format!(
            "a `by` entry is `when`, `match`, `release`, `mul`, `add` or `clip`, not {}",
            flow(value)
        )))
    }

    fn to_value(&self) -> Value {
        let mut map = Map::new();
        let mut put = |k: &str, v: Value| {
            map.insert(k.to_string(), v);
        };
        match self {
            Self::When {
                when,
                then,
                otherwise,
            } => {
                put("when", when.to_value());
                put("then", (*then).into());
                put("else", (*otherwise).into());
            }
            Self::Match {
                tag,
                language,
                truncated,
                otherwise,
            } => {
                put("match", single(tag, Value::String(language.clone())));
                put("truncated", (*truncated).into());
                put("else", (*otherwise).into());
            }
            Self::Release { step, min } => {
                put("release", "latest".into());
                put("step", (*step).into());
                put("min", (*min).into());
            }
            Self::Mul(k) => put("mul", (*k).into()),
            Self::Add(k) => put("add", (*k).into()),
            Self::Clip(a, b) => put("clip", vec![*a, *b].into()),
        }
        Value::Object(map)
    }
}

impl Serialize for WeighEntry {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WeighEntry {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(&value).map_err(|e| de::Error::custom(e.message))
    }
}

/// One name, or a list of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameList(pub Vec<String>);

impl Serialize for NameList {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match self.0.as_slice() {
            [one] => serializer.serialize_str(one),
            many => many.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for NameList {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum OneOrMany {
            One(String),
            Many(Vec<String>),
        }
        Ok(match OneOrMany::deserialize(deserializer)? {
            OneOrMany::One(one) => NameList(vec![one]),
            OneOrMany::Many(many) => NameList(many),
        })
    }
}

/// A predicate as written.
///
/// | written | means |
/// |---|---|
/// | `hybrid` | in the set the name denotes ([`Name`](Self::Name)) |
/// | `[a, b, ..]` | all of them |
/// | `{$any: [..]}`, `{$all: [..]}`, `{$not: P}` | the operators |
/// | `{KEY: VALUE}` | a test of `KEY`, a name or a `$` word ([`Test`](Self::Test)) |
/// | a map with several keys | all of its entries |
#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    /// A name: a group (by closure), a template class (by chain), a build
    /// (itself), a bool axis (offers its positive member), or a bool tag
    /// (`true`).
    Name(String),
    /// All of these.
    All(Vec<Predicate>),
    /// Any of these.
    Any(Vec<Predicate>),
    /// Not this.
    Not(Box<Predicate>),
    /// A test of `key` (a name or a `$` word) by `value`.
    Test {
        /// What is tested.
        key: String,
        /// The value, or a value test, as written.
        value: Value,
    },
}

impl Predicate {
    /// The predicate `value` writes.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] for a shape no predicate has.
    pub fn from_value(value: &Value) -> CatalogResult<Self> {
        match value {
            Value::String(name) => Ok(Self::Name(name.clone())),
            Value::Array(items) => Ok(Self::All(
                items
                    .iter()
                    .map(Self::from_value)
                    .collect::<CatalogResult<_>>()?,
            )),
            Value::Object(entries) => {
                let mut all = entries
                    .iter()
                    .map(|(key, value)| Self::entry(key, value))
                    .collect::<CatalogResult<Vec<_>>>()?;
                match all.len() {
                    0 => Err(query_error("an empty map is not a predicate")),
                    1 => Ok(all.pop().expect("one")),
                    _ => Ok(Self::All(all)),
                }
            }
            other => Err(query_error(format!(
                "a predicate is a name, a list or a map, not {other}"
            ))),
        }
    }

    fn entry(
        key: &str,
        value: &Value,
    ) -> CatalogResult<Self> {
        let operands = |op: &str| -> CatalogResult<Vec<Predicate>> {
            let Value::Array(items) = value else {
                return Err(query_error(format!("`{op}` takes a list, not {value}")));
            };
            items.iter().map(Self::from_value).collect()
        };
        Ok(match key {
            "$any" => Self::Any(operands("$any")?),
            "$all" => Self::All(operands("$all")?),
            "$not" => Self::Not(Box::new(Self::from_value(value)?)),
            _ => Self::Test {
                key: key.to_string(),
                value: value.clone(),
            },
        })
    }

    /// The predicate as written.
    pub fn to_value(&self) -> Value {
        match self {
            Self::Name(name) => Value::String(name.clone()),
            Self::All(items) => Value::Array(items.iter().map(Self::to_value).collect()),
            Self::Any(items) => single(
                "$any",
                Value::Array(items.iter().map(Self::to_value).collect()),
            ),
            Self::Not(inner) => single("$not", inner.to_value()),
            Self::Test { key, value } => single(key, value.clone()),
        }
    }
}

fn single(
    key: &str,
    value: Value,
) -> Value {
    let mut map = Map::new();
    map.insert(key.to_string(), value);
    Value::Object(map)
}

impl Serialize for Predicate {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Predicate {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(&value).map_err(|e| de::Error::custom(e.message))
    }
}

/// One sort key: `K` (ascending), `{asc: K}`, `{desc: K}`, with an optional
/// `missing: first` (rows without a value first; the default is last).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SortKey {
    /// The key: `$weight`, `$name`, `$released`, an ordered tag, or a
    /// derived `$` key.
    pub key: String,
    /// Whether it sorts descending.
    pub descending: bool,
    /// Whether rows without a value go first.
    pub missing_first: bool,
}

impl SortKey {
    /// `key`, ascending.
    pub fn asc(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            descending: false,
            missing_first: false,
        }
    }

    /// `key`, descending.
    pub fn desc(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            descending: true,
            missing_first: false,
        }
    }

    fn from_value(value: &Value) -> CatalogResult<Self> {
        match value {
            Value::String(key) => Ok(Self::asc(key.clone())),
            Value::Object(fields) => {
                let mut key = None;
                let mut descending = false;
                let mut missing_first = false;
                for (k, v) in fields {
                    match (k.as_str(), v) {
                        ("asc" | "desc", Value::String(s)) if key.is_none() => {
                            key = Some(s.clone());
                            descending = k == "desc";
                        }
                        ("missing", Value::String(m)) if m == "first" || m == "last" => {
                            missing_first = m == "first";
                        }
                        _ => {
                            return Err(query_error(format!(
                                "a sort key is `K`, `{{asc: K}}` or `{{desc: K}}`, with an \
                                 optional `missing: first | last`; not `{k}: {v}`"
                            )));
                        }
                    }
                }
                let key = key.ok_or_else(|| {
                    query_error("a sort key map names its key with `asc` or `desc`")
                })?;
                Ok(Self {
                    key,
                    descending,
                    missing_first,
                })
            }
            other => Err(query_error(format!(
                "a sort key is a key or a map, not {other}"
            ))),
        }
    }

    fn to_value(&self) -> Value {
        if !self.descending && !self.missing_first {
            return Value::String(self.key.clone());
        }
        let mut map = Map::new();
        let direction = if self.descending { "desc" } else { "asc" };
        map.insert(direction.to_string(), Value::String(self.key.clone()));
        if self.missing_first {
            map.insert("missing".to_string(), Value::String("first".to_string()));
        }
        Value::Object(map)
    }
}

/// One sort key, or a list of them, most significant first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SortKeys(pub Vec<SortKey>);

impl Serialize for SortKeys {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match self.0.as_slice() {
            [one] => one.to_value().serialize(serializer),
            many => {
                Value::Array(many.iter().map(SortKey::to_value).collect()).serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for SortKeys {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let keys = match &value {
            Value::Array(items) => items
                .iter()
                .map(SortKey::from_value)
                .collect::<CatalogResult<Vec<_>>>(),
            one => SortKey::from_value(one).map(|k| vec![k]),
        };
        keys.map(SortKeys).map_err(|e| de::Error::custom(e.message))
    }
}

/// `value` as YAML flow, for messages.
pub fn flow(value: &Value) -> String {
    match value {
        Value::String(s) => {
            let plain = s
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | '$'))
                && !s.is_empty();
            if plain {
                s.clone()
            } else {
                Value::String(s.clone()).to_string()
            }
        }
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(flow).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(k, v)| format!("{}: {}", flow(&Value::String(k.clone())), flow(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}
