//! A compiled query ([`CompiledQuery`]) and its run: each stage maps a
//! [`SelectTable`] to a table.
//!
//! **Tests**, each typed by what it tests:
//!
//! | written | tests |
//! |---|---|
//! | `qwen3` (a group), `hybrid` (a template class), `qwen3-0.6b` (a build) | membership: by closure, by chain, itself |
//! | `{qwen3: true}`, `{qwen3: false}` | in / not in that set |
//! | `thinking` (a bool axis), `{thinking: true}`, `{thinking: false}` | it offers the on / the off member |
//! | `{thinking: switch}` | the axis's implied tag: `always`, `switch` or `never` |
//! | `{task: translate}`, `{task: {in: [..]}}` (a categorical axis) | it offers that member / any of them |
//! | `instruct` (a bool tag), `{TAG: VALUE_TEST}` | the tag's value |
//! | `{family: qwen/qwen3}` (a facet) | its group in the facet |
//! | `{$presets: [P, ..]}` | it can take those presets together |
//! | `{$WORD: VALUE_TEST}` | a field or run value: `$weight`, `$name`, `$released`, `$license`, `$origin`, `$parent`, `$template_class`, `$contract`, or a derived `$KEY` |
//!
//! **Value tests**, typed by the tag's declaration; a value that doesn't fit
//! is an error:
//!
//! | written | for | means |
//! |---|---|---|
//! | `v` | any | equals; for a set-valued tag, contains |
//! | `{eq: v}`, `{ne: v}` | single-valued | equals / doesn't |
//! | `{lt: n}`, `{le: n}`, `{gt: n}`, `{ge: n}` | counts, times, `$weight`, derived keys | compares; several make a range |
//! | `{in: [..]}` | single-valued | is one of these |
//! | `{has: v}`, `{has_all: [..]}`, `{has_any: [..]}` | set-valued (`languages`, `text_list`, categorical offers) | contains |
//! | `{exists: true}` | any | has a value |
//!
//! A row without a value fails every test but `{exists: false}`. A
//! `languages` tag contains a language by RFC 4647 filtering (`en` contains
//! `en-US`).

use std::{
    cmp::Ordering,
    collections::BTreeSet,
};

use serde_json::Value;

use crate::{
    document::{
        AxisKind,
        CatalogEntry,
        EntryKind,
        TagType,
    },
    error::CatalogResult,
    lookup::LeafMatch,
    name::{
        CatalogName,
        CatalogRef,
    },
    presets::PresetSelection,
    release_time::ReleaseTime,
    resolve::CONTRACT_TAG,
    select::{
        predicate::{
            LanguageMatch,
            match_language,
        },
        query::{
            Predicate,
            SelectQuery,
            SortKey,
            Stage,
            flow,
            query_error,
        },
        state::{
            DroppedRow,
            SelectBuild,
            SelectEnv,
            SelectRun,
            SelectTable,
            SelectTrace,
            StageOrigin,
            StageTrace,
        },
    },
};

/// The kinds a membership name may name.
pub const MEMBERSHIP_KINDS: [EntryKind; 5] = [
    EntryKind::Namespace,
    EntryKind::Contract,
    EntryKind::Release,
    EntryKind::TemplateClass,
    EntryKind::Build,
];

/// The kinds a bare name or a test key may name.
const NAMED_KINDS: [EntryKind; 8] = [
    EntryKind::Namespace,
    EntryKind::Contract,
    EntryKind::Release,
    EntryKind::TemplateClass,
    EntryKind::Build,
    EntryKind::Tag,
    EntryKind::Axis,
    EntryKind::Facet,
];

/// A query with every name and test resolved: the stages it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledQuery {
    /// The kind it selects.
    pub kind: EntryKind,
    /// The least weight a row keeps between stages.
    pub floor: f64,
    /// The stages, in order.
    pub stages: Vec<CompiledStage>,
}

/// One compiled stage.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledStage {
    /// The operation.
    pub op: &'static str,
    /// Where it came from.
    pub origin: StageOrigin,
    /// Its label, when it has one.
    pub label: Option<String>,
    /// The stage as written, for messages.
    pub written: Value,
    step: Step,
}

#[derive(Clone, Debug, PartialEq)]
enum Step {
    /// Keeps the rows for which the test is `keep_if`.
    Filter {
        test: Test,
        keep_if: bool,
    },
    Order(Vec<CompiledSortKey>),
    Limit(usize),
}

/// A membership: a group, a template class or a build.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Member {
    name: CatalogName,
    kind: EntryKind,
}

impl Member {
    fn contains(
        &self,
        build: &SelectBuild,
    ) -> bool {
        let build = &build.build;
        match self.kind {
            EntryKind::Build => build.name == self.name,
            EntryKind::TemplateClass => build.template_chain.contains(&self.name),
            _ => build.memberships.contains(&self.name),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Test {
    All(Vec<Test>),
    Any(Vec<Test>),
    Not(Box<Test>),
    Member(Member),
    /// Offers the bool axis's on (`positive`) or off member.
    BoolAxis {
        axis: CatalogName,
        positive: bool,
    },
    Value {
        subject: Subject,
        check: ValueCheck,
    },
    Presets(Vec<String>),
}

#[derive(Clone, Debug, PartialEq)]
enum Subject {
    Tag(CatalogName),
    Derived(String),
    /// A categorical axis: what the build's use of it offers.
    Offers(CatalogName),
    /// A facet: the build's group in it.
    Facet(CatalogName),
    Weight,
    Name,
    Released,
    License,
    Origin,
    Parent,
    TemplateClass,
    Contract,
}

/// What a value is, for its tests.
#[derive(Clone, Debug, PartialEq)]
enum ValueKind {
    Bool,
    Number,
    Text,
    OneOf(Vec<String>),
    Time,
    /// A full name, compared as text; written operands are references.
    Name,
    TextSet,
    Languages,
    Offers(Vec<String>),
    /// A derived key: untyped.
    Any,
}

impl ValueKind {
    fn is_set(&self) -> bool {
        matches!(self, Self::TextSet | Self::Languages | Self::Offers(_))
    }

    fn is_ordered(&self) -> bool {
        matches!(self, Self::Number | Self::Time | Self::Any)
    }

    fn describe(&self) -> String {
        match self {
            Self::Bool => "`true` or `false`".into(),
            Self::Number => "a number".into(),
            Self::Text => "text".into(),
            Self::OneOf(choices) => format!("one of {choices:?}"),
            Self::Time => "a release time".into(),
            Self::Name => "a name".into(),
            Self::TextSet | Self::Languages => "text".into(),
            Self::Offers(members) => format!("one of the axis's members {members:?}"),
            Self::Any => "a value".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Cmp {
    Eq(Value),
    Ne(Value),
    Lt(Value),
    Le(Value),
    Gt(Value),
    Ge(Value),
    In(Vec<Value>),
    Has(Value),
    HasAll(Vec<Value>),
    HasAny(Vec<Value>),
    Exists(bool),
}

#[derive(Clone, Debug, PartialEq)]
struct ValueCheck {
    kind: ValueKind,
    cmps: Vec<Cmp>,
}

#[derive(Clone, Debug, PartialEq)]
struct CompiledSortKey {
    subject: Subject,
    kind: ValueKind,
    descending: bool,
    missing_first: bool,
}

/// What a test reads: the build, and the row's weight.
struct Row<'b> {
    build: &'b SelectBuild,
    weight: f64,
}

impl CompiledQuery {
    /// Compiles `query` against `env`: see [`SelectQuery::compile`].
    ///
    /// # Errors
    /// As [`SelectQuery::compile`].
    pub fn compile(
        query: &SelectQuery,
        env: &SelectEnv<'_>,
    ) -> CatalogResult<Self> {
        if query.kind != EntryKind::Build {
            return Err(query_error(format!(
                "a query selects builds: `kind: {}` isn't supported",
                query.kind
            )));
        }
        let compiler = Compiler { env };
        let stages = query
            .run
            .iter()
            .enumerate()
            .map(|(i, stage)| {
                compiler.stage(stage).map_err(|e| {
                    query_error(format!("stage {i} (`{}`): {}", stage.op(), e.message))
                })
            })
            .collect::<CatalogResult<Vec<_>>>()?;
        Ok(Self {
            kind: query.kind,
            floor: query.floor(),
            stages,
        })
    }

    /// Marks stage `index` as coming from `origin`.
    pub fn set_origin(
        &mut self,
        index: usize,
        origin: StageOrigin,
    ) {
        if let Some(stage) = self.stages.get_mut(index) {
            stage.origin = origin;
        }
    }

    /// Runs the query on `table`.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`](crate::CatalogErrorKind::Selection)
    /// when `table` holds another kind, and as
    /// [`SelectEnv::build`].
    pub fn run(
        &self,
        env: &SelectEnv<'_>,
        table: SelectTable,
    ) -> CatalogResult<SelectTable> {
        self.execute(env, table, None)
    }

    /// Runs the query on `table`, recording each stage and each row removed.
    ///
    /// # Errors
    /// As [`run`](Self::run).
    pub fn run_traced(
        &self,
        env: &SelectEnv<'_>,
        table: SelectTable,
    ) -> CatalogResult<SelectRun> {
        let mut trace = SelectTrace::default();
        let table = self.execute(env, table, Some(&mut trace))?;
        Ok(SelectRun { table, trace })
    }

    fn execute(
        &self,
        env: &SelectEnv<'_>,
        mut table: SelectTable,
        mut trace: Option<&mut SelectTrace>,
    ) -> CatalogResult<SelectTable> {
        if table.kind != self.kind {
            return Err(query_error(format!(
                "a query of `{}` entries can't run on a table of `{}` entries",
                self.kind, table.kind
            )));
        }
        for (index, stage) in self.stages.iter().enumerate() {
            let before: BTreeSet<CatalogName> =
                table.rows.iter().map(|r| r.entry.path.clone()).collect();
            let mut dropped = Vec::new();
            table.rows = match &stage.step {
                Step::Filter { test, keep_if } => {
                    let mut kept = Vec::new();
                    for row in table.rows {
                        let build = env.build(&row.entry.path)?;
                        let view = Row {
                            build: &build,
                            weight: row.weight,
                        };
                        if test.holds(&view) == *keep_if {
                            kept.push(row);
                        } else {
                            let why = explain(test, &view).unwrap_or_else(|| {
                                if *keep_if {
                                    format!("fails `{}: {}`", stage.op, flow(&stage.written))
                                } else {
                                    format!("matches `{}: {}`", stage.op, flow(&stage.written))
                                }
                            });
                            dropped.push((row, why));
                        }
                    }
                    kept
                }
                Step::Order(keys) => {
                    let mut keyed = Vec::new();
                    for row in table.rows {
                        let build = env.build(&row.entry.path)?;
                        let view = Row {
                            build: &build,
                            weight: row.weight,
                        };
                        let values: Vec<Option<Value>> =
                            keys.iter().map(|k| k.subject.read(&view)).collect();
                        keyed.push((values, row));
                    }
                    keyed.sort_by(|(a, _), (b, _)| compare_keys(keys, a, b));
                    keyed.into_iter().map(|(_, row)| row).collect()
                }
                Step::Limit(n) => {
                    let mut rows = table.rows;
                    let rest = rows.split_off((*n).min(rows.len()));
                    dropped.extend(rest.into_iter().map(|r| (r, "past limit".to_string())));
                    rows
                }
            };
            debug_assert!(
                table.rows.iter().all(|r| before.contains(&r.entry.path)),
                "no stage adds a row"
            );
            if let Some(trace) = trace.as_deref_mut() {
                trace.stages.push(StageTrace {
                    index,
                    op: stage.op,
                    origin: stage.origin.clone(),
                    label: stage.label.clone(),
                    dropped: dropped.len(),
                    weights: Vec::new(),
                });
                trace
                    .dropped
                    .extend(dropped.into_iter().map(|(row, why)| DroppedRow {
                        entry: row.entry,
                        stage: index,
                        weight: row.weight,
                        why,
                    }));
            }
        }
        Ok(table)
    }
}

/// A reason for a failed `$presets` test, from the preset check.
fn explain(
    test: &Test,
    row: &Row<'_>,
) -> Option<String> {
    match test {
        Test::Presets(presets) => row
            .build
            .build
            .presets
            .check_selection(&PresetSelection::of(presets.iter().cloned()))
            .err()
            .map(|e| e.message),
        Test::All(items) => items
            .iter()
            .find(|t| !t.holds(row))
            .and_then(|t| explain(t, row)),
        _ => None,
    }
}

fn compare_keys(
    keys: &[CompiledSortKey],
    a: &[Option<Value>],
    b: &[Option<Value>],
) -> Ordering {
    for ((key, a), b) in keys.iter().zip(a).zip(b) {
        let order = match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) if key.missing_first => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) if key.missing_first => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) => {
                let order = compare(&key.kind, a, b).unwrap_or(Ordering::Equal);
                if key.descending {
                    order.reverse()
                } else {
                    order
                }
            }
        };
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

/// `a` against `b`, as values of `kind`.
fn compare(
    kind: &ValueKind,
    a: &Value,
    b: &Value,
) -> Option<Ordering> {
    match kind {
        ValueKind::Time => {
            let a = ReleaseTime::parse(&time_text(a)?).ok()?;
            let b = ReleaseTime::parse(&time_text(b)?).ok()?;
            Some(a.cmp(&b))
        }
        _ => match (a, b) {
            (Value::Number(a), Value::Number(b)) => a.as_f64()?.partial_cmp(&b.as_f64()?),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            _ => None,
        },
    }
}

fn time_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

impl Test {
    fn holds(
        &self,
        row: &Row<'_>,
    ) -> bool {
        match self {
            Self::All(items) => items.iter().all(|t| t.holds(row)),
            Self::Any(items) => items.iter().any(|t| t.holds(row)),
            Self::Not(inner) => !inner.holds(row),
            Self::Member(member) => member.contains(row.build),
            Self::BoolAxis { axis, positive } => {
                let Some(using) = row.build.build.presets.axis_use(axis) else {
                    return false;
                };
                let on = axis.leaf();
                if *positive {
                    using.offers.iter().any(|o| o == on)
                } else {
                    using.offers.iter().any(|o| o != on)
                }
            }
            Self::Value { subject, check } => check.holds(subject.read(row).as_ref()),
            Self::Presets(presets) => row
                .build
                .build
                .presets
                .check_selection(&PresetSelection::of(presets.iter().cloned()))
                .is_ok(),
        }
    }
}

impl Subject {
    /// The value it reads from `row`, if it has one.
    fn read(
        &self,
        row: &Row<'_>,
    ) -> Option<Value> {
        let build = &row.build.build;
        match self {
            Self::Tag(name) => build.tags.get(name).map(|t| t.value.clone()),
            Self::Derived(key) => row.build.derived.get(key).cloned(),
            Self::Offers(axis) => build.presets.axis_use(axis).map(|using| {
                Value::Array(using.offers.iter().cloned().map(Value::String).collect())
            }),
            Self::Facet(facet) => build
                .facets
                .get(facet)
                .map(|g| Value::String(g.to_string())),
            Self::Weight => serde_json::Number::from_f64(row.weight).map(Value::Number),
            Self::Name => Some(Value::String(build.name.to_string())),
            Self::Released => build
                .release
                .as_ref()
                .map(|r| Value::String(r.released.canonical())),
            Self::License => build.license.clone().map(Value::String),
            Self::Origin => build.origin.clone().map(Value::String),
            Self::Parent => build.name.scope().map(|s| Value::String(s.to_string())),
            Self::TemplateClass => build
                .template_chain
                .last()
                .map(|c| Value::String(c.to_string())),
            Self::Contract => build
                .contract
                .as_ref()
                .map(|c| Value::String(c.to_string())),
        }
    }
}

impl ValueCheck {
    fn holds(
        &self,
        have: Option<&Value>,
    ) -> bool {
        self.cmps.iter().all(|cmp| {
            let Some(have) = have else {
                return matches!(cmp, Cmp::Exists(false));
            };
            let order = |want: &Value| compare(&self.kind, have, want);
            match cmp {
                Cmp::Exists(exists) => *exists,
                Cmp::Eq(want) => order(want) == Some(Ordering::Equal),
                Cmp::Ne(want) => order(want).is_some_and(|o| o != Ordering::Equal),
                Cmp::Lt(want) => order(want) == Some(Ordering::Less),
                Cmp::Le(want) => order(want).is_some_and(|o| o != Ordering::Greater),
                Cmp::Gt(want) => order(want) == Some(Ordering::Greater),
                Cmp::Ge(want) => order(want).is_some_and(|o| o != Ordering::Less),
                Cmp::In(any) => any.iter().any(|w| order(w) == Some(Ordering::Equal)),
                Cmp::Has(want) => self.contains(have, want),
                Cmp::HasAll(all) => all.iter().all(|w| self.contains(have, w)),
                Cmp::HasAny(any) => any.iter().any(|w| self.contains(have, w)),
            }
        })
    }

    fn contains(
        &self,
        have: &Value,
        want: &Value,
    ) -> bool {
        match self.kind {
            ValueKind::Languages => want
                .as_str()
                .is_some_and(|w| match_language(w, have) == LanguageMatch::Filtered),
            _ => have.as_array().is_some_and(|items| items.contains(want)),
        }
    }
}

/// What a test key or a name resolves to.
enum Resolved {
    Member(Member),
    BoolAxis(CatalogName),
    CategoricalAxis(CatalogName, Vec<String>),
    Tag(CatalogName, TagType),
    Facet(CatalogName),
}

struct Compiler<'e, 'a> {
    env: &'e SelectEnv<'a>,
}

impl Compiler<'_, '_> {
    fn stage(
        &self,
        stage: &Stage,
    ) -> CatalogResult<CompiledStage> {
        let written = serde_json::to_value(stage)
            .ok()
            .and_then(|v| v.get(stage.op()).cloned())
            .unwrap_or(Value::Null);
        let step = match stage {
            Stage::From(names) => Step::Filter {
                test: Test::Any(
                    names
                        .0
                        .iter()
                        .map(|n| self.member(n).map(Test::Member))
                        .collect::<CatalogResult<_>>()?,
                ),
                keep_if: true,
            },
            Stage::Keep(p) => Step::Filter {
                test: self.predicate(p)?,
                keep_if: true,
            },
            Stage::Drop(p) => Step::Filter {
                test: self.predicate(p)?,
                keep_if: false,
            },
            Stage::Order(keys) => Step::Order(
                keys.0
                    .iter()
                    .map(|k| self.sort_key(k))
                    .collect::<CatalogResult<_>>()?,
            ),
            Stage::Limit(n) => {
                if *n == 0 {
                    return Err(query_error("`limit` is a count, 1 or more"));
                }
                Step::Limit(*n)
            }
        };
        Ok(CompiledStage {
            op: stage.op(),
            origin: StageOrigin::Query,
            label: None,
            written,
            step,
        })
    }

    fn find(
        &self,
        name: &str,
        kinds: &[EntryKind],
    ) -> CatalogResult<CatalogName> {
        let stack = self.env.stack();
        match CatalogRef::parse(name) {
            Ok(r) if r.is_absolute() => stack
                .declaration(r.path())
                .filter(|d| kinds.contains(&d.kind()))
                .map(|d| d.name().clone())
                .ok_or_else(|| query_error(format!("nothing is declared as `{r}`"))),
            _ => stack
                .find(name, kinds, LeafMatch::Exact)
                .map_err(|e| query_error(format!("`{name}`: {}", e.message))),
        }
    }

    fn member(
        &self,
        name: &str,
    ) -> CatalogResult<Member> {
        let found = self.find(name, &MEMBERSHIP_KINDS)?;
        let kind = self
            .env
            .stack()
            .declaration(&found)
            .map(|d| d.kind())
            .expect("found");
        Ok(Member { name: found, kind })
    }

    fn resolve(
        &self,
        name: &str,
    ) -> CatalogResult<Resolved> {
        if name == CONTRACT_TAG {
            return Err(query_error(
                "the build's contract is the `$` word `$contract`, or a contract's name",
            ));
        }
        let found = self.find(name, &NAMED_KINDS)?;
        let decl = self.env.stack().declaration(&found).expect("found");
        Ok(match decl.body() {
            CatalogEntry::Axis(a) => match a.axis {
                AxisKind::Bool => Resolved::BoolAxis(found),
                AxisKind::Categorical => Resolved::CategoricalAxis(found, a.member_names()),
            },
            CatalogEntry::Tag(t) => Resolved::Tag(found, t.tag_type.clone()),
            CatalogEntry::Facet(_) => Resolved::Facet(found),
            _ => Resolved::Member(Member {
                name: found,
                kind: decl.kind(),
            }),
        })
    }

    fn predicate(
        &self,
        predicate: &Predicate,
    ) -> CatalogResult<Test> {
        Ok(match predicate {
            Predicate::All(items) => Test::All(self.predicates(items)?),
            Predicate::Any(items) => Test::Any(self.predicates(items)?),
            Predicate::Not(inner) => Test::Not(Box::new(self.predicate(inner)?)),
            Predicate::Name(name) => {
                if name.starts_with('$') {
                    return Err(query_error(format!(
                        "`{name}` is a `$` word, which is tested as `{{{name}: ..}}`"
                    )));
                }
                match self.resolve(name)? {
                    Resolved::Member(member) => Test::Member(member),
                    Resolved::BoolAxis(axis) => Test::BoolAxis {
                        axis,
                        positive: true,
                    },
                    Resolved::Tag(tag, TagType::Bool) => Test::Value {
                        subject: Subject::Tag(tag),
                        check: ValueCheck {
                            kind: ValueKind::Bool,
                            cmps: vec![Cmp::Eq(Value::Bool(true))],
                        },
                    },
                    Resolved::Tag(tag, _) => {
                        return Err(query_error(format!(
                            "`{tag}` is a valued tag: test it, as `{{{name}: {{exists: true}}}}` \
                             or with a value"
                        )));
                    }
                    Resolved::CategoricalAxis(axis, _) => {
                        return Err(query_error(format!(
                            "`{axis}` is a categorical axis: name a member, as `{{{name}: M}}`"
                        )));
                    }
                    Resolved::Facet(facet) => {
                        return Err(query_error(format!(
                            "`{facet}` is a facet: name a group of it, as `{{{name}: G}}`"
                        )));
                    }
                }
            }
            Predicate::Test { key, value } => self
                .test(key, value)
                .map_err(|e| query_error(format!("`{key}`: {}", e.message)))?,
        })
    }

    fn predicates(
        &self,
        items: &[Predicate],
    ) -> CatalogResult<Vec<Test>> {
        items.iter().map(|p| self.predicate(p)).collect()
    }

    fn test(
        &self,
        key: &str,
        value: &Value,
    ) -> CatalogResult<Test> {
        if let Some(word) = key.strip_prefix('$') {
            return self.word_test(word, value);
        }
        Ok(match self.resolve(key)? {
            Resolved::Member(member) => match value {
                Value::Bool(true) => Test::Member(member),
                Value::Bool(false) => Test::Not(Box::new(Test::Member(member))),
                other => {
                    return Err(query_error(format!(
                        "`{}` is a set: it takes `true` or `false`, not {other}",
                        member.name
                    )));
                }
            },
            Resolved::BoolAxis(axis) => match value {
                Value::Bool(positive) => Test::BoolAxis {
                    axis,
                    positive: *positive,
                },
                // The axis's implied tag: always, switch or never.
                other => Test::Value {
                    subject: Subject::Tag(axis),
                    check: self.check(
                        ValueKind::OneOf(vec!["always".into(), "switch".into(), "never".into()]),
                        other,
                    )?,
                },
            },
            Resolved::CategoricalAxis(axis, members) => Test::Value {
                subject: Subject::Offers(axis),
                check: self.check(ValueKind::Offers(members), value)?,
            },
            Resolved::Tag(tag, tag_type) => {
                let kind = match tag_type {
                    TagType::Bool => ValueKind::Bool,
                    TagType::Count => ValueKind::Number,
                    TagType::Text => ValueKind::Text,
                    TagType::OneOf(choices) => ValueKind::OneOf(choices),
                    TagType::TextList => ValueKind::TextSet,
                    TagType::Languages => ValueKind::Languages,
                    TagType::Scores => {
                        return Err(query_error("a `scores` tag can't be tested"));
                    }
                };
                Test::Value {
                    subject: Subject::Tag(tag),
                    check: self.check(kind, value)?,
                }
            }
            Resolved::Facet(facet) => {
                let check = self.check(ValueKind::Name, value)?;
                Test::Value {
                    subject: Subject::Facet(facet),
                    check: self.names(check, &[EntryKind::Namespace, EntryKind::Contract])?,
                }
            }
        })
    }

    fn word_test(
        &self,
        word: &str,
        value: &Value,
    ) -> CatalogResult<Test> {
        let (subject, kind, refs): (Subject, ValueKind, &[EntryKind]) = match word {
            "presets" => {
                let presets = match value {
                    Value::String(one) => vec![one.clone()],
                    Value::Array(items) => items
                        .iter()
                        .map(|v| {
                            v.as_str()
                                .map(str::to_string)
                                .ok_or_else(|| query_error(format!("{v} is not a preset name")))
                        })
                        .collect::<CatalogResult<_>>()?,
                    other => {
                        return Err(query_error(format!(
                            "`$presets` takes a preset name or a list of them, not {other}"
                        )));
                    }
                };
                return Ok(Test::Presets(presets));
            }
            "weight" => (Subject::Weight, ValueKind::Number, &[]),
            "name" => (Subject::Name, ValueKind::Text, &[]),
            "released" => (Subject::Released, ValueKind::Time, &[]),
            "license" => (Subject::License, ValueKind::Text, &[]),
            "origin" => (Subject::Origin, ValueKind::Text, &[]),
            "parent" => (
                Subject::Parent,
                ValueKind::Name,
                &[EntryKind::Namespace, EntryKind::Contract],
            ),
            "template_class" => (
                Subject::TemplateClass,
                ValueKind::Name,
                &[EntryKind::TemplateClass],
            ),
            "contract" => (Subject::Contract, ValueKind::Name, &[EntryKind::Contract]),
            "format" | "dtype" => {
                return Err(query_error(format!(
                    "`${word}` is a resource's field; a build has none"
                )));
            }
            "any" | "all" | "not" => {
                return Err(query_error(format!("`${word}` is an operator, not a test")));
            }
            derived if self.env.derived_keys().iter().any(|k| k.name == derived) => {
                (Subject::Derived(derived.to_string()), ValueKind::Any, &[])
            }
            other => {
                return Err(query_error(format!(
                    "`${other}` is not a `$` word: the words are `$any`, `$all`, `$not`, \
                     `$presets`, `$weight`, `$name`, `$released`, `$license`, `$origin`, \
                     `$parent`, `$template_class`, `$contract`, and the derived keys"
                )));
            }
        };
        let mut check = self.check(kind, value)?;
        if !refs.is_empty() {
            check = self.names(check, refs)?;
        }
        Ok(Test::Value { subject, check })
    }

    /// `check` with each operand, a reference, resolved to a full name.
    fn names(
        &self,
        mut check: ValueCheck,
        kinds: &[EntryKind],
    ) -> CatalogResult<ValueCheck> {
        let resolve = |v: &mut Value| -> CatalogResult<()> {
            let name = v
                .as_str()
                .ok_or_else(|| query_error(format!("{v} is not a name")))?;
            *v = Value::String(self.find(name, kinds)?.to_string());
            Ok(())
        };
        for cmp in &mut check.cmps {
            match cmp {
                Cmp::Eq(v) | Cmp::Ne(v) => resolve(v)?,
                Cmp::In(items) => items.iter_mut().try_for_each(resolve)?,
                _ => {}
            }
        }
        Ok(check)
    }

    /// The value test `written`, checked against `kind`.
    fn check(
        &self,
        kind: ValueKind,
        written: &Value,
    ) -> CatalogResult<ValueCheck> {
        let fits = |v: &Value| -> CatalogResult<()> {
            let ok = match &kind {
                ValueKind::Bool => v.is_boolean(),
                ValueKind::Number => v.is_number(),
                ValueKind::Text | ValueKind::Name | ValueKind::TextSet | ValueKind::Languages => {
                    v.is_string()
                }
                ValueKind::OneOf(choices) | ValueKind::Offers(choices) => {
                    v.as_str().is_some_and(|s| choices.iter().any(|c| c == s))
                }
                ValueKind::Time => time_text(v).is_some_and(|t| ReleaseTime::parse(&t).is_ok()),
                ValueKind::Any => true,
            };
            if ok {
                Ok(())
            } else {
                Err(query_error(format!("{v} is not {}", kind.describe())))
            }
        };
        let list = |op: &str, v: &Value| -> CatalogResult<Vec<Value>> {
            let Value::Array(items) = v else {
                return Err(query_error(format!("`{op}` takes a list, not {v}")));
            };
            items.iter().try_for_each(&fits)?;
            Ok(items.clone())
        };
        let cmps = match written {
            Value::Object(ops) => {
                let mut cmps = Vec::new();
                for (op, v) in ops {
                    let scalar_only = |cmp: Cmp| -> CatalogResult<Cmp> {
                        if kind.is_set() {
                            Err(query_error(format!(
                                "`{op}` tests a single value; a set takes `has`, `has_all`, \
                                 `has_any` or `exists`"
                            )))
                        } else {
                            Ok(cmp)
                        }
                    };
                    let ordered = |cmp: Cmp| -> CatalogResult<Cmp> {
                        if kind.is_ordered() {
                            Ok(cmp)
                        } else {
                            Err(query_error(format!(
                                "`{op}` compares, and {} is not ordered",
                                kind.describe()
                            )))
                        }
                    };
                    let set_only = |cmp: Cmp| -> CatalogResult<Cmp> {
                        if kind.is_set() {
                            Ok(cmp)
                        } else {
                            Err(query_error(format!(
                                "`{op}` tests a set, and this takes a single value"
                            )))
                        }
                    };
                    cmps.push(match op.as_str() {
                        "exists" => match v {
                            Value::Bool(b) => Cmp::Exists(*b),
                            other => {
                                return Err(query_error(format!(
                                    "`exists` takes `true` or `false`, not {other}"
                                )));
                            }
                        },
                        "eq" => {
                            fits(v)?;
                            scalar_only(Cmp::Eq(v.clone()))?
                        }
                        "ne" => {
                            fits(v)?;
                            scalar_only(Cmp::Ne(v.clone()))?
                        }
                        "lt" | "le" | "gt" | "ge" => {
                            fits(v)?;
                            let v = v.clone();
                            ordered(scalar_only(match op.as_str() {
                                "lt" => Cmp::Lt(v),
                                "le" => Cmp::Le(v),
                                "gt" => Cmp::Gt(v),
                                _ => Cmp::Ge(v),
                            })?)?
                        }
                        "in" => scalar_only(Cmp::In(list("in", v)?))?,
                        "has" => {
                            fits(v)?;
                            set_only(Cmp::Has(v.clone()))?
                        }
                        "has_all" => set_only(Cmp::HasAll(list("has_all", v)?))?,
                        "has_any" => set_only(Cmp::HasAny(list("has_any", v)?))?,
                        other => {
                            return Err(query_error(format!(
                                "`{other}` is not a value test: the tests are `eq`, `ne`, `lt`, \
                                 `le`, `gt`, `ge`, `in`, `has`, `has_all`, `has_any` and \
                                 `exists`"
                            )));
                        }
                    });
                }
                if cmps.is_empty() {
                    return Err(query_error("an empty map is not a value test"));
                }
                cmps
            }
            Value::Array(_) => {
                return Err(query_error(
                    "a list is not a value test: write `{in: [..]}` for any of them, or \
                     `{has_all: [..]}` for a set holding all of them",
                ));
            }
            Value::Null => {
                return Err(query_error(
                    "`null` is not a value test: write `{exists: false}`",
                ));
            }
            scalar => {
                fits(scalar)?;
                if kind.is_set() {
                    vec![Cmp::Has(scalar.clone())]
                } else {
                    vec![Cmp::Eq(scalar.clone())]
                }
            }
        };
        Ok(ValueCheck { kind, cmps })
    }

    fn sort_key(
        &self,
        key: &SortKey,
    ) -> CatalogResult<CompiledSortKey> {
        let (subject, kind) = match key.key.strip_prefix('$') {
            Some("weight") => (Subject::Weight, ValueKind::Number),
            Some("name") => (Subject::Name, ValueKind::Text),
            Some("released") => (Subject::Released, ValueKind::Time),
            Some(derived) if self.env.derived_keys().iter().any(|k| k.name == derived) => {
                (Subject::Derived(derived.to_string()), ValueKind::Any)
            }
            Some(other) => {
                return Err(query_error(format!(
                    "`${other}` is not a sort key: sort by `$weight`, `$name`, `$released`, a \
                     count tag or a derived key"
                )));
            }
            None => match self.resolve(&key.key)? {
                Resolved::Tag(tag, TagType::Count) => (Subject::Tag(tag), ValueKind::Number),
                _ => {
                    return Err(query_error(format!(
                        "`{}` is not ordered: sort by `$weight`, `$name`, `$released`, a count \
                         tag or a derived key",
                        key.key
                    )));
                }
            },
        };
        Ok(CompiledSortKey {
            subject,
            kind,
            descending: key.descending,
            missing_first: key.missing_first,
        })
    }
}
