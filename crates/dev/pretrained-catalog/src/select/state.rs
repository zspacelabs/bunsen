//! The state a query runs on: a [`SelectTable`] of weighted rows, the
//! [`SelectEnv`] its tests read entries through, and the [`SelectTrace`] a
//! traced run records.
//!
//! A stage maps a table to a table. The run starts from the universe of the
//! query's kind (or the caller's universe), in full-name order, each row at
//! weight 1, and its result is the last table. **No stage adds a row**: each
//! stage's output rows are a subset of its input's, so a property every row
//! has after one stage holds for every row of the result.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    rc::Rc,
};

use serde::Serialize;
use serde_json::Value;

use crate::{
    document::EntryKind,
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogResult,
    },
    name::CatalogName,
    resolve::ResolvedBuild,
    stack::PretrainedCatalogStack,
};

/// An entry, by its full name and its kind.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct KindRef {
    /// The full name.
    pub path: CatalogName,
    /// The kind.
    pub kind: EntryKind,
}

/// One row: an entry and its running weight.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SelectRow {
    /// The entry.
    pub entry: KindRef,
    /// Its weight, 0 or more.
    pub weight: f64,
}

/// Rows of one kind, in order. Every row's `entry.kind` is the table's
/// `kind`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SelectTable {
    /// The kind of every row.
    pub kind: EntryKind,
    /// The rows, in order.
    pub rows: Vec<SelectRow>,
}

impl SelectTable {
    /// Every entry of `kind` in `env`'s stack, in full-name order, each at
    /// weight 1.
    pub fn universe(
        env: &SelectEnv<'_>,
        kind: EntryKind,
    ) -> Self {
        let names: Vec<CatalogName> = env
            .stack()
            .of_kind(kind)
            .into_iter()
            .map(|e| e.name().clone())
            .collect();
        Self::of(kind, names)
    }

    /// `names`, entries of `kind`, in full-name order, each at weight 1.
    pub fn of(
        kind: EntryKind,
        names: impl IntoIterator<Item = CatalogName>,
    ) -> Self {
        let mut names: Vec<CatalogName> = names.into_iter().collect();
        names.sort();
        names.dedup();
        Self {
            kind,
            rows: names
                .into_iter()
                .map(|path| SelectRow {
                    entry: KindRef { path, kind },
                    weight: 1.0,
                })
                .collect(),
        }
    }

    /// The rows' full names, in order.
    pub fn names(&self) -> Vec<&CatalogName> {
        self.rows.iter().map(|r| &r.entry.path).collect()
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// How many rows there are.
    pub fn len(&self) -> usize {
        self.rows.len()
    }
}

/// A key a derive step writes, declared so a query can test it as `$KEY`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedKey {
    /// The key, without the `$`.
    pub name: String,
    /// What it holds, for a person.
    pub description: String,
}

/// A derive step: tags a caller computes for each build, beyond the
/// catalog's. A query reads a derived key `k` as the `$` word `$k`.
pub trait DeriveTags {
    /// The keys it writes.
    fn keys(&self) -> Vec<DerivedKey>;

    /// The tags, by key, for `build`.
    fn tags(
        &self,
        build: &ResolvedBuild,
    ) -> BTreeMap<String, Value>;
}

/// A [`DeriveTags`] that is a closure over one build.
pub struct DeriveEach<F> {
    /// The keys it writes.
    pub keys: Vec<DerivedKey>,
    /// The tags, by key, for a build.
    pub tags: F,
}

impl<F: Fn(&ResolvedBuild) -> BTreeMap<String, Value>> DeriveTags for DeriveEach<F> {
    fn keys(&self) -> Vec<DerivedKey> {
        self.keys.clone()
    }

    fn tags(
        &self,
        build: &ResolvedBuild,
    ) -> BTreeMap<String, Value> {
        (self.tags)(build)
    }
}

/// A build as a query reads it: resolved, with its derived tags.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectBuild {
    /// The build, resolved.
    pub build: ResolvedBuild,
    /// The tags the derive steps wrote, by key.
    pub derived: BTreeMap<String, Value>,
}

/// The `$` words the query language defines; a derived key may not be one.
pub const RESERVED_WORDS: [&str; 14] = [
    "any",
    "all",
    "not",
    "weight",
    "name",
    "presets",
    "released",
    "license",
    "origin",
    "parent",
    "template_class",
    "contract",
    "format",
    "dtype",
];

/// What a query's tests read entries through: the stack, the derive
/// steps, and a cache of the builds resolved so far.
pub struct SelectEnv<'a> {
    stack: &'a PretrainedCatalogStack,
    derive: Vec<&'a dyn DeriveTags>,
    derived_keys: Vec<DerivedKey>,
    builds: RefCell<BTreeMap<CatalogName, Rc<SelectBuild>>>,
}

impl<'a> SelectEnv<'a> {
    /// An environment over `stack`, with no derive steps.
    pub fn new(stack: &'a PretrainedCatalogStack) -> Self {
        Self {
            stack,
            derive: Vec::new(),
            derived_keys: Vec::new(),
            builds: RefCell::new(BTreeMap::new()),
        }
    }

    /// An environment over `stack`, with `derive`'s keys.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] when a derived key is one of the
    /// [`RESERVED_WORDS`], or two steps write the same key.
    pub fn with_derive(
        stack: &'a PretrainedCatalogStack,
        derive: Vec<&'a dyn DeriveTags>,
    ) -> CatalogResult<Self> {
        let mut derived_keys: Vec<DerivedKey> = Vec::new();
        for key in derive.iter().flat_map(|d| d.keys()) {
            if RESERVED_WORDS.contains(&key.name.as_str()) {
                return Err(CatalogError::new(
                    CatalogErrorKind::Selection,
                    format!(
                        "the derived key `{}` would be the `$` word `${}`, which the query \
                         language defines",
                        key.name, key.name
                    ),
                ));
            }
            if derived_keys.iter().any(|k| k.name == key.name) {
                return Err(CatalogError::new(
                    CatalogErrorKind::Selection,
                    format!("two derive steps write the key `{}`", key.name),
                ));
            }
            derived_keys.push(key);
        }
        Ok(Self {
            stack,
            derive,
            derived_keys,
            builds: RefCell::new(BTreeMap::new()),
        })
    }

    /// The stack.
    pub fn stack(&self) -> &'a PretrainedCatalogStack {
        self.stack
    }

    /// The keys the derive steps write.
    pub fn derived_keys(&self) -> &[DerivedKey] {
        &self.derived_keys
    }

    /// The build `name`, resolved once and kept.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] when it doesn't resolve
    /// ([`PretrainedCatalogStack::resolve_build`]).
    pub fn build(
        &self,
        name: &CatalogName,
    ) -> CatalogResult<Rc<SelectBuild>> {
        if let Some(found) = self.builds.borrow().get(name) {
            return Ok(found.clone());
        }
        let build = self.stack.resolve_build(name).map_err(|e| {
            CatalogError::new(
                CatalogErrorKind::Selection,
                format!("`{name}` doesn't resolve: {e}"),
            )
        })?;
        let mut derived = BTreeMap::new();
        for step in &self.derive {
            derived.extend(step.tags(&build));
        }
        let entry = Rc::new(SelectBuild { build, derived });
        self.builds.borrow_mut().insert(name.clone(), entry.clone());
        Ok(entry)
    }
}

/// Where a stage of a run came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageOrigin {
    /// The prefix a tool injects for the presets it applies, by the flags as
    /// written.
    Prefix(Vec<String>),
    /// The context's defaults.
    Default,
    /// A named selection, spliced in by `use`.
    Use(String),
    /// A command-line flag, as written.
    Flag(String),
    /// The query itself.
    Query,
    /// The tail the composition appends.
    Tail,
}

/// One stage of a traced run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageTrace {
    /// Its place in the run, from 0.
    pub index: usize,
    /// The stage's operation (`keep`, `limit`, ..).
    pub op: &'static str,
    /// Where it came from.
    pub origin: StageOrigin,
    /// Its label, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// How many rows it removed.
    pub dropped: usize,
    /// For a stage that weighs, the weight it committed for each row it
    /// kept, in order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub weights: Vec<(CatalogName, f64)>,
}

/// A row a stage removed.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DroppedRow {
    /// The entry.
    pub entry: KindRef,
    /// The stage that removed it.
    pub stage: usize,
    /// Its weight when it was removed.
    pub weight: f64,
    /// Why, for a person.
    pub why: String,
}

/// What a traced run records: each stage, and each row removed.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SelectTrace {
    /// One per stage run, in order.
    pub stages: Vec<StageTrace>,
    /// Every row a stage removed, in order.
    pub dropped: Vec<DroppedRow>,
}

/// A traced run: the result, and how it came about.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SelectRun {
    /// The result.
    pub table: SelectTable,
    /// The trace.
    pub trace: SelectTrace,
}
