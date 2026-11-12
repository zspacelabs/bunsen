//! Choosing: composing a query from a context and a request, and running it.
//!
//! The pipeline that runs is a concatenation; nothing is merged:
//!
//! ```text
//! the tool's prefix ++ defaults ++ the request (each `use` spliced in place) ++ the tail
//! ```
//!
//! - **The prefix** ([`preset_prefix`]) is the applicability of the presets a
//!   tool will apply: one `keep: {$presets: [..]}`, first. No stage adds a row,
//!   so every row of the result can take those presets, whatever stages follow,
//!   and the query needn't mention them.
//! - **The defaults** ([`SelectionContext::defaults`]) apply to every request
//!   but one whose only name is one build. They should hold only stages that
//!   commute: `from`, fixed tests, and multiplying `weigh`s.
//! - **The request**: its names first (a named selection's name is a `use`; the
//!   other names are one `from`), then its stages.
//! - **Labels**: a stage labeled `as: L` can be removed by a later `unset: L`,
//!   so a request can drop a hard default. Naming a label no earlier stage has,
//!   or giving two stages one label, is an error.
//! - **The tail**: `order: {desc: $weight}` when nothing orders, and, when
//!   choosing, `limit: 1`.
//!
//! A query only filters, weighs and orders: it never selects presets. The
//! caller applies the presets it chose.

use alloc::rc::Rc;

use serde_json::Value;

use crate::{
    data::catalog::{
        document::{
            AxisKind,
            CatalogEntry,
            EntryKind,
        },
        error::{
            CatalogErrorExt,
            CatalogResult,
        },
        lookup::LeafMatch,
        name::CatalogName,
        presets::PresetSelection,
        select::{
            query::{
                LabeledStage,
                NameList,
                PerStage,
                Predicate,
                SelectQuery,
                SortKey,
                SortKeys,
                Stage,
                query_error,
            },
            run::MEMBERSHIP_KINDS,
            state::{
                SelectBuild,
                SelectEnv,
                SelectRun,
                SelectTable,
                StageOrigin,
            },
        },
        stack::BunsenCatalogStack,
    },
    errors::BunsenError,
};

/// A stage, with its label and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct OriginStage {
    /// The stage and its label.
    pub stage: LabeledStage,
    /// Where it came from.
    pub origin: StageOrigin,
}

impl OriginStage {
    /// `stage`, unlabeled, from `origin`.
    pub fn new(
        stage: Stage,
        origin: StageOrigin,
    ) -> Self {
        Self::labeled(stage.into(), origin)
    }

    /// `stage`, with its label, from `origin`.
    pub fn labeled(
        stage: LabeledStage,
        origin: StageOrigin,
    ) -> Self {
        Self { stage, origin }
    }
}

/// A stored list of stages a request names: `use: en`.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedSelection {
    /// Its name.
    pub name: String,
    /// A line for a listing.
    pub description: String,
    /// The stages it splices in.
    pub run: Vec<LabeledStage>,
}

/// What a tool chooses with: the defaults every request starts with, and
/// the named selections.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectionContext {
    /// The default stages, unless the request's only name is one build.
    pub defaults: Vec<LabeledStage>,
    /// The named selections.
    pub named: Vec<NamedSelection>,
}

/// What a request asks: a kind, names, then stages.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectRequest {
    /// The kind it selects; `build` by default.
    pub kind: EntryKind,
    /// Names, in order: a named selection's name is a `use`, and the others
    /// are one `from`.
    pub names: Vec<String>,
    /// The stages, in order.
    pub stages: Vec<OriginStage>,
    /// The query's floor; the default when absent.
    pub floor: Option<f64>,
}

impl Default for SelectRequest {
    fn default() -> Self {
        Self {
            kind: EntryKind::Build,
            names: Vec::new(),
            stages: Vec::new(),
            floor: None,
        }
    }
}

impl SelectRequest {
    /// A request for entries of `kind`.
    pub fn of_kind(kind: EntryKind) -> Self {
        Self {
            kind,
            ..Self::default()
        }
    }

    /// This request with the name `name`.
    pub fn name(
        mut self,
        name: impl Into<String>,
    ) -> Self {
        self.names.push(name.into());
        self
    }

    /// This request with `stage`, from `origin`.
    pub fn stage(
        mut self,
        stage: Stage,
        origin: StageOrigin,
    ) -> Self {
        self.stages.push(OriginStage::new(stage, origin));
        self
    }
}

/// A composed query: the stages and where each came from.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedQuery {
    /// The query.
    pub query: SelectQuery,
    /// Each stage's origin, by index.
    pub origins: Vec<StageOrigin>,
    /// The build the request named alone, when it did.
    pub exact: Option<CatalogName>,
}

/// The prefix a tool injects for the presets it will apply: one
/// `keep: {$presets: [..]}`, attributed to `flags` (the flags as written).
/// `None` when there are no presets.
pub fn preset_prefix(
    presets: &PresetSelection,
    flags: Vec<String>,
) -> Option<OriginStage> {
    if presets.presets.is_empty() {
        return None;
    }
    let names = presets.presets.iter().cloned().map(Value::String).collect();
    Some(OriginStage::new(
        Stage::Keep(Predicate::Test {
            key: "$presets".to_string(),
            value: Value::Array(names),
        }),
        StageOrigin::Prefix(flags),
    ))
}

/// The member of the bool axis `axis` (a lookup) that `--on` (`on`) or
/// `--off` names: `X` or `no-X`. It is read from the axis's declaration, so
/// it needs no build.
///
/// # Errors
/// [`CatalogRule::Selection`](crate::data::catalog::CatalogRule::Selection)
/// when `axis` names no axis, or one that isn't bool.
pub fn bool_axis_member(
    stack: &BunsenCatalogStack,
    axis: &str,
    on: bool,
) -> CatalogResult<String> {
    let name = stack
        .find(axis, &[EntryKind::Axis], LeafMatch::Exact)
        .map_err(|e| query_error(format!("`{axis}`: {}", e.problem_message())))?;
    let decl = stack.declaration(&name).expect("found");
    match decl.body() {
        CatalogEntry::Axis(a) if a.axis == AxisKind::Bool => Ok(if on {
            a.name.clone()
        } else {
            a.negate(&a.name).expect("a bool axis negates its positive")
        }),
        _ => Err(query_error(format!(
            "`{name}` is not a bool axis: name its member as a preset"
        ))),
    }
}

/// The error for an empty result, with the stages that removed rows.
pub fn empty_error(run: &SelectRun) -> BunsenError {
    let stages: Vec<String> = run
        .trace
        .stages
        .iter()
        .filter(|s| s.dropped > 0)
        .map(|s| {
            format!(
                "stage {} (`{}`, {}) dropped {}",
                s.index, s.op, s.origin, s.dropped
            )
        })
        .collect();
    let mut message = "no build is left to choose".to_string();
    if !stages.is_empty() {
        message.push_str(&format!(": {}", stages.join("; ")));
    }
    query_error(message)
}

/// How deep `use` may nest.
const MAX_USE_DEPTH: usize = 8;

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

    /// The pipeline `request` makes over the context, with `prefix` first,
    /// as a query of the request's kind; `choosing` adds the tail's
    /// `limit: 1`.
    ///
    /// # Errors
    /// [`CatalogRule::Selection`](crate::data::catalog::CatalogRule::Selection)
    /// for a `use` of a name no named selection has, or one nested too deep.
    pub fn compose(
        &self,
        env: &SelectEnv<'_>,
        prefix: Option<OriginStage>,
        request: &SelectRequest,
        choosing: bool,
    ) -> CatalogResult<ComposedQuery> {
        let mut stages: Vec<OriginStage> = prefix.into_iter().collect();
        let (uses, refs): (Vec<&String>, Vec<&String>) =
            request.names.iter().partition(|n| self.named(n).is_some());
        let exact = match (refs.as_slice(), uses.is_empty()) {
            ([only], true) if request.kind == EntryKind::Build => env
                .stack()
                .find(only, &MEMBERSHIP_KINDS, LeafMatch::Exact)
                .ok()
                .filter(|n| {
                    env.stack()
                        .declaration(n)
                        .is_some_and(|d| d.kind() == EntryKind::Build)
                }),
            _ => None,
        };
        if exact.is_none() {
            for stage in &self.defaults {
                self.push(&mut stages, stage, StageOrigin::Default, 0)?;
            }
        }
        for name in uses {
            let stage = Stage::Use(name.clone()).into();
            self.push(&mut stages, &stage, StageOrigin::Query, 0)?;
        }
        if !refs.is_empty() {
            stages.push(OriginStage::new(
                Stage::From(NameList(refs.into_iter().cloned().collect())),
                StageOrigin::Query,
            ));
        }
        for stage in &request.stages {
            self.push(&mut stages, &stage.stage, stage.origin.clone(), 0)?;
        }
        if !stages
            .iter()
            .any(|s| matches!(s.stage.stage, Stage::Order(_)))
        {
            stages.push(OriginStage::new(
                Stage::Order(SortKeys(vec![SortKey::desc("$weight")])),
                StageOrigin::Tail,
            ));
        }
        if choosing {
            stages.push(OriginStage::new(Stage::Limit(1), StageOrigin::Tail));
        }
        let (run, origins) = stages.into_iter().map(|s| (s.stage, s.origin)).unzip();
        Ok(ComposedQuery {
            query: SelectQuery {
                kind: request.kind,
                floor: request.floor,
                run,
            },
            origins,
            exact,
        })
    }

    /// Pushes `stage`: a `use` splices its stages in place, an `unset`
    /// removes the earlier stage with its label, and a `per` has the `use`s
    /// in its stages spliced.
    fn push(
        &self,
        stages: &mut Vec<OriginStage>,
        stage: &LabeledStage,
        origin: StageOrigin,
        depth: usize,
    ) -> CatalogResult<()> {
        match &stage.stage {
            Stage::Use(name) => {
                for inner in &self.named_run(name, depth)?.run {
                    self.push(stages, inner, StageOrigin::Use(name.clone()), depth + 1)?;
                }
            }
            Stage::Unset(label) => {
                let at = stages
                    .iter()
                    .position(|s| s.stage.label.as_ref() == Some(label))
                    .ok_or_else(|| {
                        query_error(format!(
                            "`unset: {label}`: no earlier stage is labeled `{label}`"
                        ))
                    })?;
                stages.remove(at);
            }
            other => {
                if let Some(label) = &stage.label
                    && stages.iter().any(|s| s.stage.label.as_ref() == Some(label))
                {
                    return Err(query_error(format!("two stages are labeled `{label}`")));
                }
                let mut stage = stage.clone();
                if let Stage::Per(per) = other {
                    let mut inner = Vec::new();
                    for s in &per.run {
                        self.push(&mut inner, s, StageOrigin::Query, depth)?;
                    }
                    stage.stage = Stage::Per(PerStage {
                        group: per.group.clone(),
                        run: inner.into_iter().map(|s| s.stage).collect(),
                    });
                }
                stages.push(OriginStage::labeled(stage, origin));
            }
        }
        Ok(())
    }

    /// The named selection `name`, for a `use` nested `depth` deep.
    fn named_run(
        &self,
        name: &str,
        depth: usize,
    ) -> CatalogResult<&NamedSelection> {
        if depth >= MAX_USE_DEPTH {
            return Err(query_error(format!(
                "`use: {name}` nests more than {MAX_USE_DEPTH} deep"
            )));
        }
        self.named(name)
            .ok_or_else(|| query_error(format!("`use: {name}`: no named selection `{name}`")))
    }

    /// Runs `composed` over the entries of its kind in `env`'s stack (all of
    /// them, or those in `universe`), traced.
    ///
    /// # Errors
    /// As [`SelectQuery::compile`], and when the request named one build
    /// alone that the run removed, with the reason.
    pub fn run_composed(
        &self,
        env: &SelectEnv<'_>,
        universe: Option<&[CatalogName]>,
        composed: &ComposedQuery,
    ) -> CatalogResult<SelectRun> {
        let mut compiled = composed.query.compile(env)?;
        for (index, origin) in composed.origins.iter().enumerate() {
            compiled.set_origin(index, origin.clone());
        }
        let kind = composed.query.kind;
        let table = match universe {
            Some(names) => SelectTable::of(kind, names.iter().cloned()),
            None => SelectTable::universe(env, kind),
        };
        let run = compiled.run_traced(env, table)?;
        if let Some(build) = &composed.exact
            && !run.table.rows.iter().any(|r| &r.entry.path == build)
        {
            let why = run
                .trace
                .dropped
                .iter()
                .find(|d| &d.entry.path == build)
                .map(|d| format!(": {}", d.why))
                .unwrap_or_default();
            return Err(query_error(format!(
                "the build `{build}` was named, but can't be chosen{why}"
            )));
        }
        Ok(run)
    }

    /// Every entry `request` (with `prefix` first) selects, in order, traced.
    ///
    /// # Errors
    /// As [`compose`](Self::compose) and [`run_composed`](Self::run_composed).
    pub fn rank(
        &self,
        env: &SelectEnv<'_>,
        universe: Option<&[CatalogName]>,
        prefix: Option<OriginStage>,
        request: &SelectRequest,
    ) -> CatalogResult<SelectRun> {
        let composed = self.compose(env, prefix, request, false)?;
        self.run_composed(env, universe, &composed)
    }

    /// The one build `request` (with `prefix` first) chooses, as loading a
    /// model does: the first of the result, by the tail's `limit: 1`.
    ///
    /// # Errors
    /// As [`rank`](Self::rank), and [`empty_error`] when no build is left.
    pub fn choose(
        &self,
        env: &SelectEnv<'_>,
        universe: Option<&[CatalogName]>,
        prefix: Option<OriginStage>,
        request: &SelectRequest,
    ) -> CatalogResult<(Rc<SelectBuild>, SelectRun)> {
        if request.kind != EntryKind::Build {
            return Err(query_error(format!(
                "choosing takes a build, not a `{}`; rank a query of other kinds",
                request.kind
            )));
        }
        let composed = self.compose(env, prefix, request, true)?;
        let run = self.run_composed(env, universe, &composed)?;
        let Some(row) = run.table.rows.first() else {
            return Err(empty_error(&run));
        };
        let build = env.build(&row.entry.path)?;
        Ok((build, run))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::catalog::fixtures;

    fn leaves(run: &SelectRun) -> Vec<String> {
        run.table
            .names()
            .iter()
            .map(|n| n.leaf().to_string())
            .collect()
    }

    fn stages(yaml: &str) -> Vec<LabeledStage> {
        serde_saphyr::from_str(yaml).unwrap()
    }

    fn context() -> SelectionContext {
        SelectionContext {
            defaults: stages("[{keep: instruct}, {order: {desc: parameters}}]"),
            named: vec![NamedSelection {
                name: "en".into(),
                description: "English speech.".into(),
                run: stages("[{from: whisper}, {keep: {language: en}}]"),
            }],
        }
    }

    #[test]
    fn test_defaults_apply_unless_one_build_is_named() {
        let stack = fixtures::example_stack().unwrap();
        let env = SelectEnv::new(&stack);
        let run = context()
            .rank(&env, None, None, &SelectRequest::default())
            .unwrap();
        assert_eq!(leaves(&run), ["qwen3-4b-thinking-2507", "qwen3-0.6b"]);
        // One build named alone skips the defaults.
        let base = SelectRequest::default().name("qwen3-0.6b-base");
        let run = context().rank(&env, None, None, &base).unwrap();
        assert_eq!(leaves(&run), ["qwen3-0.6b-base"]);
    }

    #[test]
    fn test_a_named_selection_is_spliced_where_it_is_named() {
        let stack = fixtures::example_stack().unwrap();
        let env = SelectEnv::new(&stack);
        let context = SelectionContext {
            defaults: Vec::new(),
            ..context()
        };
        let request = SelectRequest::default().name("en");
        let composed = context.compose(&env, None, &request, false).unwrap();
        assert_eq!(
            composed.origins,
            [
                StageOrigin::Use("en".into()),
                StageOrigin::Use("en".into()),
                StageOrigin::Tail
            ]
        );
        let run = context.run_composed(&env, None, &composed).unwrap();
        assert_eq!(run.table.len(), 4);
        let unknown = SelectRequest::default().stage(Stage::Use("fr".into()), StageOrigin::Query);
        assert!(context.compose(&env, None, &unknown, false).is_err());
    }

    #[test]
    fn test_the_prefix_holds_for_every_row_whatever_follows() {
        let stack = fixtures::example_stack().unwrap();
        let env = SelectEnv::new(&stack);
        let off = bool_axis_member(&stack, "thinking", false).unwrap();
        assert_eq!(off, "no-thinking");
        let prefix = || {
            preset_prefix(
                &PresetSelection::of([off.as_str()]),
                vec!["--off thinking".into()],
            )
        };
        // A `limit` first still yields only builds that can take the preset.
        let request = SelectRequest::default()
            .stage(Stage::Limit(1), StageOrigin::Flag("--limit 1".into()))
            .name("qwen3");
        let context = SelectionContext::default();
        let run = context.rank(&env, None, prefix(), &request).unwrap();
        assert_eq!(leaves(&run), ["qwen3-0.6b"]);
        assert_eq!(
            run.trace.stages[0].origin,
            StageOrigin::Prefix(vec!["--off thinking".into()])
        );
        // A build named alone that can't take it is an error, with why.
        let named = SelectRequest::default().name("qwen3-4b-thinking-2507");
        let err = context.rank(&env, None, prefix(), &named).unwrap_err();
        assert!(
            err.problem_message()
                .contains("was named, but can't be chosen"),
            "{err}"
        );
        assert!(err.problem_message().contains("no-thinking"), "{err}");
        assert!(bool_axis_member(&stack, "task", true).is_err());
    }

    #[test]
    fn test_choose_takes_the_first_and_an_empty_result_says_why() {
        let stack = fixtures::example_stack().unwrap();
        let env = SelectEnv::new(&stack);
        let (build, _) = context()
            .choose(&env, None, None, &SelectRequest::default())
            .unwrap();
        assert_eq!(build.build.name.leaf(), "qwen3-4b-thinking-2507");
        let none = SelectRequest::default().name("whisper");
        let err = context().choose(&env, None, None, &none).unwrap_err();
        assert!(
            err.problem_message()
                .starts_with("no build is left to choose"),
            "{err}"
        );
        assert!(err.problem_message().contains("defaults"), "{err}");
    }
}
