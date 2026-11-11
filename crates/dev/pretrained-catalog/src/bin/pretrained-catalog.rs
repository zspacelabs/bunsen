//! `pretrained-catalog`: import catalogs, then list, show, resolve and
//! select over the state they make.
//!
//! ```text
//! pretrained-catalog --examples list --kind build
//! pretrained-catalog --catalog a.yml --catalog b.yml show qwen3
//! pretrained-catalog --examples resolve large-v3 --preset translate
//! pretrained-catalog --examples select qwen3 --off thinking --limit 1
//! pretrained-catalog --examples select whisper --weigh '{release: latest}' --desc '$weight' --rank
//! pretrained-catalog --examples releases release
//! ```
//!
//! Catalogs are imported in the order given, `--examples` (the example
//! catalogs) first when it is given.
//!
//! `select` stands in for a tool that chooses builds (bunsen-tool's chat
//! and asr commands): it turns its preset flags into the presets it applies
//! and a prefix that keeps only the builds they apply to
//! ([`preset_prefix`]), its other flags into stages, in command-line order,
//! and composes them over its defaults ([`SelectionContext::compose`]).

use std::{
    path::PathBuf,
    process::ExitCode,
};

use bunsen::data::pretrained::build_catalog::{
    CatalogErrors,
    CatalogName,
    PretrainedCatalogStack,
    document::{
        CatalogEntry,
        EntryKind,
    },
    fixtures,
    lookup::{
        LeafMatch,
        SEARCHABLE,
    },
    presets::PresetSelection,
    resolve::TagSource,
    select::{
        choose::{
            OriginStage,
            SelectRequest,
            SelectionContext,
            bool_axis_member,
            empty_error,
            preset_prefix,
        },
        query::{
            LabeledStage,
            PerGroup,
            PerStage,
            Predicate,
            SelectQuery,
            SortKey,
            SortKeys,
            Stage,
            WeighBlock,
            WeighEntry,
        },
        state::{
            SelectEnv,
            SelectRun,
            StageOrigin,
            StageTrace,
        },
    },
};
use clap::{
    ArgMatches,
    CommandFactory,
    FromArgMatches,
    Parser,
    Subcommand,
};
use serde::Serialize;
use serde_json::{
    Value,
    json,
};

/// Writes a line of output.
macro_rules! emit {
    ($out:expr, $($arg:tt)*) => {{
        use std::fmt::Write as _;
        writeln!($out, $($arg)*).expect("writing to a String");
    }};
}

/// Import pretrained catalogs, and query the state they make.
#[derive(Parser, Debug)]
#[command(name = "pretrained-catalog", version)]
struct Cli {
    /// A catalog to import, in order; repeatable. YAML, or JSON by its
    /// `.json` extension.
    #[arg(long = "catalog", value_name = "PATH")]
    catalogs: Vec<PathBuf>,

    /// Import the example catalogs (prelude, commons, qwen3, whisper) first.
    #[arg(long)]
    examples: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// The imported catalogs, in import order.
    Layers,

    /// Entries, by their shortest unique names.
    List {
        /// Only entries of this kind (`build`, `namespace`, `tag`, ...).
        #[arg(long)]
        kind: Option<String>,
        /// Only the groups of this facet ("all families").
        #[arg(long)]
        facet: Option<String>,
        /// Only the builds that are members of this group.
        #[arg(long)]
        members: Option<String>,
        /// Only the entries directly in this scope.
        #[arg(long)]
        children: Option<String>,
    },

    /// One entry: its kind, its history, and its merged fields.
    Show {
        /// The entry, by any unambiguous name.
        name: String,
    },

    /// A build, resolved: chain, memberships, tags, presets, dependencies.
    Resolve {
        /// The build, by any unambiguous name.
        build: String,
        #[command(flatten)]
        presets: PresetArgs,
        /// A merge patch applied after the presets (YAML or JSON).
        #[arg(long)]
        patch: Option<String>,
    },

    /// Select builds, printed as YAML: the composed query, why, the
    /// presets applied, and the result in order, each build resolved with
    /// its weight. Each flag is one stage, in command-line order, but
    /// `--on`, `--off` and `--preset`: they are the presets applied, and the
    /// prefix that keeps only the builds that can take them.
    Select(Box<SelectArgs>),

    /// The namespace as a tree, from the root or from one entry.
    Tree {
        /// The entry to start from, by any unambiguous name; the root when
        /// absent.
        root: Option<String>,
        /// Show only entries of these kinds (repeatable), with the
        /// containers on the way to them.
        #[arg(long = "kind", value_name = "KIND")]
        kinds: Vec<String>,
        /// How many levels below the start to show.
        #[arg(long)]
        depth: Option<usize>,
    },

    /// A release group's members, oldest first.
    Releases {
        /// The release group, by any unambiguous name.
        group: String,
    },
}

/// The `select` flags: see [`Command::Select`].
#[derive(clap::Args, Debug)]
struct SelectArgs {
    /// The kind of entry to select (`build`, `template_class`, `namespace`,
    /// `resource`, ..).
    #[arg(long, default_value = "build")]
    kind: String,
    /// Names, in order: one `from` stage; a named selection's name is a
    /// `use`.
    names: Vec<String>,
    /// `keep: P` (a name or a flow map); repeatable.
    #[arg(long, value_name = "P")]
    keep: Vec<String>,
    /// `drop: P`; repeatable.
    #[arg(long, value_name = "P")]
    drop: Vec<String>,
    /// `keep: {KEY: {lt: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    lt: Vec<String>,
    /// `keep: {KEY: {le: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    le: Vec<String>,
    /// `keep: {KEY: {gt: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    gt: Vec<String>,
    /// `keep: {KEY: {ge: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    ge: Vec<String>,
    /// `keep: {KEY: {eq: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    eq: Vec<String>,
    /// `keep: {KEY: {ne: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    ne: Vec<String>,
    /// `keep: {TAG: {has: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["TAG", "V"])]
    has: Vec<String>,
    /// `weigh: {by: [E]}`: a block of one entry (`{when: P, else: w}`);
    /// repeatable.
    #[arg(long, value_name = "E")]
    weigh: Vec<String>,
    /// `weigh: B`: a whole block (`{by: [..], floor: f}`); repeatable.
    #[arg(long, value_name = "B")]
    weigh_block: Vec<String>,
    /// The query's floor.
    #[arg(long, value_name = "F")]
    floor: Option<String>,
    /// `keep: {$weight: {ge: W}}`; repeatable.
    #[arg(long, value_name = "W")]
    cutoff: Vec<String>,
    /// A sort key, ascending; consecutive sort flags join one `order`.
    #[arg(long, value_name = "K")]
    asc: Vec<String>,
    /// A sort key, descending.
    #[arg(long, value_name = "K")]
    desc: Vec<String>,
    /// `limit: N`; repeatable.
    #[arg(long, value_name = "N")]
    limit: Vec<String>,
    /// `use: NAME`; repeatable.
    #[arg(long = "use", value_name = "NAME")]
    uses: Vec<String>,
    /// `unset: LABEL`: removes the earlier stage labeled LABEL (a default's
    /// `as`); repeatable.
    #[arg(long, value_name = "LABEL")]
    unset: Vec<String>,
    /// Starts a `per` stage, by a facet (`family`) or a list of names
    /// (`'[qwen3, whisper]'`): the flags up to its `--end` are its `run`.
    #[arg(long, value_name = "GROUP")]
    per: Vec<String>,
    /// Ends the innermost `--per`.
    #[arg(long, num_args = 0, default_missing_value = "end")]
    end: Vec<String>,
    /// A list of stages, spliced in: a file, or YAML flow; repeatable.
    #[arg(long, value_name = "FILE|[..]")]
    pipeline: Vec<String>,
    #[command(flatten)]
    presets: PresetArgs,
    /// The defaults, a list of stages (a file, or YAML flow): applied
    /// unless the only name is one build.
    #[arg(long, value_name = "FILE|[..]")]
    defaults: Option<String>,
    /// Also list every row a stage removed, with why.
    #[arg(long)]
    rank: bool,
}

#[derive(clap::Args, Debug)]
struct PresetArgs {
    /// A preset to select; repeatable.
    #[arg(long = "preset", value_name = "NAME")]
    presets: Vec<String>,
    /// Select the positive member of a bool axis (`--on thinking`).
    #[arg(long = "on", value_name = "AXIS")]
    on: Vec<String>,
    /// Select the negative member of a bool axis (`--off thinking`).
    #[arg(long = "off", value_name = "AXIS")]
    off: Vec<String>,
}

fn main() -> ExitCode {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).expect("clap parsed it");
    let mut out = String::new();
    let result = run(cli, &matches, &mut out);
    print!("{out}");
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// Runs `cli` (parsed from `matches`), writing its output to `out`.
fn run(
    cli: Cli,
    matches: &ArgMatches,
    out: &mut String,
) -> Result<(), String> {
    let stack = load(&cli)?;
    match cli.command {
        Command::Layers => layers(out, &stack),
        Command::List {
            kind,
            facet,
            members,
            children,
        } => list(out, &stack, kind, facet, members, children)?,
        Command::Show { name } => show(out, &stack, &name)?,
        Command::Resolve {
            build,
            presets,
            patch,
        } => resolve(out, &stack, &build, &presets, patch.as_deref())?,
        Command::Select(args) => {
            let matches = matches
                .subcommand_matches("select")
                .expect("the select subcommand");
            select(out, &stack, matches, *args)?
        }
        Command::Tree { root, kinds, depth } => tree(out, &stack, root.as_deref(), &kinds, depth)?,
        Command::Releases { group } => releases(out, &stack, &group)?,
    }
    Ok(())
}

fn errors(e: CatalogErrors) -> String {
    e.to_string()
}

fn load(cli: &Cli) -> Result<PretrainedCatalogStack, String> {
    let mut stack = if cli.examples {
        fixtures::example_stack().map_err(errors)?
    } else {
        PretrainedCatalogStack::new()
    };
    for path in &cli.catalogs {
        stack.push_file(path).map_err(errors)?;
    }
    if stack.layers().is_empty() {
        return Err("no catalogs: give `--catalog PATH` (repeatable) or `--examples`".into());
    }
    Ok(stack)
}

fn parse_kind(kind: &str) -> Result<EntryKind, String> {
    SEARCHABLE
        .iter()
        .copied()
        .find(|k| k.as_str() == kind)
        .ok_or_else(|| {
            let known: Vec<&str> = SEARCHABLE.iter().map(|k| k.as_str()).collect();
            format!("no kind `{kind}`; the kinds are {}", known.join(", "))
        })
}

fn find(
    stack: &PretrainedCatalogStack,
    query: &str,
    kinds: &[EntryKind],
) -> Result<CatalogName, String> {
    stack
        .find(query, kinds, LeafMatch::Exact)
        .map_err(|e| e.message)
}

fn layers(
    out: &mut String,
    stack: &PretrainedCatalogStack,
) {
    for (i, layer) in stack.layers().iter().enumerate() {
        let path = layer
            .path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(built in)".into());
        emit!(
            out,
            "{i:>2}  {:<14} {:>3} occurrences  {path}",
            layer.label(),
            layer.entries().len()
        );
    }
}

fn list(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    kind: Option<String>,
    facet: Option<String>,
    members: Option<String>,
    children: Option<String>,
) -> Result<(), String> {
    let mut names: Vec<CatalogName> = if let Some(facet) = facet {
        let facet = find(stack, &facet, &[EntryKind::Facet])?;
        stack
            .facet_groups(&facet)
            .iter()
            .map(|e| e.name().clone())
            .collect()
    } else if let Some(group) = members {
        let group = find(
            stack,
            &group,
            &[
                EntryKind::Namespace,
                EntryKind::Contract,
                EntryKind::Release,
            ],
        )?;
        stack.members_of(&group)
    } else if let Some(scope) = children {
        let scope = find(stack, &scope, &SEARCHABLE)?;
        stack
            .children(&scope)
            .iter()
            .map(|e| e.name().clone())
            .collect()
    } else {
        stack.declarations().map(|e| e.name().clone()).collect()
    };
    if let Some(kind) = kind {
        let kind = parse_kind(&kind)?;
        names.retain(|n| stack.declaration(n).is_some_and(|d| d.kind() == kind));
    }
    for name in names {
        let Some(entry) = stack.declaration(&name) else {
            continue;
        };
        let shown = stack.display_name(&name);
        let facet = stack
            .group_facet(entry)
            .map(|f| format!("  ({f})"))
            .unwrap_or_default();
        emit!(
            out,
            "{:<28} {:<44} {}{facet}",
            shown.short,
            shown.full.to_string(),
            entry.kind()
        );
    }
    Ok(())
}

fn show(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    query: &str,
) -> Result<(), String> {
    let name = find(stack, query, &SEARCHABLE)?;
    let entry = stack.declaration(&name).expect("found");
    emit!(out, "{name}  ({})", entry.kind());
    for change in entry.history() {
        emit!(
            out,
            "  {:<8} in {}",
            format!("{:?}", change.merge).to_lowercase(),
            stack.layers()[change.layer].label()
        );
    }
    let yaml = serde_saphyr::to_string(entry.body()).map_err(|e| e.to_string())?;
    emit!(out, "{yaml}");
    let kids = stack.children(&name);
    if !kids.is_empty() {
        emit!(out, "children:");
        for kid in kids {
            emit!(out, "  {} ({})", kid.name().leaf(), kid.kind());
        }
    }
    Ok(())
}

fn parse_value(text: &str) -> Result<Value, String> {
    serde_saphyr::from_str(text).map_err(|e| format!("`{text}` is not YAML: {e}"))
}

fn resolve(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    query: &str,
    args: &PresetArgs,
    patch: Option<&str>,
) -> Result<(), String> {
    let name = find(stack, query, &[EntryKind::Build])?;
    let b = stack.resolve_build(&name).map_err(errors)?;
    emit!(out, "{}  {}", b.name, b.description);
    let chain: Vec<String> = b.template_chain.iter().map(ToString::to_string).collect();
    emit!(out, "  chain        {}", chain.join(" > "));
    let groups: Vec<String> = b.memberships.iter().map(ToString::to_string).collect();
    emit!(out, "  memberships  {}", groups.join(", "));
    if let Some(contract) = &b.contract {
        emit!(out, "  contract     {contract}");
    }
    if let Some(release) = &b.release {
        emit!(
            out,
            "  release      {} ({})",
            release.group,
            release.released
        );
    }
    emit!(out, "  tags");
    for (tag, value) in &b.tags {
        let source = match &value.source {
            TagSource::Written(owner) => format!("written by {owner}"),
            TagSource::Axis => "implied by the axis".into(),
            TagSource::Facet => "implied by the facet".into(),
            TagSource::Contract => "implied by the contract".into(),
        };
        emit!(
            out,
            "    {:<30} {:<28} {source}",
            tag.to_string(),
            value.value.to_string()
        );
    }
    emit!(out, "  axes");
    for using in &b.presets.axes {
        let fixed = if using.is_fixed() { "  (fixed)" } else { "" };
        emit!(
            out,
            "    {:<30} offers {:?}, default {}{fixed}",
            using.axis.name.to_string(),
            using.offers,
            using.default
        );
    }
    let names: Vec<&str> = b.presets.presets.iter().map(|p| p.name.as_str()).collect();
    emit!(out, "  presets      {}", names.join(", "));
    emit!(out, "  dependencies");
    for (key, dep) in &b.dependencies {
        emit!(out, "    {key:<18} {} ({})", dep.resource, dep.entry.file);
    }
    // The flags as shortcuts, then the presets' document.
    let mut presets = args.presets.clone();
    for (axis, on) in args
        .on
        .iter()
        .map(|a| (a, true))
        .chain(args.off.iter().map(|a| (a, false)))
    {
        let flag = format!("--{} {axis}", if on { "on" } else { "off" });
        let name = find(stack, axis, &[EntryKind::Axis])?;
        let member = b
            .presets
            .bool_member(&name, on)
            .map_err(|e| format!("`{flag}` doesn't apply: {}", e.message))?;
        emit!(out, "  `{flag}` implies `--preset {member}`");
        presets.push(member);
    }
    let selection = PresetSelection {
        presets,
        patch: patch.map(parse_value).transpose()?,
    };
    let applied = b
        .presets
        .select(Value::Object(Default::default()), &selection)
        .map_err(|e| e.message)?;
    emit!(out, "  applied      {}", applied.applied.join(", "));
    emit!(out, "  document     {}", applied.document);
    Ok(())
}

/// One flag of `select`, by its place on the command line.
enum SelectFlag {
    /// A stage, with the flag as written.
    Stage(Stage, String),
    /// A sort key, with the flag as written; consecutive ones join.
    Sort(SortKey, String),
    /// Stages spliced in, with the flag as written.
    Pipeline(Vec<LabeledStage>, String),
    /// `--per GROUP`, with the flag as written.
    PerStart(PerGroup, String),
    /// `--end`.
    PerEnd,
    /// A preset, with the flag as written.
    Preset(String, String),
}

/// The values of the flag `id`, `per` to an occurrence, each with its
/// place on the command line.
fn occurrences(
    matches: &ArgMatches,
    id: &str,
    per: usize,
) -> Vec<(usize, Vec<String>)> {
    let (Some(indices), Some(values)) = (matches.indices_of(id), matches.get_many::<String>(id))
    else {
        return Vec::new();
    };
    let indices: Vec<usize> = indices.collect();
    let values: Vec<String> = values.cloned().collect();
    indices
        .chunks(per)
        .zip(values.chunks(per))
        .map(|(i, v)| (i[0], v.to_vec()))
        .collect()
}

/// A YAML particle, or a file holding one.
fn file_or_flow(text: &str) -> Result<Value, String> {
    let path = std::path::Path::new(text);
    if path.is_file() {
        let body = std::fs::read_to_string(path).map_err(|e| format!("`{text}`: {e}"))?;
        return parse_value(&body);
    }
    parse_value(text)
}

fn stages_of(text: &str) -> Result<Vec<LabeledStage>, String> {
    serde_json::from_value(file_or_flow(text)?)
        .map_err(|e| format!("`{text}` is not a list of stages: {e}"))
}

/// The `select` flags, in command-line order.
fn select_flags(
    stack: &PretrainedCatalogStack,
    matches: &ArgMatches,
) -> Result<Vec<SelectFlag>, String> {
    let mut flags: Vec<(usize, SelectFlag)> = Vec::new();
    let predicate = |text: &str| -> Result<Predicate, String> {
        Predicate::from_value(&parse_value(text)?).map_err(|e| e.message)
    };
    for (id, keep) in [("keep", true), ("drop", false)] {
        for (at, v) in occurrences(matches, id, 1) {
            let p = predicate(&v[0])?;
            let stage = if keep { Stage::Keep(p) } else { Stage::Drop(p) };
            flags.push((at, SelectFlag::Stage(stage, format!("--{id} {}", v[0]))));
        }
    }
    for op in ["lt", "le", "gt", "ge", "eq", "ne", "has"] {
        for (at, v) in occurrences(matches, op, 2) {
            let test = Predicate::Test {
                key: v[0].clone(),
                value: json!({ op: parse_value(&v[1])? }),
            };
            let flag = format!("--{op} {} {}", v[0], v[1]);
            flags.push((at, SelectFlag::Stage(Stage::Keep(test), flag)));
        }
    }
    for (at, v) in occurrences(matches, "weigh", 1) {
        let entry: WeighEntry = serde_json::from_value(parse_value(&v[0])?)
            .map_err(|e| format!("`--weigh {}`: {e}", v[0]))?;
        let block = WeighBlock {
            by: vec![entry],
            floor: None,
        };
        flags.push((
            at,
            SelectFlag::Stage(Stage::Weigh(block), format!("--weigh {}", v[0])),
        ));
    }
    for (at, v) in occurrences(matches, "weigh_block", 1) {
        let block: WeighBlock = serde_json::from_value(parse_value(&v[0])?)
            .map_err(|e| format!("`--weigh-block {}`: {e}", v[0]))?;
        let flag = format!("--weigh-block {}", v[0]);
        flags.push((at, SelectFlag::Stage(Stage::Weigh(block), flag)));
    }
    for (at, v) in occurrences(matches, "cutoff", 1) {
        let w: f64 = v[0]
            .parse()
            .map_err(|_| format!("`--cutoff {}`: not a weight", v[0]))?;
        let test = Predicate::Test {
            key: "$weight".into(),
            value: json!({ "ge": w }),
        };
        flags.push((
            at,
            SelectFlag::Stage(Stage::Keep(test), format!("--cutoff {}", v[0])),
        ));
    }
    for (id, descending) in [("asc", false), ("desc", true)] {
        for (at, v) in occurrences(matches, id, 1) {
            let key = if descending {
                SortKey::desc(v[0].clone())
            } else {
                SortKey::asc(v[0].clone())
            };
            flags.push((at, SelectFlag::Sort(key, format!("--{id} {}", v[0]))));
        }
    }
    for (at, v) in occurrences(matches, "limit", 1) {
        let n: usize = v[0]
            .parse()
            .map_err(|_| format!("`--limit {}`: not a count", v[0]))?;
        flags.push((
            at,
            SelectFlag::Stage(Stage::Limit(n), format!("--limit {}", v[0])),
        ));
    }
    for (at, v) in occurrences(matches, "unset", 1) {
        let flag = format!("--unset {}", v[0]);
        flags.push((at, SelectFlag::Stage(Stage::Unset(v[0].clone()), flag)));
    }
    for (at, v) in occurrences(matches, "per", 1) {
        let group: PerGroup = serde_json::from_value(parse_value(&v[0])?)
            .map_err(|e| format!("`--per {}`: a facet or a list of names: {e}", v[0]))?;
        flags.push((at, SelectFlag::PerStart(group, format!("--per {}", v[0]))));
    }
    for (at, _) in occurrences(matches, "end", 1) {
        flags.push((at, SelectFlag::PerEnd));
    }
    for (at, v) in occurrences(matches, "uses", 1) {
        flags.push((
            at,
            SelectFlag::Stage(Stage::Use(v[0].clone()), format!("--use {}", v[0])),
        ));
    }
    for (at, v) in occurrences(matches, "pipeline", 1) {
        flags.push((
            at,
            SelectFlag::Pipeline(stages_of(&v[0])?, format!("--pipeline {}", v[0])),
        ));
    }
    for (at, v) in occurrences(matches, "presets", 1) {
        flags.push((
            at,
            SelectFlag::Preset(v[0].clone(), format!("--preset {}", v[0])),
        ));
    }
    for (id, on) in [("on", true), ("off", false)] {
        for (at, v) in occurrences(matches, id, 1) {
            let flag = format!("--{id} {}", v[0]);
            let member = bool_axis_member(stack, &v[0], on)
                .map_err(|e| format!("`{flag}`: {}", e.message))?;
            flags.push((at, SelectFlag::Preset(member, flag)));
        }
    }
    flags.sort_by_key(|(at, _)| *at);
    Ok(flags.into_iter().map(|(_, f)| f).collect())
}

/// A build in a result, as `select` prints it: the resolved build's
/// fields, tags as plain values, with its weight.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct ChoiceRecord {
    /// The build's full name.
    name: String,
    /// Its weight.
    weight: f64,
    /// What it is, for a person.
    description: String,
    /// The checkpoint's license.
    #[serde(skip_serializing_if = "Option::is_none")]
    license: Option<String>,
    /// Where the checkpoint comes from.
    #[serde(skip_serializing_if = "Option::is_none")]
    origin: Option<String>,
    /// The template-class chain, root first.
    template_chain: Vec<String>,
    /// Every group it is in.
    memberships: Vec<String>,
    /// Its group in each facet, by facet.
    facets: std::collections::BTreeMap<String, String>,
    /// Its contract.
    contract: Option<String>,
    /// Its release membership.
    #[serde(skip_serializing_if = "Option::is_none")]
    release: Option<ReleaseRecord>,
    /// Its tags, written and implied, by full name.
    tags: std::collections::BTreeMap<String, Value>,
    /// The tags a derive step added, by key.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    derived: std::collections::BTreeMap<String, Value>,
    /// Its dependencies: each key's resource.
    dependencies: std::collections::BTreeMap<String, String>,
}

/// A build's release membership, in a [`ChoiceRecord`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ReleaseRecord {
    /// The release group.
    group: String,
    /// When the build was released.
    released: String,
}

/// A row a stage removed, as `select --rank` prints it.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct DroppedRecord {
    /// The build's full name.
    name: String,
    /// The stage that removed it.
    stage: usize,
    /// Its weight then.
    weight: f64,
    /// Why.
    why: String,
}

/// What `select` prints.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct SelectionReport {
    /// The composed query.
    query: SelectQuery,
    /// Each stage: where it came from, and how many rows it removed.
    why: Vec<StageTrace>,
    /// With `--rank`, every row a stage removed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    dropped: Vec<DroppedRecord>,
    /// The presets applied to every result.
    presets: Vec<String>,
    /// The result, in order.
    results: Vec<ResultRecord>,
}

/// One row of a result: a build resolved, or another kind's entry.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
enum ResultRecord {
    /// A build, resolved.
    Build(Box<ChoiceRecord>),
    /// An entry of another kind.
    Entry(EntryRecord),
}

/// An entry of a kind other than `build`, in a result.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct EntryRecord {
    /// Its full name.
    name: String,
    /// Its kind.
    kind: &'static str,
    /// Its weight.
    weight: f64,
}

fn record(
    env: &SelectEnv<'_>,
    name: &CatalogName,
    weight: f64,
) -> Result<ChoiceRecord, String> {
    let entry = env.build(name).map_err(|e| e.message)?;
    let build = &entry.build;
    let names = |names: &mut dyn Iterator<Item = &CatalogName>| -> Vec<String> {
        names.map(ToString::to_string).collect()
    };
    Ok(ChoiceRecord {
        name: build.name.to_string(),
        weight,
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
        derived: entry.derived.clone(),
        dependencies: build
            .dependencies
            .iter()
            .map(|(key, dep)| (key.clone(), dep.resource.to_string()))
            .collect(),
    })
}

fn select(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    matches: &ArgMatches,
    args: SelectArgs,
) -> Result<(), String> {
    let SelectArgs {
        kind,
        names,
        floor,
        defaults,
        rank,
        ..
    } = args;
    let kind = parse_kind(&kind)?;
    let env = SelectEnv::new(stack);
    let context = SelectionContext {
        defaults: defaults
            .as_deref()
            .map(stages_of)
            .transpose()?
            .unwrap_or_default(),
        named: Vec::new(),
    };
    let mut request = SelectRequest {
        kind,
        names,
        stages: Vec::new(),
        floor: floor
            .map(|f| {
                f.parse()
                    .map_err(|_| format!("`--floor {f}`: not a weight"))
            })
            .transpose()?,
    };
    let mut presets = PresetSelection::default();
    let mut preset_flags = Vec::new();
    // The stages being built: the request's, then one frame per open
    // `--per`, each with its pending sort keys.
    struct Frame {
        stages: Vec<OriginStage>,
        sort: Option<(Vec<SortKey>, Vec<String>)>,
        per: Option<(PerGroup, String)>,
    }
    impl Frame {
        fn new(per: Option<(PerGroup, String)>) -> Self {
            Self {
                stages: Vec::new(),
                sort: None,
                per,
            }
        }

        fn flush(&mut self) {
            if let Some((keys, flags)) = self.sort.take() {
                self.stages.push(OriginStage::new(
                    Stage::Order(SortKeys(keys)),
                    StageOrigin::Flag(flags.join(" ")),
                ));
            }
        }
    }
    let mut frames = vec![Frame::new(None)];
    for flag in select_flags(stack, matches)? {
        let frame = frames.last_mut().expect("the request's frame");
        match flag {
            SelectFlag::Sort(key, text) => {
                let (keys, flags) = frame.sort.get_or_insert_with(Default::default);
                keys.push(key);
                flags.push(text);
            }
            SelectFlag::Preset(preset, text) => {
                presets.presets.push(preset);
                preset_flags.push(text);
            }
            SelectFlag::Stage(stage, text) => {
                frame.flush();
                frame
                    .stages
                    .push(OriginStage::new(stage, StageOrigin::Flag(text)));
            }
            SelectFlag::Pipeline(stages, text) => {
                frame.flush();
                for stage in stages {
                    frame
                        .stages
                        .push(OriginStage::labeled(stage, StageOrigin::Flag(text.clone())));
                }
            }
            SelectFlag::PerStart(group, text) => {
                frame.flush();
                frames.push(Frame::new(Some((group, text))));
            }
            SelectFlag::PerEnd => {
                frame.flush();
                if frames.len() == 1 {
                    return Err("`--end` closes no `--per`".to_string());
                }
                let done = frames.pop().expect("an open `--per`");
                let (group, text) = done.per.expect("a `--per` frame");
                let per = Stage::Per(PerStage {
                    group,
                    run: done.stages.into_iter().map(|s| s.stage).collect(),
                });
                frames
                    .last_mut()
                    .expect("the request's frame")
                    .stages
                    .push(OriginStage::new(
                        per,
                        StageOrigin::Flag(format!("{text} .. --end")),
                    ));
            }
        }
    }
    if frames.len() > 1 {
        let (_, text) = frames.pop().and_then(|f| f.per).expect("a `--per` frame");
        return Err(format!("`{text}` has no `--end`"));
    }
    let mut top = frames.pop().expect("the request's frame");
    top.flush();
    request.stages = top.stages;
    let prefix = preset_prefix(&presets, preset_flags);
    let composed = context
        .compose(&env, prefix, &request, false)
        .map_err(|e| e.message)?;
    let run: SelectRun = context
        .run_composed(&env, None, &composed)
        .map_err(|e| e.message)?;
    if run.table.is_empty() {
        return Err(empty_error(&run).message);
    }
    let results = run
        .table
        .rows
        .iter()
        .map(|r| {
            if kind == EntryKind::Build {
                record(&env, &r.entry.path, r.weight).map(|c| ResultRecord::Build(Box::new(c)))
            } else {
                Ok(ResultRecord::Entry(EntryRecord {
                    name: r.entry.path.to_string(),
                    kind: kind.as_str(),
                    weight: r.weight,
                }))
            }
        })
        .collect::<Result<_, String>>()?;
    let dropped = if rank {
        run.trace
            .dropped
            .iter()
            .map(|d| DroppedRecord {
                name: d.entry.path.to_string(),
                stage: d.stage,
                weight: d.weight,
                why: d.why.clone(),
            })
            .collect()
    } else {
        Vec::new()
    };
    let report = SelectionReport {
        query: composed.query,
        why: run.trace.stages,
        dropped,
        presets: presets.presets,
        results,
    };
    let yaml = serde_saphyr::to_string(&report).map_err(|e| e.to_string())?;
    emit!(out, "{}", yaml.trim_end());
    Ok(())
}

/// What an entry's tree line adds after its name and kind.
fn annotation(
    stack: &PretrainedCatalogStack,
    name: &CatalogName,
) -> String {
    let Some(entry) = stack.declaration(name) else {
        return String::new();
    };
    let note = match entry.body() {
        CatalogEntry::Namespace(_) | CatalogEntry::Contract(_) => {
            stack.group_facet(entry).map(|f| f.to_string())
        }
        CatalogEntry::Build(_) => stack
            .resolve_build(name)
            .ok()
            .and_then(|b| b.release)
            .map(|r| format!("{} {}", r.group.leaf(), r.released)),
        CatalogEntry::TemplateClass(t) => t.base.as_ref().map(|b| format!("base {b}")),
        CatalogEntry::Resource(r) => Some(r.file.clone()),
        CatalogEntry::Axis(a) => Some(format!("{:?}", a.axis).to_lowercase()),
        CatalogEntry::Tag(t) => Some(format!("{:?}", t.tag_type).to_lowercase()),
        _ => None,
    };
    note.map(|n| format!("  [{n}]")).unwrap_or_default()
}

fn tree(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    root: Option<&str>,
    kinds: &[String],
    depth: Option<usize>,
) -> Result<(), String> {
    let kinds: Vec<EntryKind> = kinds
        .iter()
        .map(|k| parse_kind(k))
        .collect::<Result<_, _>>()?;
    // Whether `name` is shown: it is of a wanted kind, or something below it
    // is.
    fn shown(
        stack: &PretrainedCatalogStack,
        name: &CatalogName,
        kinds: &[EntryKind],
    ) -> bool {
        kinds.is_empty()
            || stack
                .declaration(name)
                .is_some_and(|d| kinds.contains(&d.kind()))
            || stack
                .children(name)
                .iter()
                .any(|c| shown(stack, c.name(), kinds))
    }
    fn walk(
        out: &mut String,
        stack: &PretrainedCatalogStack,
        names: &[CatalogName],
        prefix: &str,
        kinds: &[EntryKind],
        depth: Option<usize>,
    ) {
        let names: Vec<&CatalogName> = names.iter().filter(|n| shown(stack, n, kinds)).collect();
        for (i, name) in names.iter().enumerate() {
            let last = i + 1 == names.len();
            let kind = stack
                .declaration(name)
                .map(|d| d.kind().as_str())
                .unwrap_or("?");
            emit!(
                out,
                "{prefix}{} {} ({kind}){}",
                if last { "└──" } else { "├──" },
                name.leaf(),
                annotation(stack, name)
            );
            if depth == Some(0) {
                continue;
            }
            let children: Vec<CatalogName> = stack
                .children(name)
                .iter()
                .map(|c| c.name().clone())
                .collect();
            let prefix = format!("{prefix}{}", if last { "    " } else { "│   " });
            walk(out, stack, &children, &prefix, kinds, depth.map(|d| d - 1));
        }
    }
    let (start, names): (String, Vec<CatalogName>) = match root {
        Some(query) => {
            let name = find(stack, query, &SEARCHABLE)?;
            let kids = stack
                .children(&name)
                .iter()
                .map(|c| c.name().clone())
                .collect();
            let kind = stack.declaration(&name).map(|d| d.kind()).expect("found");
            (format!("{name} ({kind}){}", annotation(stack, &name)), kids)
        }
        None => (
            "/".to_string(),
            stack
                .declarations()
                .filter(|e| e.name().scope().is_none())
                .map(|e| e.name().clone())
                .collect(),
        ),
    };
    emit!(out, "{start}");
    walk(
        out,
        stack,
        &names,
        "",
        &kinds,
        depth.map(|d| d.saturating_sub(1)),
    );
    Ok(())
}

fn releases(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    query: &str,
) -> Result<(), String> {
    let group = find(stack, query, &[EntryKind::Release])?;
    let members = stack.releases(&group).map_err(|e| e.message)?;
    let latest = members.last().map(|(n, _)| n.clone());
    for (build, released) in &members {
        let mark = if Some(build) == latest.as_ref() {
            "  (latest)"
        } else {
            ""
        };
        emit!(out, "{:<12} {build}{mark}", released.to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Runs the command line `args` (after the program name) over the
    /// example catalogs, as files.
    fn cli(args: &str) -> Result<String, String> {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../public/bunsen/src/data/pretrained/build_catalog/fixtures"
        );
        let mut argv = vec!["pretrained-catalog".to_string()];
        for file in ["prelude", "commons", "qwen3", "whisper", "supplemental"] {
            argv.push("--catalog".into());
            argv.push(format!("{dir}/{file}.yml"));
        }
        argv.extend(shell_words(args));
        let matches = Cli::command()
            .try_get_matches_from(argv)
            .map_err(|e| e.to_string())?;
        let parsed = Cli::from_arg_matches(&matches).map_err(|e| e.to_string())?;
        let mut out = String::new();
        run(parsed, &matches, &mut out)?;
        Ok(out)
    }

    /// `args` split on spaces, keeping single-quoted runs together.
    fn shell_words(args: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut word = String::new();
        let mut quoted = false;
        for c in args.chars() {
            match c {
                '\'' => quoted = !quoted,
                ' ' if !quoted => {
                    if !word.is_empty() {
                        words.push(std::mem::take(&mut word));
                    }
                }
                c => word.push(c),
            }
        }
        if !word.is_empty() {
            words.push(word);
        }
        words
    }

    #[test]
    fn test_layers_and_lists() {
        let out = cli("layers").unwrap();
        assert_eq!(out.lines().count(), 5);
        assert!(out.contains("supplemental"));
        let builds = cli("list --kind build").unwrap();
        assert_eq!(builds.lines().count(), 8);
        let families = cli("list --facet family").unwrap();
        assert!(families.contains("qwen/qwen3") && families.contains("openai/whisper"));
        let members = cli("list --members release").unwrap();
        assert_eq!(members.lines().count(), 3);
        let classes = cli("list --children qwen3 --kind template_class").unwrap();
        assert!(classes.contains("qwen/qwen3/hybrid"));
        assert!(
            cli("list --kind family")
                .unwrap_err()
                .contains("no kind `family`")
        );
    }

    #[test]
    fn test_show_tells_the_history() {
        let out = cli("show qwen").unwrap();
        assert!(out.contains("deny     in commons"));
        assert!(out.contains("update   in supplemental"));
        assert!(out.contains("facet: publisher"));
        assert!(cli("show nothing-here").is_err());
    }

    #[test]
    fn test_resolve_with_shortcuts_and_a_patch() {
        let out = cli(
            "resolve qwen3-0.6b-base --off thinking --patch '{generation: {temperature: 0.1}}'",
        )
        .unwrap();
        assert!(out.contains("(fixed)"));
        assert!(out.contains("`--off thinking` implies `--preset no-thinking`"));
        assert!(out.contains("\"temperature\":0.1"));
        let err = cli("resolve qwen3-0.6b-base --on thinking").unwrap_err();
        assert!(err.contains("fixed at `no-thinking`"), "{err}");
    }

    /// The YAML report `select ARGS` prints.
    fn report(args: &str) -> Value {
        let out = cli(&format!("select {args}")).unwrap();
        serde_saphyr::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"))
    }

    fn field(
        report: &Value,
        key: &str,
    ) -> Vec<Value> {
        report["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r[key].clone())
            .collect()
    }

    #[test]
    fn test_select_reports_the_query_why_and_the_result() {
        let all = report("qwen3 --off thinking");
        assert_eq!(
            field(&all, "name"),
            [
                "qwen/qwen3/qwen3-0.6b",
                "qwen/qwen3/qwen3-0.6b-base",
                "qwen/qwen3/qwen3-1.7b",
            ]
        );
        assert_eq!(all["presets"], json!(["no-thinking"]));
        assert_eq!(all["results"][0]["weight"], json!(1.0));
        assert_eq!(all["results"][0]["tags"]["thinking"], json!("switch"));
        assert_eq!(all["results"][0]["contract"], json!("qwen3-lm"));
        // The prefix is stage 0, attributed to the flag.
        assert_eq!(
            all["query"]["run"][0],
            json!({"keep": {"$presets": ["no-thinking"]}})
        );
        assert_eq!(all["why"][0]["origin"], json!("prefix: --off thinking"));
        assert_eq!(all["why"][0]["dropped"], json!(5));
        // The tail orders by weight when nothing else does.
        assert_eq!(all["why"][2]["origin"], json!("tail"));
    }

    #[test]
    fn test_select_flags_are_stages_in_command_line_order() {
        // Ascending parameters: turbo (1.0), v1 (0.5), v2 (0.75), v3 (1.0);
        // the cutoff drops v1, the limit v3.
        let cut = report(
            "whisper --weigh '{release: latest, step: 0.25}' --asc parameters --cutoff 0.6 \
             --limit 2 --rank",
        );
        assert_eq!(field(&cut, "weight"), [1.0, 0.75]);
        let dropped: Vec<(String, String)> = cut["dropped"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| d["name"].as_str().unwrap().starts_with("openai"))
            .map(|d| {
                (
                    d["name"]
                        .as_str()
                        .unwrap()
                        .rsplit('/')
                        .next()
                        .unwrap()
                        .to_string(),
                    d["why"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(
            dropped,
            [
                (
                    "large-v1".to_string(),
                    "fails `keep: {$weight: {ge: 0.6}}`".to_string()
                ),
                ("large-v3".to_string(), "past limit".to_string()),
            ]
        );
        assert_eq!(
            cut["results"][1]["release"],
            json!({"group": "openai/whisper/large/release", "released": "2022-12"})
        );
        // Consecutive sort flags join one `order`.
        let sorted =
            report("whisper --desc '$weight' --desc parameters --weigh '{release: latest}'");
        assert_eq!(
            sorted["query"]["run"][1],
            json!({"order": [{"desc": "$weight"}, {"desc": "parameters"}]})
        );
        let tests = report("--le parameters 8.9e8 --has language en");
        assert_eq!(
            field(&tests, "name"),
            ["openai/whisper/large/large-v3-turbo"]
        );
    }

    #[test]
    fn test_the_preset_prefix_holds_wherever_the_flags_sit() {
        // The prefix runs first, so the limit can't pick a build the preset
        // can't apply to.
        let late = report("qwen3 --limit 1 --off thinking");
        let early = report("qwen3 --off thinking --limit 1");
        assert_eq!(field(&late, "name"), ["qwen/qwen3/qwen3-0.6b"]);
        assert_eq!(field(&late, "name"), field(&early, "name"));
        // A pipeline whose own limit comes first: the same.
        let piped = report("--pipeline '[{from: qwen3}, {limit: 1}]' --off thinking");
        assert_eq!(field(&piped, "name"), ["qwen/qwen3/qwen3-0.6b"]);
        // The printed query, fed back with no preset flags, returns the same
        // rows.
        let query = serde_json::to_string(&late["query"]["run"]).unwrap();
        let again = report(&format!("--pipeline '{query}'"));
        assert_eq!(field(&again, "name"), field(&late, "name"));
        // A build named alone that can't take the presets is an error.
        let err = cli("select qwen3-4b-thinking-2507 --off thinking").unwrap_err();
        assert!(err.contains("was named, but can't be chosen"), "{err}");
        let dropped = report("whisper --preset translate --rank");
        assert_eq!(dropped["dropped"][0]["stage"], json!(0));
        assert!(
            cli("select --on task")
                .unwrap_err()
                .contains("not a bool axis")
        );
    }

    #[test]
    fn test_select_per_and_unset() {
        let largest = report("--per family --desc parameters --limit 1 --end");
        assert_eq!(
            field(&largest, "name"),
            [
                "openai/whisper/large/large-v1",
                "qwen/qwen3/qwen3-4b-thinking-2507"
            ]
        );
        assert_eq!(
            largest["query"]["run"][0]["per"]["run"],
            json!([{"order": {"desc": "parameters"}}, {"limit": 1}])
        );
        assert!(
            cli("select --per family --limit 1")
                .unwrap_err()
                .contains("has no `--end`")
        );
        assert!(
            cli("select --end")
                .unwrap_err()
                .contains("closes no `--per`")
        );
        let all = report("--defaults '[{keep: instruct, as: tuned}]' whisper --unset tuned");
        assert_eq!(field(&all, "name").len(), 4);
    }

    #[test]
    fn test_select_any_kind() {
        let classes = report("--kind template_class --keep thinking");
        assert_eq!(
            field(&classes, "name"),
            ["qwen/qwen3/hybrid", "qwen/qwen3/thinking-only"]
        );
        assert_eq!(classes["results"][0]["kind"], json!("template_class"));
        assert_eq!(classes["query"]["kind"], json!("template_class"));
        let resources = report("--kind resource qwen3-1.7b --eq '$format' json");
        assert_eq!(field(&resources, "name"), ["qwen/qwen3/qwen3-1.7b/config"]);
        assert!(
            cli("select --kind family")
                .unwrap_err()
                .contains("no kind `family`")
        );
    }

    #[test]
    fn test_select_defaults_and_refusals() {
        let defaults = report("--defaults '[{keep: instruct}, {order: {desc: parameters}}]'");
        assert_eq!(
            field(&defaults, "name")[0],
            "qwen/qwen3/qwen3-4b-thinking-2507"
        );
        assert_eq!(defaults["why"][0]["origin"], json!("defaults"));
        // One build named alone skips them.
        let one = report("large-v1 --defaults '[{keep: instruct}]'");
        assert_eq!(field(&one, "name"), ["openai/whisper/large/large-v1"]);
        let err = cli("select --keep '{kit: qwen3}'").unwrap_err();
        assert!(err.contains("nothing matches `kit`"), "{err}");
        let typed = cli("select --keep '{instruct: maybe}'").unwrap_err();
        assert!(typed.contains("is not `true` or `false`"), "{typed}");
        let empty = cli("select qwen3 --cutoff 2").unwrap_err();
        assert!(empty.contains("no build is left"), "{empty}");
        assert!(empty.contains("--cutoff 2"), "{empty}");
        let unknown = cli("select --use nothing").unwrap_err();
        assert!(unknown.contains("no named selection"), "{unknown}");
    }

    #[test]
    fn test_tree() {
        let out = cli("tree").unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "/");
        assert!(out.contains("├── qwen (namespace)  [publisher]"), "{out}");
        assert!(out.contains("qwen3 (namespace)  [family]"), "{out}");
        assert!(
            out.contains("thinking-only (template_class)  [base hybrid]"),
            "{out}"
        );
        assert!(
            out.contains("large-v3 (build)  [release 2023-11-06]"),
            "{out}"
        );
        // Children from the supplemental catalog sit under the group it
        // extends.
        assert!(out.contains("qwen3-1.7b (build)"), "{out}");

        let builds = cli("tree whisper --kind build").unwrap();
        let lines: Vec<&str> = builds.lines().collect();
        assert_eq!(lines[0], "openai/whisper (namespace)  [family]");
        assert!(lines[1].ends_with("large (namespace)"), "{builds}");
        assert_eq!(lines.len(), 6, "{builds}");
        assert!(!builds.contains("checkpoint"));
        assert!(!builds.contains("vocabulary"));

        let shallow = cli("tree --depth 1").unwrap();
        assert!(shallow.contains("qwen (namespace)"));
        assert!(!shallow.contains("qwen3 (namespace)"), "{shallow}");
        assert!(cli("tree --kind nope").is_err());
    }

    #[test]
    fn test_releases() {
        let out = cli("releases release").unwrap();
        let rows: Vec<&str> = out.lines().collect();
        assert_eq!(rows.len(), 3);
        assert!(rows[2].ends_with("(latest)"));
    }

    #[test]
    fn test_needs_catalogs() {
        let matches = Cli::command()
            .try_get_matches_from(["pretrained-catalog", "layers"])
            .unwrap();
        let parsed = Cli::from_arg_matches(&matches).unwrap();
        let err = run(parsed, &matches, &mut String::new()).unwrap_err();
        assert!(err.contains("no catalogs"));
    }
}
