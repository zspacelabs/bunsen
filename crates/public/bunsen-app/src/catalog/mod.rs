//! The build catalogs' commands: import catalogs, then dump, browse, show,
//! resolve and select over the state they make
//! ([`bunsen::data::pretrained::build_catalog`]).
//!
//! ```text
//! catalog --examples tree
//! catalog --examples dump --format json
//! catalog --catalog a.yml --catalog b.yml show qwen3
//! catalog --examples resolve large-v3 --preset translate
//! catalog --examples select qwen3 --off thinking --limit 1
//! catalog --examples select whisper --weigh '{release: latest}' --desc '$weight' --rank
//! catalog --examples releases release
//! ```
//!
//! Catalogs are imported in the order given, `--examples` (the example
//! catalogs) first when it is given.
//!
//! `select` stands in for a tool that chooses builds: it turns its preset
//! flags into the presets it applies and a prefix that keeps only the builds
//! they apply to ([`preset_prefix`]), its other flags into stages, in
//! command-line order, and composes them over its defaults
//! ([`SelectionContext::compose`]).
//!
//! `select` reads its flags' order from clap's matches, so a tool that
//! embeds [`CatalogCommand`] keeps the [`ArgMatches`] it parsed, and hands
//! [`run`](CatalogCommand::run) the matches of the command's own level.

use std::path::PathBuf;

use bunsen::{
    data::pretrained::build_catalog::{
        CatalogErrorExt,
        CatalogName,
        PretrainedCatalogStack,
        document::{
            CatalogEntry,
            EntryKind,
            OrderedNode,
        },
        fixtures,
        lookup::{
            LeafMatch,
            SEARCHABLE,
        },
        presets::{
            PresetSelection,
            ResolvedAxisUse,
        },
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
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};
use clap::{
    ArgMatches,
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

mod select;
mod tree;

pub use select::SelectArgs;
pub use tree::DumpFormat;
#[cfg(test)]
mod tests;

/// The build catalogs' commands, over the catalogs given.
#[derive(clap::Args, Debug)]
pub struct CatalogCommand {
    /// A catalog to import, in order; repeatable. YAML, or JSON by its
    /// `.json` extension.
    #[arg(long = "catalog", value_name = "PATH")]
    catalogs: Vec<PathBuf>,

    /// Import the example catalogs (prelude, commons, qwen3, whisper) first.
    #[arg(long)]
    examples: bool,

    #[command(subcommand)]
    action: CatalogAction,
}

impl CatalogCommand {
    /// Runs the command, printing its output to stdout.
    ///
    /// `matches` are the matches clap parsed at the level of this command:
    /// the ones whose subcommand is the action.
    ///
    /// # Errors
    /// A [`Policy`](bunsen::errors::BunsenErrorKind::Policy) error for a
    /// catalog that doesn't import, or a request the catalogs refuse.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        let mut out = String::new();
        let result = self.run_into(matches, &mut out);
        print!("{out}");
        result.map_err(BunsenError::policy)
    }

    /// Runs the command, writing its output to `out`.
    ///
    /// # Errors
    /// What went wrong, for a person.
    pub fn run_into(
        &self,
        matches: &ArgMatches,
        out: &mut String,
    ) -> Result<(), String> {
        let stack = load(self)?;
        match &self.action {
            CatalogAction::Layers => layers(out, &stack),
            CatalogAction::List {
                kind,
                facet,
                members,
                children,
            } => list(
                out,
                &stack,
                kind.clone(),
                facet.clone(),
                members.clone(),
                children.clone(),
            )?,
            CatalogAction::Show { name } => show(out, &stack, name)?,
            CatalogAction::Resolve {
                build,
                presets,
                patch,
            } => resolve(out, &stack, build, presets, patch.as_deref())?,
            CatalogAction::Select(args) => {
                let matches = matches
                    .subcommand_matches("select")
                    .expect("the select subcommand");
                select::select(out, &stack, matches, (**args).clone())?
            }
            CatalogAction::Dump { format } => tree::dump(out, &stack, *format)?,
            CatalogAction::Tree {
                root,
                kinds,
                all,
                depth,
            } => tree::tree(out, &stack, root.as_deref(), kinds, *all, *depth)?,
            CatalogAction::Releases { group } => releases(out, &stack, group)?,
        }
        Ok(())
    }
}

/// A `catalog` subcommand.
#[derive(Subcommand, Debug)]
pub enum CatalogAction {
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
    Select(Box<select::SelectArgs>),

    /// The imported catalogs, in import order, as one sequence: each
    /// item's `layer`, `path` and `catalog`, the document in its keyed form.
    Dump {
        /// The output format.
        #[arg(long, value_enum, default_value_t = DumpFormat::Yml)]
        format: DumpFormat,
    },

    /// The namespace as a directory view, from the root's entries or from
    /// one entry: `name: description` for a namespace, `name: aliases {axes}
    /// description` for a build, `name -> target` for an alias. By default
    /// it shows namespaces, builds and the aliases that name them.
    Tree {
        /// The entry to start from, by any unambiguous name; the root when
        /// absent.
        root: Option<String>,
        /// Show only entries of these kinds (repeatable), with the
        /// containers on the way to them.
        #[arg(long = "kind", value_name = "KIND", conflicts_with = "all")]
        kinds: Vec<String>,
        /// Show every kind: contracts, releases, template classes,
        /// resources, tags, axes and facets too.
        #[arg(long)]
        all: bool,
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

/// The preset flags: the presets to apply, by name or by a bool axis's
/// member.
#[derive(clap::Args, Clone, Debug)]
pub struct PresetArgs {
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

fn errors(e: BunsenError) -> String {
    let lines = e.problem_lines();
    if lines.is_empty() {
        e.to_string()
    } else {
        lines
    }
}

fn load(cli: &CatalogCommand) -> Result<PretrainedCatalogStack, String> {
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
    EntryKind::from_name(kind).ok_or_else(|| {
        let known: Vec<&str> = EntryKind::ALL.iter().map(|k| k.as_str()).collect();
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
        .map_err(|e| e.problem_message())
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
        let group = find(stack, &group, &[EntryKind::Namespace, EntryKind::Release])?;
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
    let mut kinds = SEARCHABLE.to_vec();
    kinds.push(EntryKind::Contract);
    let name = find(stack, query, &kinds)?;
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
    let aliases = stack.aliases_of(&name);
    if !aliases.is_empty() {
        emit!(out, "aliases:");
        for alias in aliases {
            emit!(out, "  {alias}");
        }
    }
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
    if let Some(contract) = b.contract.as_ref().and_then(CatalogName::contract_path) {
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
    emit!(out, "  resources");
    for (slot, bound) in &b.resources {
        emit!(
            out,
            "    {slot:<18} {} ({})",
            bound.resource,
            bound.entry.file
        );
    }
    if !b.args.is_empty() {
        emit!(out, "  args");
        for (slot, bound) in &b.args {
            emit!(out, "    {slot:<18} {}", bound.value);
        }
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
            .map_err(|e| format!("`{flag}` doesn't apply: {}", e.problem_message()))?;
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
        .map_err(|e| e.problem_message())?;
    emit!(out, "  applied      {}", applied.applied.join(", "));
    emit!(out, "  document     {}", applied.document);
    Ok(())
}

fn releases(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    query: &str,
) -> Result<(), String> {
    let group = find(stack, query, &[EntryKind::Release])?;
    let members = stack.releases(&group).map_err(|e| e.problem_message())?;
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
