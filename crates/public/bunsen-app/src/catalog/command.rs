use std::path::PathBuf;

use bunsen::{
    data::catalog::BunsenCatalogStack,
    errors::{
        BunsenError,
        BunsenResult,
    },
};
use clap::{
    ArgMatches,
    Subcommand,
};

use super::{
    AliasView,
    DumpFormat,
    inspect::*,
    resolve::resolve,
    select,
    support::*,
    tree,
};

/// The build catalogs' commands, over a base catalog the binary gives and
/// the extensions the command line adds.
///
/// The base is the binary's: bunsen-tool's is the built-in catalog
/// ([`builtin_stack`](bunsen::data::catalog::builtin_stack)).
/// An extension is imported over it, so it may add builds, template classes,
/// groups and resources, and update what the base declares, but never
/// declare contracts.
#[derive(clap::Args, Debug)]
pub struct CatalogCommand {
    /// An extension catalog to import over the base; repeatable.
    ///
    /// Imported in order, each over the ones before. YAML, or JSON by its
    /// `.json` extension.
    #[arg(long = "catalog", value_name = "PATH")]
    catalogs: Vec<PathBuf>,

    #[command(subcommand)]
    action: CatalogAction,
}

impl CatalogCommand {
    /// Runs the command over `base` and the extensions, printing its output to
    /// stdout.
    ///
    /// `matches` are the matches clap parsed at the level of this command:
    /// the ones whose subcommand is the action.
    ///
    /// # Errors
    /// A [`Policy`](bunsen::errors::BunsenErrorKind::Policy) error for an
    /// extension that doesn't import, or a request the catalogs refuse.
    pub fn run(
        &self,
        matches: &ArgMatches,
        base: BunsenCatalogStack,
    ) -> BunsenResult<()> {
        let mut out = String::new();
        let result = self.run_into(matches, base, &mut out);
        print!("{out}");
        result.map_err(BunsenError::policy)
    }

    /// Runs the command over `base` and the extensions, writing its output
    /// to `out`.
    ///
    /// # Errors
    /// What went wrong, for a person.
    pub fn run_into(
        &self,
        matches: &ArgMatches,
        base: BunsenCatalogStack,
        out: &mut String,
    ) -> Result<(), String> {
        let stack = extend(base, &self.catalogs)?;
        match &self.action {
            CatalogAction::Layers => layers(out, &stack),
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
                aliases,
                depth,
            } => tree::tree(out, &stack, root.as_deref(), kinds, *all, *aliases, *depth)?,
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

    /// Select builds through a pipeline of stages, printed as YAML.
    ///
    /// The output: the composed query, why, the presets applied, and the
    /// result in order, each build resolved with its weight. Each flag is one
    /// stage, in command-line order, but `--on`, `--off` and `--preset`: they
    /// are the presets applied, and the prefix that keeps only the builds that
    /// can take them.
    Select(Box<select::SelectArgs>),

    /// The imported catalogs, in import order, as one sequence.
    ///
    /// Each item's `layer`, `path` and `catalog`, the document in its keyed
    /// form.
    Dump {
        /// The output format.
        #[arg(long, value_enum, default_value_t = DumpFormat::Yml)]
        format: DumpFormat,
    },

    /// The namespace as a directory view, from the root's entries or one entry.
    ///
    /// `name: description` for a namespace, `name: aliases {axes}
    /// description` for a build, `name -> target` for an alias. By default
    /// it shows namespaces, builds and the aliases that name them.
    Tree {
        /// The entry to start from, by any unambiguous name; the root when
        /// absent.
        root: Option<String>,
        /// Show only entries of these kinds (repeatable).
        ///
        /// The containers on the way to them show too.
        #[arg(long = "kind", value_name = "KIND", conflicts_with = "all")]
        kinds: Vec<String>,
        /// Show every kind of entry.
        ///
        /// Contracts, releases, template classes, resources, tags, axes and
        /// facets too.
        #[arg(long)]
        all: bool,
        /// Where aliases show: on their build's line, as entries, both, or not
        /// at all.
        ///
        /// On their build's line (`inline`), as entries where they sit
        /// (`entries`), both, or not at all.
        #[arg(long, value_enum, default_value_t = AliasView::Both)]
        aliases: AliasView,
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
#[derive(clap::Args, Clone, Debug, Default)]
pub struct PresetArgs {
    /// A preset to select; repeatable.
    #[arg(long = "preset", value_name = "NAME")]
    pub(super) presets: Vec<String>,
    /// Select the positive member of a bool axis (`--on thinking`).
    #[arg(long = "on", value_name = "AXIS")]
    pub(super) on: Vec<String>,
    /// Select the negative member of a bool axis (`--off thinking`).
    #[arg(long = "off", value_name = "AXIS")]
    pub(super) off: Vec<String>,
}
