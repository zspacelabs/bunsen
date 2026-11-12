//! # `cargo catalog`: builds bunsen's built-in catalog
//!
//! ```text
//! cargo catalog build  [--kit K]... [--verify-remote]
//! cargo catalog check  [--kit K]... [--verify-remote]
//! cargo catalog update [--kit K]... [--verify-remote]
//! ```
//!
//! The sources are the probe contracts in `probe_contracts/`: a
//! `MANIFEST.txt` listing the layers in import order, with the bunsen
//! feature each needs, and one keyed catalog layer per line, the prelude
//! first. A source writes what a person decides: the namespaces, builds,
//! template classes, presets and resources. The builder adds what is read
//! ([`layer`]): every kit's contract, into the prelude's `$contracts`
//! ([`catalog_contracts`](bunsen::kits::support::catalog_contracts)); each
//! file's digest and size, from the Hub's listing or the file; a
//! checkpoint's dtype, from its `config.json`; and the presets the probes
//! find, from the files they read. Every built layer imports into one stack,
//! or the build fails.
//!
//! - `build` writes the catalog into the validation workspace's
//!   `target/catalog-builder/catalog/`.
//! - `check` builds, then fails when the checked-in catalog,
//!   `crates/public/bunsen-catalog/catalog/`, differs: a change to a source, to
//!   the kits' contracts or probes, to the builder, or to a remote.
//! - `update` builds, then replaces the checked-in catalog with the build.
//!
//! Each layer's build is kept in `target/catalog-builder/layers/`, keyed by
//! its source, the builder's version and, for the prelude, the contracts:
//! a layer whose key is there is not built again, so `check` and `update`
//! need the network only for what changed. `--kit K` builds the layer `K`
//! from the remotes again, whatever the cache holds; `--verify-remote`
//! fetches every file built and checks it hashes to its pin.
//!
//! Nothing else in the validation workspace uses this crate: the kits and
//! their checks read the built catalog, through bunsen.

use std::process::ExitCode;

use clap::{
    Args,
    Parser,
    Subcommand,
};

pub mod catalog;
pub mod layer;
pub mod remote;
pub mod tree;

/// Builds bunsen's built-in catalog from its probe contracts.
#[derive(Debug, Parser)]
#[command(name = "cargo catalog")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build the catalog into `target/catalog-builder/catalog/`.
    Build(BuildArgs),

    /// Build, then fail if the checked-in catalog differs.
    Check(BuildArgs),

    /// Build, then replace the checked-in catalog.
    Update(BuildArgs),
}

#[derive(Debug, Args)]
struct BuildArgs {
    /// Build this layer from the remotes again, whatever the cache holds.
    #[arg(long = "kit", value_name = "K")]
    kits: Vec<String>,

    /// Fetch every file built, and check it hashes to its pin.
    #[arg(long)]
    verify_remote: bool,
}

impl BuildArgs {
    fn options(&self) -> catalog::BuildOptions {
        catalog::BuildOptions {
            fresh: self.kits.clone(),
            all: self.verify_remote && self.kits.is_empty(),
            verify_remote: self.verify_remote,
        }
    }
}

fn run(cli: Cli) -> Result<bool, String> {
    let paths = catalog::Paths::of_checkout();
    let remote = remote::Http::new();
    let (args, after) = match &cli.command {
        Command::Build(args) => (args, None),
        Command::Check(args) => (args, Some(false)),
        Command::Update(args) => (args, Some(true)),
    };
    let built = catalog::build(&remote, &paths, &args.options())?;
    catalog::write_dir(&built, &paths.built())?;
    eprintln!("built: {}", paths.built().display());
    match after {
        None => Ok(true),
        Some(false) => {
            let differ = catalog::differences(&built, &paths.checked_in)?;
            if differ.is_empty() {
                eprintln!("{} is current", paths.checked_in.display());
                return Ok(true);
            }
            eprintln!("{} is out of date:", paths.checked_in.display());
            for line in differ {
                eprintln!("  {line}");
            }
            eprintln!("`cargo catalog update` replaces it with the build");
            Ok(false)
        }
        Some(true) => {
            catalog::write_dir(&built, &paths.checked_in)?;
            eprintln!("wrote {}", paths.checked_in.display());
            Ok(true)
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
