//! `bunsen-tool`: bunsen's models on the command line. `bunsen-tool asr`
//! is speech to text, by preset or by model family, and who spoke when;
//! `bunsen-tool chat` chats with a Qwen3 or nanochat checkpoint; `bunsen-tool
//! decide` answers typed questions about a context with a Laya decision
//! model; `bunsen-tool cache <path|list|clear>` manages the weights cache every
//! command loads through; `bunsen-tool card` describes the build, its cache and
//! the models it knows; `bunsen-tool catalog` imports build catalogs, and
//! dumps, browses, resolves and selects over them.

use std::process::ExitCode;

use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::CacheCommand,
    catalog::CatalogCommand,
};
#[cfg(feature = "speech")]
use bunsen_tool::asr;
use bunsen_tool::{
    card::CardCommand,
    chat::ChatCommandGroup,
    decide::DecideCommandGroup,
    devices::DevicesCommand,
    support::did_you_mean,
};
use clap::{
    ArgMatches,
    CommandFactory,
    FromArgMatches,
    Parser,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Speech to text, by preset (`asr transcribe`, `asr live`) or by model
    /// family (`asr family`), and who spoke when (`asr diarize`).
    #[cfg(feature = "speech")]
    Asr(asr::AsrCommandGroup),

    /// Chat with a Qwen3 or nanochat checkpoint: one prompt (`chat run`), a
    /// conversation (`chat repl`), and the checkpoints (`chat list`, `chat
    /// fetch`); `chat family` for one family's.
    Chat(ChatCommandGroup),

    /// Typed questions about a context, answered with calibrated
    /// probabilities by a Laya decision model: a request (YAML or JSON) in
    /// (`decide run`), one result per question out; and the checkpoints
    /// (`decide list`, `decide fetch`).
    Decide(DecideCommandGroup),

    /// The weights cache every command loads through: where it is, what it
    /// holds (by kit: `parakeet`, `qwen3-asr`, `voxtral`, `whisper`,
    /// `sortformer`, `silero_vad`, `qwen3`, `nanochat`, `laya`), and clearing
    /// it.
    Cache(CacheCommand),

    /// Build catalogs, imported in order from `--catalog` files (or the
    /// examples): the state as a directory view (`tree`), each catalog as
    /// written (`dump`), one entry (`show`), a resolved build (`resolve`),
    /// and selection pipelines, each stage explained (`select`).
    Catalog(CatalogCommand),

    /// What this build is (`build`: its version and features), its weights
    /// cache (`config`), and the models it knows by name (`kits`: by kind and
    /// family, a name and a description each), as YAML or JSON.
    Card(CardCommand),

    /// The backends this build can run on, each one's default device, and
    /// what the device does with each float dtype (store, compute,
    /// accelerate), as YAML or JSON.
    Devices(DevicesCommand),
}

/// Runs the command; a failure is printed as its report, then, for a name
/// a table does not have, the names nearest it ([`did_you_mean`]).
fn main() -> ExitCode {
    let matches = Args::command().get_matches();
    let args = match Args::from_arg_matches(&matches) {
        Ok(args) => args,
        Err(error) => error.exit(),
    };
    match run(&args, &matches) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:?}");
            if let Some(hint) = did_you_mean::hint(&error) {
                eprintln!("{hint}");
            }
            ExitCode::FAILURE
        }
    }
}

/// Runs the command `args`, parsed from `matches`: `catalog select` reads
/// its flags' order from them.
fn run(
    args: &Args,
    matches: &ArgMatches,
) -> BunsenResult<()> {
    bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
    match &args.command {
        #[cfg(feature = "speech")]
        Command::Asr(cmd) => cmd.run(),
        Command::Chat(cmd) => cmd.run(),
        Command::Decide(cmd) => cmd.run(),
        Command::Cache(cmd) => cmd.run(),
        Command::Catalog(cmd) => cmd.run(
            matches
                .subcommand_matches("catalog")
                .expect("the catalog subcommand"),
        ),
        Command::Card(cmd) => cmd.run(),
        Command::Devices(cmd) => cmd.run(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flags of every subcommand are well formed.
    #[test]
    fn test_cli_is_well_formed() {
        Args::command().debug_assert();
    }

    /// `decide`'s commands parse.
    #[test]
    fn test_decide_parses() {
        for argv in [
            &[
                "bunsen-tool",
                "decide",
                "run",
                "--model",
                "laya-multilingual",
                "--format",
                "json",
                "req.yml",
            ][..],
            &["bunsen-tool", "decide", "run"],
            &["bunsen-tool", "decide", "list", "--urls"],
            &["bunsen-tool", "decide", "fetch", "laya"],
        ] {
            Args::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        }
    }

    /// The weights cache's commands parse.
    #[test]
    fn test_cache_parses() {
        for action in [
            &["path"][..],
            &["list", "whisper"],
            &["clear", "qwen3-asr/fermion", "--yes"],
        ] {
            let mut argv = vec!["bunsen-tool", "cache", "--cache-dir", "/tmp/c"];
            argv.extend_from_slice(action);
            Args::try_parse_from(argv).unwrap();
        }
    }

    /// `catalog` runs at its own level of the matches, so `select` sees its
    /// flags in command-line order.
    #[test]
    fn test_catalog_runs_through_its_matches() {
        let fixtures = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../bunsen/src/data/pretrained/build_catalog/fixtures"
        );
        let argv = [
            "bunsen-tool".to_string(),
            "catalog".into(),
            "--catalog".into(),
            format!("{fixtures}/prelude.yml"),
            "--catalog".into(),
            format!("{fixtures}/commons.yml"),
            "--catalog".into(),
            format!("{fixtures}/qwen3.yml"),
            "select".into(),
            "qwen3".into(),
            "--off".into(),
            "thinking".into(),
            "--desc".into(),
            "parameters".into(),
            "--limit".into(),
            "1".into(),
        ];
        let matches = Args::command().try_get_matches_from(argv).unwrap();
        let args = Args::from_arg_matches(&matches).unwrap();
        let Command::Catalog(cmd) = &args.command else {
            panic!("the catalog command")
        };
        let mut out = String::new();
        cmd.run_into(matches.subcommand_matches("catalog").unwrap(), &mut out)
            .unwrap();
        let results: Vec<&str> = out.lines().filter(|l| l.starts_with("- name:")).collect();
        assert_eq!(results, ["- name: qwen/qwen3/qwen3-0.6b"], "{out}");
    }
}
