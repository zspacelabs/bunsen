//! `bunsen-tool`: bunsen's models on the command line. `bunsen-tool asr`
//! is speech to text, by preset or by model family, and who spoke when;
//! `bunsen-tool chat` chats with a Qwen3, Qwen3.5, Gemma or nanochat
//! checkpoint; `bunsen-tool decide` answers typed questions about a context
//! with a Laya decision model; `bunsen-tool cache <path|list|clear>` manages
//! the weights cache every command loads through; `bunsen-tool card` describes
//! the build, its cache and the models it knows; `bunsen-tool catalog` imports
//! build catalogs, and dumps, browses, resolves and selects over them.

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
    summary::SummaryCommand,
    support::did_you_mean,
};
use clap::{
    ArgMatches,
    CommandFactory,
    FromArgMatches,
    Parser,
};

/// bunsen's models on the command line.
///
/// `bunsen-tool asr` is speech to text, by preset or by model family, and who
/// spoke when; `bunsen-tool chat` chats with a Qwen3, Qwen3.5, Gemma or
/// nanochat checkpoint; `bunsen-tool decide` answers typed questions about a
/// context with a Laya decision model; `bunsen-tool cache` manages the weights
/// cache every command loads through; `bunsen-tool card` describes the build,
/// its cache and the models it knows; `bunsen-tool devices` lists the backends
/// and their devices; `bunsen-tool catalog` browses, resolves and selects over
/// the model catalog.
#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Speech to text, by preset or by model family, and who spoke when.
    ///
    /// By preset (`asr transcribe`, `asr live`) or by model family (`asr
    /// family`), and who spoke when (`asr diarize`).
    #[cfg(feature = "speech")]
    Asr(asr::AsrCommandGroup),

    /// Chat with a Qwen3, Qwen3.5, Gemma or nanochat checkpoint.
    ///
    /// One prompt (`chat run`), a conversation (`chat repl`), and the
    /// checkpoints (`chat list`, `chat fetch`); `chat family` for one family's.
    Chat(Box<ChatCommandGroup>),

    /// Typed questions about a context, answered by a Laya decision model.
    ///
    /// Answered with calibrated probabilities: a request (YAML or JSON) in
    /// (`decide run`), one result per question out; and the checkpoints
    /// (`decide list`, `decide fetch`).
    Decide(DecideCommandGroup),

    /// The weights cache every command loads through.
    ///
    /// Where it is, what it holds (each file under its resource's full name:
    /// `qwen/qwen3/qwen3-0.6b/checkpoint`, an `hf:` repo's
    /// `hf/<org>/<repo>/<slot>`), and clearing it by name (`qwen`,
    /// `hf/openai`).
    Cache(CacheCommand),

    /// The built-in model catalog: browse it, resolve builds, select them.
    ///
    /// bunsen's built-in catalog, with any `--catalog` extensions imported over
    /// it in order: the state as a directory view (`tree`), each catalog as
    /// written (`dump`), one entry (`show`), a resolved build (`resolve`), and
    /// selection pipelines, each stage explained (`select`).
    Catalog(CatalogCommand),

    /// What this build is, its weights cache and the models it knows, as YAML
    /// or JSON.
    ///
    /// What this build is (`build`: its version and features), its weights
    /// cache (`config`), and the models it knows by name (`kits`: by kind and
    /// family, a name and a description each).
    Card(CardCommand),

    /// The builds this binary runs, in brief: by kind, family and namespace.
    ///
    /// An ASCII bulleted list: each kind (`asr`, `chat`, `decide`), its
    /// families, their builds nested by catalog namespace, each with its
    /// size, dtypes, preset axes (the default starred) and aliases, and its
    /// description wrapped below it.
    Summary(SummaryCommand),

    /// The backends this build can run on and their default devices, as YAML or
    /// JSON.
    ///
    /// Each backend's default device, and what the device does with each float
    /// dtype (store, compute, accelerate).
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

/// The matches of the subcommand `matches` ends in: the leaf command's
/// own, whose flags it reads in command-line order.
fn leaf(matches: &ArgMatches) -> &ArgMatches {
    let mut at = matches;
    while let Some((_, sub)) = at.subcommand() {
        at = sub;
    }
    at
}

/// Runs the command `args`, parsed from `matches`: `catalog select`, `chat`
/// and `decide` read their selection flags' order from them.
fn run(
    args: &Args,
    matches: &ArgMatches,
) -> BunsenResult<()> {
    bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
    match &args.command {
        #[cfg(feature = "speech")]
        Command::Asr(cmd) => cmd.run(),
        Command::Chat(cmd) => cmd.run(leaf(matches)),
        Command::Decide(cmd) => cmd.run(leaf(matches)),
        Command::Cache(cmd) => cmd.run(),
        Command::Catalog(cmd) => cmd.run(
            matches
                .subcommand_matches("catalog")
                .expect("the catalog subcommand"),
            bunsen::data::catalog::builtin_stack()?,
        ),
        Command::Card(cmd) => cmd.run(),
        Command::Summary(cmd) => cmd.run(),
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

    /// Every command's and argument's short help is one summary of at most
    /// [`SUMMARY`] characters: clap's derive takes a doc comment's first
    /// paragraph, up to its first blank line, as the short help (`-h`, and
    /// the command and flag lists), and the whole comment as the long help
    /// (`--help`).
    #[test]
    fn test_help_summaries_are_short() {
        const SUMMARY: usize = 80;
        fn walk(
            command: &clap::Command,
            path: &str,
            long: &mut Vec<String>,
        ) {
            let path = format!("{path} {}", command.get_name()).trim().to_string();
            let mut check = |what: String, text: String| {
                if text.chars().count() > SUMMARY {
                    long.push(format!("{what} ({}): {text}", text.chars().count()));
                }
            };
            if let Some(about) = command.get_about() {
                check(path.clone(), about.to_string());
            }
            for arg in command.get_arguments() {
                if let Some(help) = arg.get_help() {
                    check(format!("{path} {}", arg.get_id()), help.to_string());
                }
            }
            for sub in command.get_subcommands() {
                walk(sub, &path, long);
            }
        }
        let mut long = Vec::new();
        walk(&Args::command(), "", &mut long);
        long.sort();
        long.dedup();
        assert!(
            long.is_empty(),
            "{} long summaries:\n{}",
            long.len(),
            long.join("\n")
        );
    }

    /// `catalog` runs over the built-in catalog, at its own level of the
    /// matches, so `select` sees its flags in command-line order.
    #[test]
    fn test_catalog_runs_through_its_matches() {
        let argv = [
            "bunsen-tool".to_string(),
            "catalog".into(),
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
        cmd.run_into(
            matches.subcommand_matches("catalog").unwrap(),
            bunsen::data::catalog::builtin_stack().unwrap(),
            &mut out,
        )
        .unwrap();
        let results: Vec<&str> = out.lines().filter(|l| l.starts_with("- name:")).collect();
        assert_eq!(results, ["- name: qwen/qwen3/qwen3-8b"], "{out}");
    }
}
