//! `bunsen-tool`: bunsen's models on the command line. `bunsen-tool asr`
//! is speech to text, by preset or by model family, and who spoke when;
//! `bunsen-tool chat` chats with a Qwen3 or nanochat checkpoint; `bunsen-tool
//! decide` answers typed questions about a context with a Laya decision
//! model; `bunsen-tool cache <path|list|clear>` manages the weights cache every
//! command loads through; `bunsen-tool card` describes the build, its cache and
//! the models it knows.

use std::process::ExitCode;

use bunsen::errors::BunsenResult;
#[cfg(feature = "speech")]
use bunsen_tool::asr;
use bunsen_tool::{
    card,
    chat,
    decide,
    devices,
    support::did_you_mean,
};
use clap::Parser;

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
    Asr(asr::cli::AsrCmd),

    /// Chat with a Qwen3 or nanochat checkpoint: one prompt (`chat run`), a
    /// conversation (`chat repl`), and the checkpoints (`chat list`, `chat
    /// fetch`); `chat family` for one family's.
    Chat(chat::cli::ChatCmd),

    /// Typed questions about a context, answered with calibrated
    /// probabilities by a Laya decision model: a request (YAML or JSON) in
    /// (`decide run`), one result per question out; and the checkpoints
    /// (`decide list`, `decide fetch`).
    Decide(decide::cli::DecideCmd),

    /// The weights cache every command loads through: where it is, what it
    /// holds (by kit: `parakeet`, `qwen3-asr`, `voxtral`, `whisper`,
    /// `sortformer`, `silero_vad`, `qwen3`, `nanochat`, `laya`), and clearing
    /// it.
    Cache(bunsen_app::cache::CacheCommand),

    /// What this build is (`build`: its version and features), its weights
    /// cache (`config`), and the models it knows by name (`kits`: by kind and
    /// family, a name and a description each), as YAML or JSON.
    Card(card::CardCmd),

    /// The backends this build can run on, each one's default device, and
    /// what the device does with each float dtype (store, compute,
    /// accelerate), as YAML or JSON.
    Devices(devices::DevicesCmd),
}

/// Runs the command; a failure is printed as its report, then, for a name
/// a table does not have, the names nearest it ([`did_you_mean`]).
fn main() -> ExitCode {
    match run(&Args::parse()) {
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

fn run(args: &Args) -> BunsenResult<()> {
    bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
    match &args.command {
        #[cfg(feature = "speech")]
        Command::Asr(cmd) => cmd.run(),
        Command::Chat(cmd) => cmd.run(),
        Command::Decide(cmd) => cmd.run(),
        Command::Cache(cmd) => cmd.run(),
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
        use clap::CommandFactory;
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
}
