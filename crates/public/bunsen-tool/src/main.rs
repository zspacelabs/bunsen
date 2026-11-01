//! `bunsen-tool`: bunsen's models on the command line. `bunsen-tool asr`
//! is speech to text, by profile or by model family, and who spoke when;
//! `bunsen-tool chat` chats with a Qwen3 or nanochat checkpoint; `bunsen-tool
//! cache <path|list|clear>` manages the weights cache every command loads
//! through; `bunsen-tool card` describes the build, its cache and the models
//! it knows.

use bunsen::errors::BunsenResult;
use clap::Parser;

mod cli;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Speech to text, by profile (`asr transcribe`, `asr live`) or by model
    /// family (`asr family`), and who spoke when (`asr diarize`).
    Asr(cli::asr::AsrCmd),

    /// Chat with a Qwen3 or nanochat checkpoint: one prompt (`chat run`), a
    /// conversation (`chat chat`), and the checkpoints (`chat list`, `chat
    /// fetch`); `chat family` for one family's.
    Chat(cli::chat::ChatCmd),

    /// The weights cache every command loads through: where it is, what it
    /// holds (by kit: `parakeet`, `qwen3-asr`, `voxtral`, `whisper`,
    /// `sortformer`, `silero_vad`, `qwen3`, `nanochat`), and clearing it.
    Cache(bunsen_app::cache::CacheCommand),

    /// What this build is (`build`: its version and features), its weights
    /// cache (`config`), and the models it knows by name (`kits`: by kind and
    /// family, a name and a description each), as YAML or JSON.
    Card(cli::card::CardCmd),
}

fn main() -> BunsenResult<()> {
    bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
    match &Args::parse().command {
        Command::Asr(cmd) => cmd.run(),
        Command::Chat(cmd) => cmd.run(),
        Command::Cache(cmd) => cmd.run(),
        Command::Card(cmd) => cmd.run(),
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
