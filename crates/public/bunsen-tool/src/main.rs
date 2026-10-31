//! `bunsen-tool`: bunsen's models on the command line. `bunsen-tool asr`
//! is speech to text, by profile or by model family, and who spoke when;
//! `bunsen-tool cache <path|list|clear>` manages the weights cache every
//! command loads through.

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

    /// The weights cache every command loads through: where it is, what it
    /// holds (by kit: `parakeet`, `qwen3-asr`, `voxtral`, `whisper`,
    /// `sortformer`, `silero_vad`), and clearing it.
    Cache(bunsen_app::cache::CacheCommand),
}

fn main() -> BunsenResult<()> {
    bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
    match &Args::parse().command {
        Command::Asr(cmd) => cmd.run(),
        Command::Cache(cmd) => cmd.run(),
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
