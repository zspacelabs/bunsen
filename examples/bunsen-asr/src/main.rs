//! `phonon`: speech to text with `FermionResearch`'s Phonon models on
//! bunsen: Phonon-2 (Parakeet TDT) by default, and Phonon-1 (Qwen3-ASR).

use bunsen::errors::BunsenResult;
use clap::Parser;

mod args;
mod models_cmd;
mod transcribe_cmd;
mod transcriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    pub command: Commands,
}

fn main() -> BunsenResult<()> {
    Args::parse().command.run()
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Transcribe audio files.
    Transcribe(transcribe_cmd::TranscribeCmd),

    /// List and fetch the models `--model` can name.
    Models(models_cmd::ModelsCmd),
}

impl Commands {
    fn run(&self) -> BunsenResult<()> {
        match self {
            Commands::Transcribe(cmd) => cmd.run(),
            Commands::Models(cmd) => cmd.run(),
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    /// The flags of every subcommand are well formed.
    #[test]
    fn test_cli_is_well_formed() {
        Args::command().debug_assert();
    }
}
