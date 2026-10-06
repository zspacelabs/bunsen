use bunsen::errors::*;
use clap::Parser;

pub mod commands;
mod resample;
mod whisper_clap;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    pub command: Commands,
}

fn main() -> BunsenResult<()> {
    let args = Args::parse();
    args.command.run()
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Transcribe audio files.
    Transcribe(commands::transcribe_cmd::TranscribeCmd),

    /// Transcribe the microphone, live.
    Live(commands::live_cmd::LiveCmd),

    /// List, fetch and inspect the models `--model` can name.
    Models(commands::models_cmd::ModelsCmd),
}

impl Commands {
    pub fn run(&self) -> BunsenResult<()> {
        match self {
            Commands::Transcribe(cmd) => cmd.run(),
            Commands::Live(cmd) => cmd.run(),
            Commands::Models(cmd) => cmd.run(),
        }
    }
}
