use bunsen::{
    errors::*,
    support::testing::PerformanceBackend,
};
use burn::prelude::Backend;
use clap::Parser;

pub mod commands;
mod whisper_clap;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    pub command: Commands,
}

fn main() -> BunsenResult<()> {
    let args = Args::parse();
    args.command.run::<PerformanceBackend>()
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Transcribe audio.
    Transcribe(commands::transcribe_cmd::TranscribeCmd),
}

impl Commands {
    pub fn run<B: Backend>(&self) -> BunsenResult<()> {
        match self {
            Commands::Transcribe(cmd) => cmd.run::<B>(),
        }
    }
}
