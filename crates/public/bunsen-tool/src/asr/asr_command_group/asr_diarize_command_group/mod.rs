//! `asr diarize`: the command group.

mod asr_diarize_args;
#[cfg(feature = "live")]
mod asr_diarize_live;
mod asr_diarize_run;
mod support;

pub use asr_diarize_args::*;
#[cfg(feature = "live")]
pub use asr_diarize_live::*;
pub use asr_diarize_run::*;
use bunsen::errors::BunsenResult;
pub use support::load_diarizer;
use support::{
    JsonDiarization,
    JsonTurn,
    speaker_totals,
    text_line,
};

use crate::asr::{
    AsrKitFetchCommand,
    AsrKitListCommand,
    Kit,
};

/// The diarizer's commands.
#[derive(clap::Args, Debug)]
pub struct AsrDiarizeCommandGroup {
    #[command(subcommand)]
    action: AsrDiarizeSubCommand,
}

#[derive(clap::Subcommand, Debug)]
enum AsrDiarizeSubCommand {
    /// List the models `--model` can name, their licences and files.
    ///
    /// Each file is marked with whether it is cached.
    List(AsrKitListCommand),

    /// Bring every file of the named models into the weights cache.
    Fetch(AsrKitFetchCommand),

    /// Diarize audio files: each one's speaker turns.
    Run(AsrDiarizeRunCommand),

    /// Diarize the microphone: each speaker turn as it closes.
    #[cfg(feature = "live")]
    Live(AsrDiarizeLiveCommand),
}

impl AsrDiarizeCommandGroup {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.action {
            AsrDiarizeSubCommand::List(list) => list.run(Kit::Diarizer),
            AsrDiarizeSubCommand::Fetch(fetch) => fetch.run(Kit::Diarizer),
            AsrDiarizeSubCommand::Run(run) => run.run(),
            #[cfg(feature = "live")]
            AsrDiarizeSubCommand::Live(live) => live.run(),
        }
    }
}

#[cfg(test)]
mod tests;
