//! The `diarize` command group.

use bunsen::errors::BunsenResult;

use super::{
    asr_diarize_live::AsrDiarizeLiveCommand,
    asr_diarize_run::AsrDiarizeRunCommand,
};
use crate::asr::{
    AsrKit,
    AsrKitFetchCommand,
    AsrKitListCommand,
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
            AsrDiarizeSubCommand::List(list) => list.run(AsrKit::Diarizer),
            AsrDiarizeSubCommand::Fetch(fetch) => fetch.run(AsrKit::Diarizer),
            AsrDiarizeSubCommand::Run(run) => run.run(),
            #[cfg(feature = "live")]
            AsrDiarizeSubCommand::Live(live) => live.run(),
        }
    }
}
