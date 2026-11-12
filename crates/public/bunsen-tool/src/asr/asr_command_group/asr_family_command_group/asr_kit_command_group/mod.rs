//! `asr family <family>`: one family's command group.

mod asr_kit_fetch;
mod asr_kit_list;
#[cfg(feature = "live")]
mod asr_kit_live;
mod asr_kit_transcribe;

pub use asr_kit_fetch::*;
pub use asr_kit_list::*;
#[cfg(feature = "live")]
pub use asr_kit_live::*;
pub use asr_kit_transcribe::*;
use bunsen::errors::BunsenResult;

use crate::asr::{
    Family,
    FamilyFlags,
};

/// One family's commands; `F` is its own transcription flags.
#[derive(clap::Args, Debug)]
pub struct AsrKitCommandGroup<F: clap::Args> {
    #[command(subcommand)]
    action: AsrKitSubCommand<F>,
}

/// Every family's commands: `asr family <family>`'s, and among Whisper's.
#[derive(clap::Subcommand, Debug)]
pub enum AsrKitSubCommand<F: clap::Args> {
    /// List the models `--model` can name, their licences and files.
    ///
    /// Each file is marked with whether it is cached.
    List(AsrKitListCommand),

    /// Bring every file of the named models into the weights cache.
    Fetch(AsrKitFetchCommand),

    /// Transcribe audio files.
    Transcribe(AsrKitTranscribeCommand<F>),

    /// Transcribe the microphone, as it speaks.
    #[cfg(feature = "live")]
    Live(AsrKitLiveCommand<F>),
}

impl<F: clap::Args + FamilyFlags> AsrKitCommandGroup<F> {
    /// Runs the command on `family`'s models.
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.action.run(family)
    }
}

impl<F: clap::Args + FamilyFlags> AsrKitSubCommand<F> {
    /// Runs the command on `family`'s models.
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        match self {
            AsrKitSubCommand::List(list) => list.run(family.into()),
            AsrKitSubCommand::Fetch(fetch) => fetch.run(family.into()),
            AsrKitSubCommand::Transcribe(transcribe) => transcribe.run(family),
            #[cfg(feature = "live")]
            AsrKitSubCommand::Live(live) => live.run(family),
        }
    }
}
