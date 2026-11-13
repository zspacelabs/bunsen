//! `asr family whisper`: Whisper's command group.

mod asr_whisper_inspect;
mod asr_whisper_prefabs;

pub use asr_whisper_inspect::*;
pub use asr_whisper_prefabs::*;
use bunsen::errors::BunsenResult;

use crate::asr::{
    AsrFamily,
    AsrKitSubCommand,
    whisper::WhisperFlags,
};

/// Whisper's commands: every family's, and its own.
#[derive(clap::Args, Debug)]
pub struct AsrWhisperCommandGroup {
    #[command(subcommand)]
    action: AsrWhisperSubCommand,
}

#[derive(clap::Subcommand, Debug)]
enum AsrWhisperSubCommand {
    #[command(flatten)]
    Kit(Box<AsrKitSubCommand<WhisperFlags>>),

    /// Scan a checkpoint and check it against its prefab.
    Inspect(AsrWhisperInspectCommand),

    /// List the prefabs: the geometries, without weights.
    Prefabs(AsrWhisperPrefabsCommand),
}

impl AsrWhisperCommandGroup {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.action {
            AsrWhisperSubCommand::Kit(action) => action.run(AsrFamily::Whisper),
            AsrWhisperSubCommand::Inspect(inspect) => inspect.run(),
            AsrWhisperSubCommand::Prefabs(prefabs) => prefabs.run(),
        }
    }
}
