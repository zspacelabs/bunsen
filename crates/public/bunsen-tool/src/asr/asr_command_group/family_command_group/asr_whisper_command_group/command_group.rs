//! The `asr_whisper` command group.

use bunsen::errors::BunsenResult;

use super::{
    asr_whisper_inspect::AsrWhisperInspectCommand,
    asr_whisper_prefabs::AsrWhisperPrefabsCommand,
};
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
