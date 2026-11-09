//! `asr family`: the command group.

mod asr_kit_command_group;
mod asr_whisper_command_group;

pub use asr_kit_command_group::*;
pub use asr_whisper_command_group::*;
use bunsen::errors::BunsenResult;

use crate::asr::{
    Family,
    parakeet::ParakeetFlags,
    qwen3_asr::Qwen3AsrFlags,
    voxtral::VoxtralFlags,
};

/// The model families' commands.
#[derive(clap::Args, Debug)]
pub struct AsrFamilyCommandGroup {
    #[command(subcommand)]
    family: AsrFamilySubCommand,
}

/// The model families, each with its own `list` and `transcribe`.
#[derive(clap::Subcommand, Debug)]
enum AsrFamilySubCommand {
    /// NVIDIA's Parakeet TDT: `FermionResearch`'s Phonon-2 by default.
    Parakeet(AsrKitCommandGroup<ParakeetFlags>),

    /// Qwen's Qwen3-ASR: `FermionResearch`'s Phonon-1 by default.
    #[command(name = "qwen3-asr")]
    Qwen3Asr(AsrKitCommandGroup<Qwen3AsrFlags>),

    /// Mistral's Voxtral Realtime: `mistralai/voxtral-mini-4b-realtime-2602`
    /// by default.
    Voxtral(AsrKitCommandGroup<VoxtralFlags>),

    /// `OpenAI`'s Whisper: `openai/base` by default.
    Whisper(AsrWhisperCommandGroup),
}

impl AsrFamilyCommandGroup {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.family {
            AsrFamilySubCommand::Parakeet(cmd) => cmd.run(Family::Parakeet),
            AsrFamilySubCommand::Qwen3Asr(cmd) => cmd.run(Family::Qwen3Asr),
            AsrFamilySubCommand::Voxtral(cmd) => cmd.run(Family::Voxtral),
            AsrFamilySubCommand::Whisper(cmd) => cmd.run(),
        }
    }
}
