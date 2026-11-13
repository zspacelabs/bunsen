//! `bunsen-tool asr`: the command group.

mod asr_construct_args;
#[cfg(feature = "live")]
mod asr_live;
mod asr_transcribe;
mod diarize_command_group;
mod family_command_group;
mod presets_command_group;

pub use asr_construct_args::*;
#[cfg(feature = "live")]
pub use asr_live::*;
pub use asr_transcribe::*;
use bunsen::errors::BunsenResult;
pub use diarize_command_group::*;
pub use family_command_group::*;
pub use presets_command_group::*;

/// `bunsen-tool asr`'s commands: `asr <transcribe|live> [--preset NAME]
/// [FLAGS]` runs a preset, a model chosen by what it is for, and `asr presets
/// <list|show|fetch>` shows them; `asr family
/// <parakeet|qwen3-asr|voxtral|whisper> <list|fetch|transcribe|live> [FLAGS]`
/// (and `family whisper inspect|prefabs`) runs a family's model by name; `asr
/// diarize <list|fetch|run|live>` says who spoke when.
#[derive(clap::Args, Debug)]
pub struct AsrCommandGroup {
    #[clap(subcommand)]
    command: AsrSubCommand,
}

#[derive(clap::Subcommand, Debug)]
enum AsrSubCommand {
    /// Transcribe audio files with a preset: a model chosen by what it is for.
    ///
    /// The preset is `en` by default.
    Transcribe(Box<AsrTranscribeCommand>),

    /// Transcribe the microphone, as it speaks, with a preset.
    #[cfg(feature = "live")]
    Live(Box<AsrLiveCommand>),

    /// The presets: what each is for, and what it runs.
    Presets(AsrPresetsCommandGroup),

    /// The model families, each with its own `list` and `transcribe`.
    ///
    /// Each takes only the flags that apply to it.
    Family(AsrFamilyCommandGroup),

    /// NVIDIA's Sortformer, who spoke when: `nvidia/nemotron-3-diarization`
    /// by default.
    Diarize(AsrDiarizeCommandGroup),
}

impl AsrCommandGroup {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.command {
            AsrSubCommand::Transcribe(cmd) => cmd.run(),
            #[cfg(feature = "live")]
            AsrSubCommand::Live(cmd) => cmd.run(),
            AsrSubCommand::Presets(cmd) => cmd.run(),
            AsrSubCommand::Family(cmd) => cmd.run(),
            AsrSubCommand::Diarize(cmd) => cmd.run(),
        }
    }
}

#[cfg(test)]
mod tests;
