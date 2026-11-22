//! `bunsen-tool tts`: the command group.

mod tts_build_voice;
mod tts_clone;
mod tts_collect_voice;
mod tts_fetch;
mod tts_list;
mod tts_run;
mod tts_voices;

use bunsen::errors::BunsenResult;
use clap::ArgMatches;
pub use tts_build_voice::*;
pub use tts_clone::*;
pub use tts_collect_voice::*;
pub use tts_fetch::*;
pub use tts_list::*;
pub use tts_run::*;
pub use tts_voices::*;

/// `bunsen-tool tts`'s commands: text to speech with every TTS family's
/// checkpoints (ZONOS3-Mini, Qwen3-TTS). `tts run` speaks a text in a voice
/// to a WAV file or a speaker; `tts voices` lists a checkpoint's stock
/// voices; `tts list` and `tts fetch` are the checkpoints'. Cloning is in two
/// steps: `tts collect-voice` makes a model-free reference from clips, `tts
/// build-voice` makes a checkpoint's voice file from a reference; `tts clone`
/// does both.
#[derive(clap::Args, Debug)]
pub struct TtsCommandGroup {
    #[clap(subcommand)]
    command: TtsSubCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum TtsSubCommand {
    /// Speaks a text in a voice, to a WAV file or a speaker.
    ///
    /// The text, from the arguments or stdin, normalized as the model's
    /// release normalizes it, split into takes at its sentences, each decoded
    /// whole: written as mono 16-bit audio at the model's rate, played, or
    /// both.
    Run(Box<TtsRunCommand>),

    /// The checkpoint's stock voices: language, style, gender.
    ///
    /// Reads only the voices' files or the config (asking first when they
    /// must be fetched, unless `--yes`), not the model.
    Voices(Box<TtsVoicesCommand>),

    /// The checkpoints bunsen knows by name, by family.
    ///
    /// Each one's origin, and each of its files' path in its repo and whether
    /// it is cached.
    List(TtsListCommand),

    /// Brings every file of the named checkpoints into the weights cache.
    Fetch(TtsFetchCommand),

    /// Collects reference clips into a voice reference, for any model to clone.
    ///
    /// The clips are copied into the reference directory as they are, at their
    /// own rate and channels, with `reference.json`: each clip's hash and
    /// geometry, and what is given of the voice (its name, transcript,
    /// language). Loads no model.
    CollectVoice(Box<TtsCollectVoiceCommand>),

    /// Builds one checkpoint's voice file from a voice reference.
    ///
    /// The checkpoint's encoder reads the reference's clips (resampled and
    /// mixed as it was trained) into a voice of its own tensors, which `tts run
    /// --voice-file` (or `--voice NAME --voice-dir DIR`) speaks in.
    BuildVoice(Box<TtsBuildVoiceCommand>),

    /// Clones a voice from reference clips: `collect-voice`, then
    /// `build-voice`.
    ///
    /// The reference is kept, so the voice can be built again for another
    /// checkpoint.
    Clone(Box<TtsCloneCommand>),
}

impl TtsCommandGroup {
    /// Runs the command; `matches` are the leaf command's own.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        match &self.command {
            TtsSubCommand::Run(cmd) => cmd.run(matches),
            TtsSubCommand::Voices(cmd) => cmd.run(matches),
            TtsSubCommand::List(cmd) => cmd.run(),
            TtsSubCommand::Fetch(cmd) => cmd.run(),
            TtsSubCommand::CollectVoice(cmd) => cmd.run(),
            TtsSubCommand::BuildVoice(cmd) => cmd.run(matches),
            TtsSubCommand::Clone(cmd) => cmd.run(matches),
        }
    }
}
