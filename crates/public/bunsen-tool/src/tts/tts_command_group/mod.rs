//! `bunsen-tool tts`: the command group.

mod tts_fetch;
mod tts_list;
mod tts_run;
mod tts_voices;

use bunsen::errors::BunsenResult;
use clap::ArgMatches;
pub use tts_fetch::*;
pub use tts_list::*;
pub use tts_run::*;
pub use tts_voices::*;

/// `bunsen-tool tts`'s commands: text to speech with every TTS family's
/// checkpoints (ZONOS3-Mini, Qwen3-TTS). `tts run` speaks a text in a voice
/// to a WAV file or a speaker; `tts voices` lists a checkpoint's stock
/// voices; `tts list` and `tts fetch` are the checkpoints'.
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
        }
    }
}
