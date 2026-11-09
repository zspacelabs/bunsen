//! `asr transcribe`.

use bunsen::errors::BunsenResult;

use super::asr_construct_args::with_flags;
use crate::asr::{
    AsrConstructArgs,
    AsrTranscribeIo,
    Mode,
};

/// Transcribes audio files with a preset's model for files.
///
/// The preset picks the family, the model and when to decode; `--model`,
/// `--emission` and `--vad` change them, `presets show` prints them, and `-v`
/// logs them as the `bunsen-tool asr family` command they are. `--json` records
/// the preset and its recipe with each transcript.
#[derive(clap::Args, Debug)]
pub struct AsrTranscribeCommand {
    #[clap(flatten)]
    pub preset: AsrConstructArgs,

    #[clap(flatten)]
    pub io: AsrTranscribeIo,
}

impl AsrTranscribeCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.io.init_logging()?;
        let resolved = self.preset.resolve(Mode::Offline)?;
        let stamp = resolved.stamp();
        with_flags!(resolved, flags => self.io.run(
            resolved.family,
            &resolved.model,
            &flags,
            &resolved.streaming(),
            Some(&stamp),
        ))
    }
}
