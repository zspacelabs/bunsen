//! `asr live`.

use bunsen::errors::BunsenResult;

use super::asr_construct_args::with_flags;
use crate::asr::{
    AsrConstructArgs,
    AsrLiveIo,
    Mode,
};

/// Transcribes the microphone, as it speaks, with a preset's model for
/// the microphone.
///
/// The preset picks the family, the model and when to decode; `--model`,
/// `--emission` and `--vad` change them, `presets show` prints them, and `-v`
/// logs them as the `bunsen-tool asr family` command they are.
#[derive(clap::Args, Debug)]
pub struct AsrLiveCommand {
    #[clap(flatten)]
    pub preset: AsrConstructArgs,

    #[clap(flatten)]
    pub io: AsrLiveIo,
}

impl AsrLiveCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.io.init_logging()?;
        let resolved = self.preset.resolve(Mode::Live)?;
        with_flags!(resolved, flags => self.io.run(
            resolved.family,
            &resolved.model,
            &flags,
            &resolved.streaming(),
        ))
    }
}
