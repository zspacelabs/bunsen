//! `tts build-voice`.

use std::path::PathBuf;

use bunsen::{
    errors::BunsenResult,
    kits::speech::tts::driver::{
        TtsDriverConfig,
        TtsVoiceReference,
    },
};
use bunsen_app::logging::{
    LogArgs,
    LogPolicy,
};
use clap::ArgMatches;
use log::LevelFilter;

use crate::tts::{
    TtsModelArgs,
    TtsVoiceOutArgs,
};

/// Builds one checkpoint's voice file from a voice reference.
#[derive(clap::Args, Debug)]
pub struct TtsBuildVoiceCommand {
    #[clap(flatten)]
    logging: LogArgs,

    /// The voice reference (`tts collect-voice` makes one).
    pub reference_dir: PathBuf,

    /// The checkpoint the voice is for.
    #[clap(flatten)]
    pub model: TtsModelArgs,

    /// The clone mode.
    ///
    /// The model's first the reference satisfies when omitted (ZONOS3's
    /// `embedding`).
    #[arg(long)]
    pub mode: Option<String>,

    /// Where the voice file goes.
    #[clap(flatten)]
    pub out: TtsVoiceOutArgs,
}

impl TtsBuildVoiceCommand {
    /// Builds the voice from the reference and writes it; `matches` are the
    /// command's own, for the stage flags' order.
    ///
    /// # Errors
    /// As [`TtsVoiceReference::read_dir`], the model's load and `build_voice`,
    /// and the voice's write.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let reference = TtsVoiceReference::read_dir(&self.reference_dir)?;
        build_voice_file(
            &reference,
            &self.reference_dir,
            &self.model,
            self.mode.as_deref(),
            &self.out,
            matches,
        )
    }
}

/// The voice built from `reference` for the checkpoint `model` chooses, in
/// `mode`, written where `out` says.
///
/// # Errors
/// The model's load, `build_voice`, and the voice's write.
pub fn build_voice_file(
    reference: &TtsVoiceReference,
    reference_dir: &std::path::Path,
    model: &TtsModelArgs,
    mode: Option<&str>,
    out: &TtsVoiceOutArgs,
    matches: &ArgMatches,
) -> BunsenResult<()> {
    let loaded = model.load(matches, |_| Ok(TtsDriverConfig::new()))?;
    let voice = loaded.synthesizer.build_voice(reference, mode)?;
    let path = out.write(&voice, reference, reference_dir)?;
    eprintln!(
        "{}: a {} voice for {} (mode {}), from {}",
        path.display(),
        voice.identity.family,
        loaded.model,
        voice.identity.mode,
        reference_dir.display()
    );
    Ok(())
}
