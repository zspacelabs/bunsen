//! `tts collect-voice`.

use std::path::PathBuf;

use bunsen::{
    errors::BunsenResult,
    kits::speech::tts::driver::TtsVoiceReference,
};
use bunsen_app::logging::{
    LogArgs,
    LogPolicy,
};
use log::LevelFilter;

use crate::tts::TtsAnnotationArgs;

/// Collects clips into a voice reference: the model-free half of cloning.
#[derive(clap::Args, Debug)]
pub struct TtsCollectVoiceCommand {
    #[clap(flatten)]
    logging: LogArgs,

    /// The reference directory to make: new, or empty.
    pub reference_dir: PathBuf,

    /// The reference clips, copied in as they are.
    ///
    /// Any rate and channels; WAV, mp3, ...
    #[arg(required = true)]
    pub clips: Vec<PathBuf>,

    /// What is given of the voice.
    #[clap(flatten)]
    pub annotations: TtsAnnotationArgs,
}

impl TtsCollectVoiceCommand {
    /// Collects the reference and says what it holds.
    ///
    /// # Errors
    /// As [`TtsVoiceReference::collect`].
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let reference = TtsVoiceReference::collect(
            &self.reference_dir,
            &self.clips,
            &self.annotations.annotations(),
        )?;
        eprintln!("{}", describe_reference(&reference, &self.reference_dir));
        Ok(())
    }
}

/// One line on a reference: where, its clips and length, its digest.
pub fn describe_reference(
    reference: &TtsVoiceReference,
    dir: &std::path::Path,
) -> String {
    format!(
        "{}: a voice reference of {} clips, {:.2} s{}; digest {}",
        dir.display(),
        reference.clips.len(),
        reference.seconds(),
        if reference.transcript().is_some() {
            ", with a transcript"
        } else {
            ""
        },
        &reference.digest[..12.min(reference.digest.len())],
    )
}
