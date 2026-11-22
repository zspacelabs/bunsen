//! `tts clone`.

use std::path::PathBuf;

use bunsen::{
    errors::BunsenResult,
    kits::speech::tts::driver::TtsVoiceReference,
};
use bunsen_app::logging::{
    LogArgs,
    LogPolicy,
};
use clap::ArgMatches;
use log::LevelFilter;

use super::{
    build_voice_file,
    describe_reference,
};
use crate::tts::{
    TtsAnnotationArgs,
    TtsModelArgs,
    TtsVoiceOutArgs,
};

/// Clones a voice from clips: `tts collect-voice`, then `tts build-voice`.
#[derive(clap::Args, Debug)]
pub struct TtsCloneCommand {
    #[clap(flatten)]
    logging: LogArgs,

    /// The reference directory to make, new or empty.
    ///
    /// The voice can be built again from it, for another checkpoint.
    pub reference_dir: PathBuf,

    /// The reference clips, any rate and channels: copied in as they are.
    #[arg(required = true)]
    pub clips: Vec<PathBuf>,

    /// What is given of the voice.
    #[clap(flatten)]
    pub annotations: TtsAnnotationArgs,

    /// The checkpoint the voice is for.
    #[clap(flatten)]
    pub model: TtsModelArgs,

    /// The clone mode; the model's first the reference satisfies when omitted.
    #[arg(long)]
    pub mode: Option<String>,

    /// Where the voice file goes.
    #[clap(flatten)]
    pub out: TtsVoiceOutArgs,
}

impl TtsCloneCommand {
    /// Collects the reference, then builds the voice from it; `matches` are the
    /// command's own.
    ///
    /// # Errors
    /// As [`TtsVoiceReference::collect`] and [`build_voice_file`].
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let reference = TtsVoiceReference::collect(
            &self.reference_dir,
            &self.clips,
            &self.annotations.annotations(),
        )?;
        eprintln!("{}", describe_reference(&reference, &self.reference_dir));
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        clone: TtsCloneCommand,
    }

    /// The reference directory and clips are positional; the voice goes to
    /// exactly one of `-o` and `--voice-dir`; the annotations and the mode
    /// carry through.
    #[test]
    fn test_args() {
        let cli = Cli::try_parse_from([
            "clone",
            "ada.voice-ref",
            "a.wav",
            "b.mp3",
            "--voice-dir",
            "voices",
            "--name",
            "Ada",
            "--transcript",
            "hello",
            "--mode",
            "embedding",
        ])
        .unwrap()
        .clone;
        assert_eq!(cli.reference_dir, PathBuf::from("ada.voice-ref"));
        assert_eq!(cli.clips, [PathBuf::from("a.wav"), PathBuf::from("b.mp3")]);
        assert_eq!(cli.out.voice_dir, Some(PathBuf::from("voices")));
        assert_eq!(cli.mode.as_deref(), Some("embedding"));
        let annotations = cli.annotations.annotations();
        assert_eq!(annotations.name.as_deref(), Some("Ada"));
        assert_eq!(annotations.transcript.as_deref(), Some("hello"));

        for bad in [
            vec!["clone", "src", "a.wav"],
            vec!["clone", "src", "-o", "v.voice.safetensors"],
            vec!["clone", "src", "a.wav", "-o", "v", "--voice-dir", "d"],
        ] {
            assert!(Cli::try_parse_from(bad.clone()).is_err(), "{bad:?}");
        }
    }
}
