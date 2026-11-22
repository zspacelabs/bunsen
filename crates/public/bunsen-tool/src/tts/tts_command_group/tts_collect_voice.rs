//! `tts collect-voice`.

#[cfg(feature = "live")]
use std::path::Path;
use std::path::PathBuf;

use bunsen::{
    errors::BunsenResult,
    kits::speech::tts::driver::TtsVoiceReference,
};
#[cfg(feature = "live")]
use bunsen::{
    errors::{
        BunsenError,
        sys_at,
    },
    support::audio::save_wav_mono,
};
use bunsen_app::logging::{
    LogArgs,
    LogPolicy,
};
use log::LevelFilter;

#[cfg(feature = "live")]
use crate::support::CaptureArgs;
use crate::tts::TtsAnnotationArgs;

/// A recorded clip's file name in a voice reference.
#[cfg(feature = "live")]
pub const TTS_RECORDING_FILE: &str = "recording.wav";

/// Collects clips into a voice reference: the model-free half of cloning.
#[derive(clap::Args, Debug)]
pub struct TtsCollectVoiceCommand {
    #[clap(flatten)]
    logging: LogArgs,

    /// The reference directory to make: new, or empty.
    pub reference_dir: PathBuf,

    /// The reference clips, copied in as they are.
    ///
    /// Any rate and channels; WAV, mp3, ... None are needed with `--record`.
    #[cfg_attr(feature = "live", arg(required_unless_present = "record"))]
    #[cfg_attr(not(feature = "live"), arg(required = true))]
    pub clips: Vec<PathBuf>,

    /// What is given of the voice.
    #[clap(flatten)]
    pub annotations: TtsAnnotationArgs,

    /// A clip recorded from the microphone.
    #[cfg(feature = "live")]
    #[clap(flatten)]
    pub record: TtsRecordArgs,
}

/// A clip recorded from the microphone, into a voice reference.
#[cfg(feature = "live")]
#[derive(clap::Args, Debug, Default)]
pub struct TtsRecordArgs {
    /// Record a clip from the microphone, after the clips given.
    ///
    /// It is captured at the input device's own rate, mixed to mono, until
    /// Ctrl-C or `--seconds`, and kept as `recording.wav`.
    #[arg(long)]
    pub record: bool,

    /// The input device to record from, by a substring of its id or name.
    ///
    /// The host's default input device when omitted; `asr live
    /// --list-devices` lists them.
    #[arg(long, requires = "record")]
    pub input_device: Option<String>,

    /// Stop recording after this many seconds; on Ctrl-C when omitted.
    #[arg(long, requires = "record")]
    pub seconds: Option<f64>,
}

#[cfg(feature = "live")]
impl TtsRecordArgs {
    /// Records a clip into `dir` as [`TTS_RECORDING_FILE`], at the input
    /// device's own rate; its path.
    ///
    /// # Errors
    /// As opening and running the capture; [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal)
    /// when nothing was captured; the WAV's write.
    pub fn record(
        &self,
        dir: &Path,
    ) -> BunsenResult<PathBuf> {
        let capture = CaptureArgs {
            input_device: self.input_device.clone(),
            list_devices: false,
            seconds: self.seconds,
        }
        .open()?
        .expect("a capture, not a listing");
        let rate = capture.native_rate()?;
        let mut samples = Vec::new();
        let seconds = capture.run(rate, rate / 10, |chunk, _| {
            samples.extend_from_slice(chunk);
            Ok(())
        })?;
        if samples.is_empty() {
            return Err(BunsenError::illegal("nothing was recorded"));
        }
        let path = dir.join(TTS_RECORDING_FILE);
        save_wav_mono(&path, &samples, rate as u32)?;
        eprintln!("# recorded {seconds:.1} s at {rate} Hz");
        Ok(path)
    }
}

impl TtsCollectVoiceCommand {
    /// Collects the reference and says what it holds.
    ///
    /// # Errors
    /// As [`TtsVoiceReference::collect`]; with `--record`, as
    /// [`TtsRecordArgs::record`], and before recording, as `collect` refuses
    /// a directory with anything in it or a clip named `recording.wav`.
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        #[allow(unused_mut)]
        let mut clips = self.clips.clone();
        #[cfg(feature = "live")]
        let _recording = if self.record.record {
            self.check_before_recording()?;
            let scratch = tempfile::tempdir().map_err(|e| {
                BunsenError::sys("could not make a directory for the recording").with_cause(e)
            })?;
            clips.push(self.record.record(scratch.path())?);
            Some(scratch)
        } else {
            None
        };
        let reference = TtsVoiceReference::collect(
            &self.reference_dir,
            &clips,
            &self.annotations.annotations(),
        )?;
        eprintln!("{}", describe_reference(&reference, &self.reference_dir));
        Ok(())
    }
}

#[cfg(feature = "live")]
impl TtsCollectVoiceCommand {
    /// What [`TtsVoiceReference::collect`] would refuse after the recording,
    /// refused before it: a directory with anything in it, a clip of the
    /// recording's name.
    fn check_before_recording(&self) -> BunsenResult<()> {
        let dir = &self.reference_dir;
        if dir.exists() {
            let mut entries = std::fs::read_dir(dir).map_err(sys_at("read", dir))?;
            if entries.next().is_some() {
                return Err(BunsenError::illegal(format!(
                    "{}: a voice reference is collected into a new or empty directory",
                    dir.display()
                )));
            }
        }
        if let Some(clip) = self
            .clips
            .iter()
            .find(|c| c.file_name().is_some_and(|f| f == TTS_RECORDING_FILE))
        {
            return Err(BunsenError::illegal(format!(
                "{}: a clip named as the recording is; rename it",
                clip.display()
            )));
        }
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        collect: TtsCollectVoiceCommand,
    }

    fn parse(args: &[&str]) -> Result<TtsCollectVoiceCommand, clap::Error> {
        Cli::try_parse_from([&["collect-voice"], args].concat()).map(|cli| cli.collect)
    }

    /// The clips are needed unless a clip is recorded; the device and the
    /// length go only with `--record`.
    #[test]
    fn test_args() {
        let cli = parse(&["ada.voice-ref", "a.wav", "--name", "Ada"]).unwrap();
        assert_eq!(cli.clips, [PathBuf::from("a.wav")]);
        assert!(parse(&["ada.voice-ref"]).is_err(), "no clips");

        #[cfg(feature = "live")]
        {
            let cli = parse(&["ada.voice-ref", "--record", "--seconds", "20"]).unwrap();
            assert!(cli.record.record);
            assert!(cli.clips.is_empty());
            assert_eq!(cli.record.seconds, Some(20.0));
            let cli = parse(&[
                "ada.voice-ref",
                "a.wav",
                "--record",
                "--input-device",
                "usb",
            ]);
            assert_eq!(cli.unwrap().record.input_device.as_deref(), Some("usb"));
            assert!(parse(&["ada.voice-ref", "a.wav", "--seconds", "20"]).is_err());
            assert!(parse(&["ada.voice-ref", "a.wav", "--input-device", "usb"]).is_err());
        }
    }

    /// Before recording: a directory with anything in it is refused, and so
    /// is a clip of the recording's name.
    #[cfg(feature = "live")]
    #[test]
    fn test_check_before_recording() {
        let work = tempfile::tempdir().unwrap();
        let dir = work.path().join("ada.voice-ref");
        let cli = parse(&[dir.to_str().unwrap(), "--record"]).unwrap();
        cli.check_before_recording().unwrap();
        std::fs::create_dir(&dir).unwrap();
        cli.check_before_recording().unwrap();
        std::fs::write(dir.join("x"), b"").unwrap();
        assert!(cli.check_before_recording().is_err(), "not empty");
        let other = work.path().join("bea.voice-ref");
        let cli = parse(&[other.to_str().unwrap(), "take/recording.wav", "--record"]).unwrap();
        assert!(
            cli.check_before_recording().is_err(),
            "the recording's name"
        );
    }
}
