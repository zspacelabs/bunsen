//! `bunsen-asr <family> transcribe`.
use std::{
    path::PathBuf,
    time::{
        Duration,
        Instant,
    },
};

use bunsen::{
    errors::BunsenResult,
    support::audio::load_audio_mono_sr,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LogArgs,
        LogLevelNum,
    },
};
use serde::Serialize;

use crate::{
    args::DeviceArgs,
    family::{
        Family,
        Loaded,
    },
    parakeet::{
        self,
        ParakeetFlags,
    },
    qwen3_asr::{
        self,
        Qwen3AsrFlags,
    },
    stream::Preset,
    whisper,
};

/// A family's own transcription flags.
pub trait FamilyFlags: std::fmt::Debug {
    /// Parakeet's, when these are Parakeet's.
    fn parakeet(&self) -> Option<&ParakeetFlags> {
        None
    }

    /// Qwen3-ASR's, when these are Qwen3-ASR's.
    fn qwen3_asr(&self) -> Option<&Qwen3AsrFlags> {
        None
    }

    /// Whisper's, when these are Whisper's.
    fn whisper(&self) -> Option<&whisper::WhisperFlags> {
        None
    }
}

impl FamilyFlags for whisper::WhisperFlags {
    fn whisper(&self) -> Option<&whisper::WhisperFlags> {
        Some(self)
    }
}

impl FamilyFlags for ParakeetFlags {
    fn parakeet(&self) -> Option<&ParakeetFlags> {
        Some(self)
    }
}

impl FamilyFlags for Qwen3AsrFlags {
    fn qwen3_asr(&self) -> Option<&Qwen3AsrFlags> {
        Some(self)
    }
}

/// Transcribes audio files.
///
/// Each file is decoded at the model's rate (16 kHz, mono) and streamed
/// through the family's stream driver, about a second a push. `--model`
/// names a row of `list`, or `hf:org/repo` for a Hugging Face repo in
/// `transformers`' layout; the model's info card, and after each transcript
/// how long it took, are logged at `INFO` (`-v`).
#[derive(clap::Args, Debug)]
pub struct TranscribeCmd<F: clap::Args> {
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The model, by name: `list` shows them; the family's default when
    /// omitted.
    #[arg(long)]
    pub model: Option<String>,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: DeviceArgs,

    /// When to decode.
    #[arg(long, value_enum, default_value_t = Preset::Offline)]
    pub preset: Preset,

    /// Print one JSON object per file: the text, the language when the model
    /// names one, each word with its start and end in seconds when the model
    /// times words, and each segment when it emits segments.
    #[arg(long)]
    pub json: bool,

    #[clap(flatten)]
    pub flags: F,

    /// The audio files.
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

/// A transcript, whichever family made it.
#[derive(Clone, Debug, Serialize)]
pub struct Transcript {
    /// The text.
    pub text: String,

    /// The language the model named, if it names one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Each word with its start and end in seconds, if the model times them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,

    /// Each segment with its start and end in seconds, if the model emits
    /// segments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
}

/// A timed word.
#[derive(Clone, Debug, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// A timed segment.
#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// One file, as `--json` prints it.
#[derive(Serialize)]
struct JsonTranscript<'a> {
    file: String,
    #[serde(flatten)]
    transcript: &'a Transcript,
}

/// A family's stream driver; Whisper's, much the largest, boxed.
enum Driver {
    Parakeet(bunsen::kits::speech::parakeet::ParakeetStreamDriver),
    Qwen3Asr(bunsen::kits::speech::qwen3_asr::Qwen3AsrStreamDriver),
    Whisper(Box<bunsen::kits::speech::whisper::driver::WhisperStreamDriver>),
}

impl Driver {
    fn sample_rate(&self) -> usize {
        match self {
            Driver::Parakeet(d) => d.sample_rate(),
            Driver::Qwen3Asr(d) => d.sample_rate(),
            Driver::Whisper(d) => d.sample_rate(),
        }
    }

    fn transcribe(
        &self,
        samples: &[f32],
    ) -> BunsenResult<Transcript> {
        match self {
            Driver::Parakeet(d) => parakeet::transcribe(d, samples),
            Driver::Qwen3Asr(d) => qwen3_asr::transcribe(d, samples),
            Driver::Whisper(d) => whisper::transcribe(d, samples),
        }
    }
}

impl<F: clap::Args + FamilyFlags> TranscribeCmd<F> {
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Warn))?;
        let device = self.device.device()?;
        let cache = self.cache.init()?;
        let model = self.model.as_deref().unwrap_or(family.default_model());
        let loaded = family.load(model, &cache, &device)?;

        let flags = &self.flags;
        let preset = self.preset;
        let driver = match loaded {
            Loaded::Parakeet(bundle) => Driver::Parakeet(parakeet::driver(
                bundle,
                flags.parakeet().expect("Parakeet's flags"),
                preset,
                &cache,
                &device,
            )?),
            Loaded::Qwen3Asr(bundle) => Driver::Qwen3Asr(qwen3_asr::driver(
                bundle,
                flags.qwen3_asr().expect("Qwen3-ASR's flags"),
                preset,
                &cache,
                &device,
            )?),
            Loaded::Whisper(bundle) => Driver::Whisper(Box::new(whisper::driver(
                bundle,
                flags.whisper().expect("Whisper's flags"),
                preset,
                &cache,
                &device,
            )?)),
        };
        let rate = driver.sample_rate();

        let (mut audio, mut busy) = (0.0, Duration::ZERO);
        for path in &self.files {
            let samples = load_audio_mono_sr(path, rate)?;
            let t0 = Instant::now();
            let transcript = driver.transcribe(&samples)?;
            let elapsed = t0.elapsed();

            if self.json {
                let json = JsonTranscript {
                    file: path.display().to_string(),
                    transcript: &transcript,
                };
                println!(
                    "{}",
                    serde_json::to_string(&json).expect("a transcript serializes")
                );
            } else {
                println!("{}", transcript.text);
            }

            let seconds = samples.len() as f64 / rate as f64;
            log::info!(
                "{}: {seconds:.1} s of audio in {elapsed:.2?} ({:.1}x realtime)",
                path.display(),
                seconds / elapsed.as_secs_f64()
            );
            audio += seconds;
            busy += elapsed;
        }
        if self.files.len() > 1 {
            log::info!(
                "{} files: {audio:.1} s of audio in {busy:.2?} ({:.1}x realtime)",
                self.files.len(),
                audio / busy.as_secs_f64()
            );
        }
        Ok(())
    }
}

/// Seconds, rounded to the millisecond.
pub fn millis(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}
