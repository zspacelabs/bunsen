//! `asr family <family> transcribe`, and what `asr transcribe` shares.

use std::{
    path::PathBuf,
    time::{
        Duration,
        Instant,
    },
};

use bunsen::{
    data::resources::ResourceCache,
    errors::{
        BunsenResult,
        ResultContext,
    },
    support::audio::load_audio_mono_sr,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::BackendArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::{
    asr::{
        AsrSpeakerArgs,
        Driver,
        Emission,
        Family,
        FamilyFlags,
        Stamp,
        Streaming,
        VadArgs,
        publish::{
            Clock,
            Event,
            PublishArgs,
            statements,
        },
    },
    support::PresetArgs,
};

/// Transcribes audio files.
///
/// Each file is decoded at the model's rate (16 kHz, mono) and streamed
/// through the family's stream driver, about a second a push. `--model`
/// names a row of `list`, or `hf:org/repo` for a Hugging Face repo in
/// `transformers`' layout; the model's info card, and after each transcript
/// how long it took, are logged at `INFO` (`-v`).
#[derive(clap::Args, Debug)]
pub struct AsrKitTranscribeCommand<F: clap::Args> {
    /// The model, as a lookup expression; the family's default when omitted.
    ///
    /// The expression also names its presets and patches its policy
    /// (`'openai/small[translate]{beam_size: 1}'`): `list` shows the names
    /// and presets. Whisper also takes a checkpoint's path.
    #[arg(long, value_name = "EXPR")]
    pub model: Option<String>,

    #[clap(flatten)]
    pub presets: PresetArgs,

    /// When to decode.
    #[arg(long, value_enum, default_value_t = Emission::Offline)]
    pub emission: Emission,

    /// Milliseconds of audio per push.
    ///
    /// Each file is fed to the driver as a live loop would feed it, in pieces
    /// of this.
    #[arg(long, default_value_t = 1000)]
    pub chunk_ms: usize,

    #[clap(flatten)]
    pub vad: VadArgs,

    #[clap(flatten)]
    pub flags: F,

    #[clap(flatten)]
    pub io: AsrTranscribeIo,
}

/// What `transcribe` reads and prints, whichever model it runs: the
/// options a family's command and a preset's share.
#[derive(clap::Args, Debug)]
pub struct AsrTranscribeIo {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: BackendArgs,

    #[clap(flatten)]
    pub publish: PublishArgs,

    #[clap(flatten)]
    pub speakers: AsrSpeakerArgs,

    /// The audio files.
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

impl<F: clap::Args + FamilyFlags> AsrKitTranscribeCommand<F> {
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.io.init_logging()?;
        let model = self
            .presets
            .apply(self.model.as_deref().unwrap_or(family.default_model()))?;
        let model = model.as_str();
        let streaming = Streaming {
            emission: self.emission,
            vad: self.vad.slot(self.emission)?,
            chunk_ms: self.chunk_ms,
        };
        self.io.run(family, model, &self.flags, &streaming, None)
    }
}

impl AsrTranscribeIo {
    /// Logs as `-v`/`-q` say, warnings by default.
    pub fn init_logging(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))
    }

    /// Transcribes the files with `model` of `family` under `flags`, streamed
    /// as `streaming` says, publishing each file's statements, then its
    /// transcript with `stamp`, what chose them.
    pub fn run<F: FamilyFlags>(
        &self,
        family: Family,
        model: &str,
        flags: &F,
        streaming: &Streaming,
        stamp: Option<&Stamp>,
    ) -> BunsenResult<()> {
        // The targets open first: a bad config, or a broker, fails before
        // the model loads.
        let mut publisher = self
            .publish
            .init(Clock::Offset, self.speakers.diarize.is_some())?;
        let device = self.device.open()?;
        // The options come from the command line.
        let cache = ResourceCache::new(flags.cache_options(self.cache.options())).as_policy()?;
        let loaded = family.load(model, flags, &cache, &device)?;
        let speakers = self.speakers.diarizer(&cache, &device)?;

        let driver = Driver::build(
            loaded,
            streaming.emission,
            streaming.vad.as_deref(),
            speakers,
            &cache,
            &device,
        )?;
        let rate = driver.sample_rate();
        let chunk = driver.push_samples(streaming.chunk_ms);

        let (mut audio, mut busy) = (0.0, Duration::ZERO);
        for path in &self.files {
            let samples = load_audio_mono_sr(path, rate)?;
            let t0 = Instant::now();
            let (transcript, segments) = driver.transcribe(&samples, chunk)?;
            let elapsed = t0.elapsed();

            let file = path.display().to_string();
            for statement in statements(&segments) {
                publisher.push(&Event::Statement {
                    file: Some(&file),
                    statement: &statement,
                })?;
            }
            publisher.push(&Event::Transcript {
                file: &file,
                stamp,
                transcript: &transcript,
            })?;

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
        publisher.finish()
    }
}
