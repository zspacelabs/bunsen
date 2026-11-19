//! `tts run`.

use std::{
    io::Read,
    path::PathBuf,
    time::{
        Instant,
        SystemTime,
        UNIX_EPOCH,
    },
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        speech::tts::{
            driver::{
                TtsDriver,
                TtsDriverConfig,
                TtsRequest,
                TtsStop,
                TtsVoiceChoice,
            },
            zonos3::{
                driver::{
                    Zonos3Engine,
                    Zonos3EngineConfig,
                    zonos3_release_normalization,
                },
                pretrained::default_zonos3_factory,
            },
        },
        support::{
            text::TextNormalizationConfig,
            tokens::BunsenTokenizer,
        },
    },
    support::audio::save_wav_mono,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::BackendArgs,
    logging::{
        LogArgs,
        LogPolicy,
    },
    precision::MinPrecisionArgs,
};
use clap::ArgMatches;
use log::LevelFilter;

use crate::{
    support::{
        ChooseArgs,
        with_patch,
        with_presets,
    },
    tts::choose_tts_model,
};

/// The stock voice `tts run` speaks in when none is named.
pub const DEFAULT_TTS_VOICE: &str = "Savannah";

/// How `tts run` normalizes its text, as `--normalize` takes it.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtsNormalize {
    /// Zyphra's normalizer, the release's: numbers, money and dates spoken.
    Zyphra,

    /// The text as typed, line breaks collapsed.
    None,
}

/// Speaks a text, to a WAV file.
#[derive(clap::Args, Debug)]
pub struct TtsRunCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The device. The language model loads at the precision it ships in
    /// (bf16) where the device accelerates it, else in f32
    /// (`LoadPolicy::performance`); `--min-precision` puts a floor under that.
    #[clap(flatten)]
    device: BackendArgs,

    /// The narrowest float the checkpoint's weights are placed in.
    #[clap(flatten)]
    precision: MinPrecisionArgs,

    /// The checkpoint, chosen among ZONOS3's builds by `--model` and the
    /// stage flags; the default checkpoint when nothing is named.
    #[clap(flatten)]
    choose: ChooseArgs,

    /// The stock voice the speech is in (`tts voices` lists them).
    ///
    /// Its reference codes lead the prompt, the release's default speaker
    /// mode.
    #[arg(long, default_value = DEFAULT_TTS_VOICE)]
    voice: String,

    /// Where the speech is written: a mono 16-bit WAV file at 44.1 kHz.
    ///
    /// Needed unless the speech is played (`--play`, with `live`).
    #[arg(long, short)]
    out: Option<PathBuf>,

    /// Whether, and where, the speech is played.
    #[cfg(feature = "live")]
    #[clap(flatten)]
    playback: crate::tts::PlaybackArgs,

    /// Seeds the take, to repeat it; a seed from the clock when omitted.
    ///
    /// The seed is printed with the take either way.
    #[arg(long)]
    seed: Option<u64>,

    /// How the text is normalized before it is spoken.
    #[arg(long, value_enum, default_value_t = TtsNormalize::Zyphra)]
    normalize: TtsNormalize,

    /// The text's language, which `--normalize zyphra` reads by.
    ///
    /// A request code of the release (`en_us`, `fr_fr`) or a language tag
    /// (`en`, `fr-FR`); one the normalizer has no grammar for is an error.
    #[arg(long, default_value = "en_us")]
    language: String,

    /// The sampling temperature (the release's 0.75 when omitted).
    #[arg(long)]
    temperature: Option<f64>,

    /// Codes kept per codebook, the most probable (the release's 80).
    #[arg(long)]
    top_k: Option<usize>,

    /// The probability mass kept per codebook (the release's 0.69).
    #[arg(long)]
    top_p: Option<f64>,

    /// Speak the text as one take, however long.
    ///
    /// A long text is otherwise spoken as several takes, split at its
    /// sentences (about 20 seconds each), joined by a short pause.
    #[arg(long)]
    no_split: bool,

    /// The most frames each take generates, at 15 a second.
    ///
    /// As many as the model's context leaves when omitted.
    #[arg(long)]
    max_frames: Option<usize>,

    /// The text; stdin when omitted (words joined by spaces).
    text: Vec<String>,
}

impl TtsRunCommand {
    /// Speaks the text; `matches` are the command's own, for the stage
    /// flags' order.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        #[cfg(feature = "live")]
        if self.playback.list()? {
            return Ok(());
        }
        #[cfg(feature = "live")]
        let playback = self.playback.open()?;
        #[cfg(feature = "live")]
        let playing = playback.is_some();
        #[cfg(not(feature = "live"))]
        let playing = false;
        if self.out.is_none() && !playing {
            return Err(BunsenError::illegal(
                "nowhere for the speech to go: name a file with --out, or --play it",
            ));
        }
        let text = self.read_text()?;
        let cache = self.cache.init()?;
        let device = self.device.open()?;
        let flags = self.choose.flags(matches)?;
        let mut model = choose_tts_model(&self.choose.model, &flags)?;
        if !flags.presets.presets.is_empty() {
            let presets: Vec<&str> = flags.presets.presets.iter().map(String::as_str).collect();
            model = with_presets(&model, &presets)?;
        }
        if let Some(patch) = self.choose.patch()? {
            model = with_patch(&model, &patch)?;
        }
        let construct = default_zonos3_factory()?.load_construct(
            &model,
            &cache,
            &device,
            self.precision.load_policy(),
        )?;
        let tokenizer = BunsenTokenizer::from_file(&construct.tokenizer)?;
        let engine = Zonos3Engine::new(
            construct.clone(),
            tokenizer,
            self.engine_config(Zonos3EngineConfig::from_generation(&construct.generation)),
        )?;
        let driver = TtsDriver::new(engine, self.driver_config())?;

        let seed = self.seed.unwrap_or_else(clock_seed);
        let request = self.request(text, seed);
        let started = Instant::now();
        let speech = driver.synthesize(&request)?;
        let spoken = started.elapsed().as_secs_f64();
        let rows: usize = speech.takes.iter().map(|take| take.steps).sum();
        if speech.takes.iter().any(|take| take.stop == TtsStop::Budget) {
            log::warn!("the take ran out of frames before it ended: raise --max-frames");
        }
        if let Some(out) = &self.out {
            save_wav_mono(out, &speech.samples, speech.sample_rate)?;
        }
        let seconds = speech.samples.len() as f64 / f64::from(speech.sample_rate);
        let place = self
            .out
            .as_ref()
            .map_or_else(|| "the speech".to_string(), |out| out.display().to_string());
        eprintln!(
            "{place}: {seconds:.2} s in {}'s voice, seed {seed}; synthesized in {spoken:.2} s \
             ({} rows, {:.1} a second)",
            self.voice,
            rows,
            rows as f64 / spoken,
        );
        #[cfg(feature = "live")]
        if let Some(playback) = &playback {
            eprintln!("# playing on {}; Ctrl-C to stop", playback.name());
            playback.play(&speech.samples, speech.sample_rate)?;
        }
        Ok(())
    }

    /// The text: the arguments, or stdin without any.
    ///
    /// # Errors
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) for no text; a
    /// read's failure.
    fn read_text(&self) -> BunsenResult<String> {
        let text = if self.text.is_empty() {
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .map_err(sys_at("read", "stdin"))?;
            text
        } else {
            self.text.join(" ")
        };
        if text.trim().is_empty() {
            return Err(BunsenError::illegal("no text to speak"));
        }
        Ok(text)
    }

    /// The request: the text in the flags' voice and language, seeded.
    fn request(
        &self,
        text: String,
        seed: u64,
    ) -> TtsRequest {
        TtsRequest::new(text)
            .with_voice(TtsVoiceChoice::Stock(self.voice.clone()))
            .with_language(self.language.clone())
            .with_seed(seed)
    }

    /// The driver's config: the flags' normalization and splitting.
    fn driver_config(&self) -> TtsDriverConfig {
        TtsDriverConfig::new()
            .with_normalization(match self.normalize {
                TtsNormalize::Zyphra => zonos3_release_normalization(),
                TtsNormalize::None => TextNormalizationConfig::CollapseLineBreaks,
            })
            .with_split(!self.no_split)
    }

    /// The engine's `config` with the flags' settings over it.
    fn engine_config(
        &self,
        mut config: Zonos3EngineConfig,
    ) -> Zonos3EngineConfig {
        if let Some(temperature) = self.temperature {
            config.sampler.temperature = temperature;
        }
        if let Some(top_k) = self.top_k {
            config.sampler.top_k = top_k;
        }
        if let Some(top_p) = self.top_p {
            config.sampler.top_p = top_p;
        }
        config.max_new_frames = self.max_frames;
        config
    }
}

/// A seed from the clock: its nanoseconds, below 2^31 (as the release's
/// random seeds are).
fn clock_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    u64::from(nanos) % (1 << 31)
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        run: TtsRunCommand,
    }

    /// Without `--out` or `--play`, the speech has nowhere to go.
    #[test]
    fn test_needs_a_destination() {
        let cli = Cli::try_parse_from(["tts", "hi"]).unwrap();
        let matches = clap::ArgMatches::default();
        let error = cli.run.run(&matches).unwrap_err();
        assert_eq!(error.kind(), bunsen::errors::BunsenErrorKind::Illegal);
        assert!(error.to_string().contains("--out"), "{error}");
    }

    /// The flags lay over the release's settings; the text joins its words.
    #[test]
    fn test_config() {
        let cli = Cli::try_parse_from([
            "tts",
            "-o",
            "out.wav",
            "--normalize",
            "none",
            "--temperature",
            "0.5",
            "--top-k",
            "40",
            "--max-frames",
            "90",
            "Hello",
            "there.",
        ])
        .unwrap();
        assert_eq!(
            cli.run.driver_config().normalization,
            TextNormalizationConfig::CollapseLineBreaks
        );
        assert!(cli.run.driver_config().split, "split by default");
        let config = cli.run.engine_config(Zonos3EngineConfig::new());
        assert_eq!(config.sampler.temperature, 0.5);
        assert_eq!(config.sampler.top_k, 40);
        assert_eq!(config.sampler.top_p, 0.69, "the release's");
        assert_eq!(config.max_new_frames, Some(90));
        assert_eq!(cli.run.read_text().unwrap(), "Hello there.");
        assert_eq!(cli.run.voice, DEFAULT_TTS_VOICE);

        let cli =
            Cli::try_parse_from(["tts", "-o", "o.wav", "--language", "fr_fr", "Bonjour"]).unwrap();
        assert_eq!(
            cli.run.driver_config().normalization,
            zonos3_release_normalization()
        );
        let request = cli.run.request("Bonjour".into(), 3);
        assert_eq!(request.language.as_deref(), Some("fr_fr"));
        assert_eq!(
            request.voice,
            TtsVoiceChoice::Stock(DEFAULT_TTS_VOICE.into())
        );
        assert_eq!(request.seed, Some(3));

        let cli = Cli::try_parse_from(["tts", "-o", "o.wav", "--no-split", "Hi"]).unwrap();
        assert!(!cli.run.driver_config().split);
    }

    #[test]
    fn test_clock_seed() {
        assert!(clock_seed() < 1 << 31);
    }
}
