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
        speech::tts::driver::{
            TtsDriverConfig,
            TtsRequest,
            TtsSamplingOverrides,
            TtsStop,
            TtsVoiceChoice,
        },
        support::text::TextNormalizationConfig,
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
    tts::{
        TtsFamily,
        TtsFamilyArgs,
        choose_tts_model,
    },
};

/// Speaks a text, to a WAV file.
#[derive(clap::Args, Debug)]
pub struct TtsRunCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The device. A model loads at the precision it ships in where the
    /// device accelerates it, else in f32 (`LoadPolicy::performance`);
    /// `--min-precision` puts a floor under that.
    #[clap(flatten)]
    device: BackendArgs,

    /// The narrowest float the checkpoint's weights are placed in.
    #[clap(flatten)]
    precision: MinPrecisionArgs,

    /// The checkpoint, chosen among every TTS family's builds by `--model`
    /// and the stage flags; the default checkpoint when nothing is named.
    #[clap(flatten)]
    choose: ChooseArgs,

    /// The family the checkpoint is chosen among (`zonos3`, `qwen3-tts`).
    ///
    /// Every family's when omitted. Needed for a provider ref
    /// (`hf:org/repo`), which names no family.
    #[arg(long, value_enum)]
    family: Option<TtsFamily>,

    /// The stock voice the speech is in (`tts voices` lists them).
    ///
    /// The model's default voice when omitted: ZONOS3's Savannah; Qwen3-TTS
    /// has none, and needs one named.
    #[arg(long)]
    voice: Option<String>,

    /// A voice described in words, in place of `--voice`.
    ///
    /// For a model that designs one (Qwen3-TTS `VoiceDesign`).
    #[arg(long, conflicts_with = "voice")]
    describe: Option<String>,

    /// A style instruction, for a model that takes one.
    ///
    /// Qwen3-TTS `CustomVoice` 1.7B takes one; a model that takes none refuses
    /// it unless `--strip-instruction`.
    #[arg(long)]
    instruct: Option<String>,

    /// Drop `--instruct`, with a note, for a model that takes none.
    #[arg(long)]
    strip_instruction: bool,

    /// Where the speech is written: a mono 16-bit WAV file at the model's
    /// rate.
    ///
    /// Needed unless the speech is played (`--play`, with `live`).
    #[arg(long, short)]
    out: Option<PathBuf>,

    /// Whether, and where, the speech is played.
    #[cfg(feature = "live")]
    #[clap(flatten)]
    playback: crate::tts::PlaybackArgs,

    /// Seeds the takes, to repeat them; a seed from the clock when omitted.
    ///
    /// The seed is printed with the speech either way.
    #[arg(long)]
    seed: Option<u64>,

    /// How the text is normalized: `release` (the default), or specs, in order.
    ///
    /// `release` is the model's release's; a spec names a normalizer, as
    /// `text normalize` takes them, and repeated they run in order. The
    /// release's is ZONOS3's Zyphra normalizer (line breaks collapsed,
    /// then numbers, money and dates spoken), Qwen3-TTS's the text as typed.
    /// A spec: `none`, `collapse-line-breaks`, `zyphra`, `tokenizers:NFKC`,
    /// `FILE.json`, ... (`text normalizers` lists them).
    #[arg(long = "normalize", value_name = "SPEC")]
    normalizers: Vec<String>,

    /// The flags of one family's engine.
    #[clap(flatten)]
    family_flags: TtsFamilyArgs,

    /// The text's language, as one of the model's keys.
    ///
    /// ZONOS3's request codes (`en_us`, `fr_fr`) or a language tag;
    /// Qwen3-TTS's languages (`english`, `sichuan_dialect`) or `auto`. The
    /// model's default when omitted (ZONOS3's `en_us`, Qwen3-TTS's
    /// `auto`).
    #[arg(long)]
    language: Option<String>,

    /// The sampling temperature; the release's when omitted.
    #[arg(long)]
    temperature: Option<f64>,

    /// Codes kept per choice, the most probable; the release's when
    /// omitted.
    #[arg(long)]
    top_k: Option<usize>,

    /// The probability mass kept per choice; the release's when omitted.
    #[arg(long)]
    top_p: Option<f64>,

    /// Speak the text as one take, however long.
    ///
    /// A long text is otherwise spoken as several takes, split at its
    /// sentences (about 20 seconds each), joined by a short pause.
    #[arg(long)]
    no_split: bool,

    /// The longest take, in seconds; as long as the model allows when
    /// omitted.
    #[arg(long)]
    max_seconds: Option<f64>,

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
        let (mut model, family) = choose_tts_model(&self.choose.model, &flags, self.family)?;
        if !flags.presets.presets.is_empty() {
            let presets: Vec<&str> = flags.presets.presets.iter().map(String::as_str).collect();
            model = with_presets(&model, &presets)?;
        }
        if let Some(patch) = self.choose.patch()? {
            model = with_patch(&model, &patch)?;
        }
        let synthesizer = family.load(
            &model,
            &cache,
            &device,
            self.precision.load_policy(),
            self.driver_config(family)?,
            &self.family_flags,
        )?;

        let seed = self.seed.unwrap_or_else(clock_seed);
        let request = self.request(text, seed);
        let started = Instant::now();
        let speech = synthesizer.synthesize(&request)?;
        let spoken = started.elapsed().as_secs_f64();
        for note in &speech.notes {
            log::warn!("{}: {}", note.stage, note.note);
        }
        if speech.takes.iter().any(|take| take.stop == TtsStop::Budget) {
            log::warn!("a take ran out of frames before it ended: raise --max-seconds");
        }
        if let Some(out) = &self.out {
            save_wav_mono(out, &speech.samples, speech.sample_rate)?;
        }
        let steps: usize = speech.takes.iter().map(|take| take.steps).sum();
        let place = self
            .out
            .as_ref()
            .map_or_else(|| "the speech".to_string(), |out| out.display().to_string());
        let voice = self
            .voice
            .clone()
            .or_else(|| self.describe.as_ref().map(|_| "a described".to_string()))
            .or_else(|| synthesizer.capabilities().default_voice.clone())
            .unwrap_or_else(|| "the default".to_string());
        eprintln!(
            "{place}: {:.2} s in {voice}'s voice ({model}), seed {seed}; {} takes synthesized \
             in {spoken:.2} s ({steps} steps, {:.1} a second)",
            speech.seconds(),
            speech.takes.len(),
            steps as f64 / spoken,
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

    /// The request: the text in the flags' voice, language and instruction,
    /// seeded, at their sampling and length.
    fn request(
        &self,
        text: String,
        seed: u64,
    ) -> TtsRequest {
        let voice = match (&self.voice, &self.describe) {
            (Some(name), _) => TtsVoiceChoice::Stock(name.clone()),
            (None, Some(description)) => TtsVoiceChoice::Described(description.clone()),
            (None, None) => TtsVoiceChoice::Default,
        };
        let mut request = TtsRequest::new(text)
            .with_voice(voice)
            .with_seed(seed)
            .with_sampling(TtsSamplingOverrides {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                ..TtsSamplingOverrides::default()
            });
        request.language.clone_from(&self.language);
        request.instruction.clone_from(&self.instruct);
        request.max_seconds = self.max_seconds;
        request
    }

    /// The driver's config for `family`: the flags' normalization,
    /// splitting and instruction stripping.
    ///
    /// # Errors
    /// A spec's, as [`TextNormalizationConfig::parse_specs`] reads it.
    fn driver_config(
        &self,
        family: TtsFamily,
    ) -> BunsenResult<TtsDriverConfig> {
        let normalization = match self.normalizers.as_slice() {
            [] => family.release_normalization(),
            [only] if only == "release" => family.release_normalization(),
            specs => TextNormalizationConfig::parse_specs(specs)?,
        };
        Ok(TtsDriverConfig::new()
            .with_normalization(normalization)
            .with_split(!self.no_split)
            .with_strip_instruction(self.strip_instruction))
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

    fn cli(argv: &[&str]) -> TtsRunCommand {
        Cli::try_parse_from([&["tts"], argv].concat()).unwrap().run
    }

    /// Without `--out` or `--play`, the speech has nowhere to go.
    #[test]
    fn test_needs_a_destination() {
        let matches = clap::ArgMatches::default();
        let error = cli(&["hi"]).run(&matches).unwrap_err();
        assert_eq!(error.kind(), bunsen::errors::BunsenErrorKind::Illegal);
        assert!(error.to_string().contains("--out"), "{error}");
    }

    /// The request carries the flags: the voice (named, described, or the
    /// model's), the language and instruction, the seed, the sampling and the
    /// take length; the text joins its words.
    #[test]
    fn test_request() {
        let run = cli(&[
            "-o",
            "out.wav",
            "--voice",
            "Ryan",
            "--language",
            "english",
            "--instruct",
            "slowly",
            "--temperature",
            "0.5",
            "--top-k",
            "40",
            "--max-seconds",
            "9",
            "Hello",
            "there.",
        ]);
        assert_eq!(run.read_text().unwrap(), "Hello there.");
        let request = run.request("Hello there.".into(), 3);
        assert_eq!(request.voice, TtsVoiceChoice::Stock("Ryan".into()));
        assert_eq!(request.language.as_deref(), Some("english"));
        assert_eq!(request.instruction.as_deref(), Some("slowly"));
        assert_eq!(request.seed, Some(3));
        assert_eq!(request.max_seconds, Some(9.0));
        assert_eq!(request.sampling.temperature, Some(0.5));
        assert_eq!(request.sampling.top_k, Some(40));
        assert_eq!(request.sampling.top_p, None, "the release's");

        let described = cli(&["-o", "o.wav", "--describe", "a calm voice", "Hi"]);
        assert_eq!(
            described.request("Hi".into(), 1).voice,
            TtsVoiceChoice::Described("a calm voice".into())
        );
        let default = cli(&["-o", "o.wav", "Hi"]).request("Hi".into(), 1);
        assert_eq!(default.voice, TtsVoiceChoice::Default);
        assert_eq!(default.language, None);
        assert!(Cli::try_parse_from(["tts", "--voice", "a", "--describe", "b", "x"]).is_err());
    }

    /// The driver's config: each family's release normalization by default
    /// (or `release`), specs in order otherwise; splitting unless
    /// `--no-split`; stripping on request.
    #[test]
    fn test_driver_config() {
        let run = cli(&["-o", "o.wav", "Hi"]);
        let zonos3 = run.driver_config(TtsFamily::Zonos3).unwrap();
        assert_eq!(
            zonos3.normalization,
            TtsFamily::Zonos3.release_normalization()
        );
        assert!(zonos3.split && !zonos3.strip_instruction);
        assert_eq!(
            run.driver_config(TtsFamily::Qwen3Tts)
                .unwrap()
                .normalization,
            TextNormalizationConfig::None
        );
        let release = cli(&["-o", "o.wav", "--normalize", "release", "Hi"]);
        assert_eq!(
            release
                .driver_config(TtsFamily::Qwen3Tts)
                .unwrap()
                .normalization,
            TextNormalizationConfig::None
        );

        let run = cli(&[
            "-o",
            "o.wav",
            "--normalize",
            "collapse-line-breaks",
            "--normalize",
            "transcript",
            "--no-split",
            "--strip-instruction",
            "Hi",
        ]);
        let config = run.driver_config(TtsFamily::Qwen3Tts).unwrap();
        assert_eq!(
            config.normalization,
            TextNormalizationConfig::parse_specs(["collapse-line-breaks", "transcript"]).unwrap()
        );
        assert!(!config.split && config.strip_instruction);
        let bad = cli(&["-o", "o.wav", "--normalize", "no-such", "Hi"]);
        assert!(bad.driver_config(TtsFamily::Zonos3).is_err());
    }

    #[test]
    fn test_clock_seed() {
        assert!(clock_seed() < 1 << 31);
    }
}
