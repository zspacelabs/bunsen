use std::{
    fmt::Display,
    path::{
        Path,
        PathBuf,
    },
    str::FromStr,
    sync::Arc,
};

use bunsen::{
    data::{
        cache::BunsenDiskCacheOptions,
        pretrained::{
            Deferred,
            PretrainedCache,
            PretrainedCacheOptions,
            PretrainedFactory,
            ResourceMap,
        },
    },
    errors::BunsenResult,
    kits::speech::{
        silero_vad::{
            SileroVadMeta,
            pretrained::default_silero_factory,
        },
        whisper::{
            WhisperFallbackConfig,
            driver::{
                PresetEmissionPolicy,
                RunningMaxClamp,
                StreamClock,
                TranscriptEvent,
                WhisperBundle,
                WhisperStreamContext,
                WhisperStreamDriver,
                WhisperStreamDriverConfig,
                WhisperTask,
            },
            pretrained::{
                CHECKPOINT,
                OPENAI_LOCAL_DIR,
                VOCABULARY,
                WhisperConstruct,
                default_whisper_factory,
            },
        },
    },
    ops::signal::perceptive_audio::PerceptiveAudioConverterMeta,
};
use burn::prelude::Backend;

/// Where fetched weights live, and whether fetching is allowed.
#[derive(clap::Args, Debug)]
pub struct WeightsCacheArgs {
    /// Directory for fetched weights; `$BUNSEN_CACHE_DIR`, then the
    /// platform's cache directory, when omitted.
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Never reach the network: a model not already local is an error.
    #[arg(long)]
    offline: bool,

    /// `openai-whisper`'s download cache, whose files are used in place;
    /// `~/.cache/whisper` when omitted.
    #[arg(long)]
    upstream_cache_dir: Option<PathBuf>,
}

impl WeightsCacheArgs {
    /// Opens the cache.
    pub fn init(&self) -> BunsenResult<PretrainedCache> {
        let mut options = PretrainedCacheOptions::default()
            .with_disk(BunsenDiskCacheOptions::default().with_cache_dir(self.cache_dir.clone()))
            .with_offline(self.offline);
        if let Some(dir) = &self.upstream_cache_dir {
            options = options.with_local_dir(OPENAI_LOCAL_DIR, dir.clone());
        }
        PretrainedCache::new(options)
    }
}

/// What `--model` names: a row of the factory, resolved through the
/// cache (a Hugging Face ref asks the hub what the repo holds, once), or
/// a checkpoint on disk as a one-resource map under [`CHECKPOINT`].
/// Either way a deferred model with the kit's hook for it; nothing here
/// builds one.
pub fn resolve_model(
    factory: &PretrainedFactory<WhisperConstruct>,
    spec: &str,
    cache: &PretrainedCache,
) -> BunsenResult<Deferred<WhisperConstruct>> {
    let path = Path::new(spec);
    if path.is_file() {
        return Deferred::from_map(ResourceMap::given(spec, CHECKPOINT, path));
    }
    factory.resolve(spec, cache)
}

fn gcd(
    a: usize,
    b: usize,
) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(
    a: usize,
    b: usize,
) -> usize {
    a / gcd(a, b) * b
}

/// Parses one of the kit's enums by variant name, in any case: `offline`,
/// `Offline` and `OFFLINE` are the same preset. The kit's enums parse their
/// exact variant names only; the flags are documented in lower case.
fn parse_variant<T>(s: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: Display,
{
    if let Ok(value) = s.parse::<T>() {
        return Ok(value);
    }
    let mut chars = s.chars();
    let capitalized = match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
        None => String::new(),
    };
    capitalized.parse::<T>().map_err(|err| err.to_string())
}

/// The mechanics `transcribe` and `live` share: which model, how it decodes,
/// how audio is pushed, and how emissions are printed.
///
/// A command flattens this beside its own flags, and supplies its own
/// default for the two settings that depend on the source (`--preset`,
/// `--chunk-ms`): a file is read whole, a microphone is read as it speaks.
#[derive(clap::Args, Debug)]
pub struct WhisperDriverArgs {
    /// The model: `provider:ref` or a bare ref from `models list`
    /// (`well-known:openai/tiny.en`, `openai/tiny.en`, `large`), a Hugging
    /// Face repo in `transformers`' layout (`hf:openai/whisper-tiny`, one
    /// file or shards), or a path to a checkpoint. The default is fetched
    /// into the cache on first use (145 MB, digest-checked), or found
    /// where a deployment put it ahead of time.
    #[arg(long, default_value = "openai/base")]
    model: String,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// A `.tiktoken` vocabulary by path, in place of the one the
    /// checkpoint's token layout selects (`multilingual.tiktoken` for a
    /// multilingual checkpoint, `gpt2.tiktoken` for an English-only one),
    /// which comes from the cache or one fetch.
    #[arg(long)]
    vocab: Option<PathBuf>,

    /// Language of the speech, as a Whisper code (`en`, `ja`, ...); detected
    /// from the first window when omitted.
    #[arg(long)]
    language: Option<String>,

    /// Transcribe, or translate to English.
    #[arg(
        long,
        value_name = "transcribe|translate",
        value_parser = parse_variant::<WhisperTask>,
        default_value = "transcribe",
    )]
    task: WhisperTask,

    /// Emit timestamp tokens and split segments on them, seeking to the last
    /// closed timestamp, as upstream's `transcribe()` does.
    #[arg(long)]
    timestamps: bool,

    /// Beams per window; one is greedy.
    #[arg(long, default_value = "5")]
    beam_size: usize,

    /// Cap on generated tokens per window.
    #[arg(long, default_value = "224")]
    max_tokens: usize,

    /// Prompt each window with the transcript so far?
    #[arg(
      long,
      action=clap::ArgAction::Set,
      default_value_t = true,
      default_missing_value = "true"
    )]
    prompt_carry: bool,

    /// Climb upstream's temperature ladder when a window's decode fails its
    /// thresholds; without this, temperature zero alone.
    #[arg(
      long,
      action=clap::ArgAction::Set,
      default_value_t = true,
      default_missing_value = "true"
    )]
    fallback: bool,

    /// When to decode, and when a decode is final: `offline` (whole windows,
    /// all final), `conservative` (speech regions as well, all final), or
    /// `responsive` (drafts every 600 ms of speech besides). The last two
    /// load the bundled VAD. Each command has its own default: `offline`
    /// for `transcribe`, `conservative` for `live`.
    #[arg(
        long,
        value_name = "offline|conservative|responsive",
        value_parser = parse_variant::<PresetEmissionPolicy>,
    )]
    preset: Option<PresetEmissionPolicy>,

    /// Milliseconds of audio per push: how `transcribe` feeds a file, as a
    /// live loop would, and how `live` batches the capture callbacks. Each
    /// command has its own default: 1000 for `transcribe`, 250 for `live`.
    #[arg(long)]
    chunk_ms: Option<usize>,

    /// Print each segment's ids beside its text.
    #[arg(long)]
    ids: bool,
}

impl WhisperDriverArgs {
    /// Loads `--model` at the precision it ships in, with its vocabulary.
    ///
    /// A name is resolved against the default whisper factory; a path to a
    /// checkpoint is a given map; either way the model carries the kit's
    /// hook for it. Every
    /// resource of the map comes from the cache, a local source, or a
    /// digest-checked download; a name's checkpoint is checked against the
    /// prefab it promised before it is materialized; and the vocabulary is
    /// the one the checkpoint's layout selects, or the file `--vocab`
    /// names, which is trusted as given.
    ///
    /// `OpenAI`'s checkpoints are fp16 while the mel front end works in the
    /// backend's float, but the model casts at its own edges — mels in,
    /// logits out — so nothing here has to re-type it.
    pub fn load_bundle<B: Backend>(
        &self,
        cache: &PretrainedCache,
        device: &B::Device,
    ) -> BunsenResult<Arc<WhisperBundle<B>>> {
        let factory = default_whisper_factory()?;
        let mut model = resolve_model(&factory, &self.model, cache)?;
        if let Some(path) = &self.vocab {
            model = model.with_overlay(ResourceMap::given("--vocab", VOCABULARY, path))?;
        }
        model.load_bundle::<B>(cache, device)
    }

    /// Initialize the cache.
    pub fn init_cache(&self) -> BunsenResult<PretrainedCache> {
        self.cache.init()
    }

    /// The emission preset: `--preset`, or the command's `default`.
    pub fn preset(
        &self,
        default: PresetEmissionPolicy,
    ) -> PresetEmissionPolicy {
        self.preset.unwrap_or(default)
    }

    /// Samples per push: `--chunk-ms`, or the command's `default_ms`, at
    /// the driver's rate, rounded up to the driver's grain.
    ///
    /// The grain is a whole number of mel hops, encoder grid steps and, when
    /// a VAD is attached, its chunks (2560 samples at 16 kHz with the
    /// bundled VAD; 320 without), so a push leaves nothing staged behind
    /// it and every push has the same shape: the front end, the gate and
    /// the autotuned kernels behind them see one size, not a drift of
    /// remainders.
    pub fn chunk_samples<B: Backend>(
        &self,
        driver: &WhisperStreamDriver<B>,
        default_ms: usize,
    ) -> usize {
        let want = (self.chunk_ms.unwrap_or(default_ms) * driver.sample_rate() / 1000).max(1);
        let mut grain = lcm(driver.audio_converter().hop(), driver.encoder_grid());
        if let Some(vad) = driver.silero_vad_model() {
            grain = lcm(grain, vad.chunk_size());
        }
        want.div_ceil(grain) * grain
    }

    /// Load and setup the [`WhisperStreamDriver`].
    ///
    /// `default_preset` is the command's emission preset, for when
    /// `--preset` is omitted.
    pub fn init_driver<B: Backend>(
        &self,
        device: &B::Device,
        default_preset: PresetEmissionPolicy,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
        let preset = self.preset(default_preset);
        let cache = self.init_cache()?;
        let bundle = self.load_bundle::<B>(&cache, device)?;
        log::info!("Loaded Whisper: {bundle}");

        // The token layout follows from the vocabulary size, and the
        // vocabulary from the layout, through the same cache as the
        // weights; the driver takes its detokenizer and upstream's default
        // suppress list from the bundle.
        let ids = *bundle.layout.ids();
        let language = if ids.is_multilingual() {
            self.language.clone()
        } else {
            if self.language.is_some() {
                log::warn!("English-only checkpoint: --language and --task do not apply");
            }
            None
        };
        let fallback = if self.fallback {
            WhisperFallbackConfig::upstream()
        } else {
            WhisperFallbackConfig::new()
        };

        let mut driver: WhisperStreamDriver<B> = WhisperStreamDriverConfig::new()
            .with_language(language)
            .with_task(self.task)
            .with_timestamps(self.timestamps)
            .with_beam_size(self.beam_size)
            .with_max_tokens(self.max_tokens)
            .with_condition_on_previous_text(self.prompt_carry)
            .with_emission(preset.into())
            .with_fallback(fallback)
            .init_from_bundle(bundle, device)?;

        if preset != PresetEmissionPolicy::Offline {
            // The bundled burnpack, through the same cache as the weights:
            // written in from the binary on first use, cached after.
            let vad = default_silero_factory()?
                .load::<B>("bundled:silero/vad", &cache, device)?
                .handle;
            driver = driver.with_vad(vad.expect_branch(16000).clone(), Default::default())?;
        }

        if driver.detects_language() {
            log::info!("language: detected from the first window");
        } else {
            log::info!("prompt: {:?}", driver.prompt());
        }

        Ok(driver)
    }

    /// Opens one stream through `driver`, printing as it goes.
    ///
    /// A bare stream: a clock from zero at the model's rate, and the running
    /// maximum as the mel clamp reference. A source that knows its capture
    /// times anchors them through
    /// [`anchor_write_read`](TranscriptStream::anchor_write_read).
    pub fn open_stream<B: Backend>(
        &self,
        driver: &WhisperStreamDriver<B>,
    ) -> BunsenResult<TranscriptStream<B>> {
        let ctx = driver.new_context(
            StreamClock::uniform(driver.sample_rate()),
            RunningMaxClamp::new(),
        )?;
        Ok(TranscriptStream {
            ctx,
            detects_language: driver.detects_language(),
            announced: false,
            ids: self.ids,
            last_end: 0.0,
        })
    }
}

/// One stream through the driver, and the printing `transcribe` and `live`
/// share: every emission is reported as it arrives, and the language once
/// it is known.
///
/// Opened by [`WhisperDriverArgs::open_stream`]; pushed with
/// [`write_read`](Self::write_read) or
/// [`anchor_write_read`](Self::anchor_write_read); ended with
/// [`end_read`](Self::end_read).
pub struct TranscriptStream<B: Backend> {
    ctx: WhisperStreamContext<B>,
    detects_language: bool,
    announced: bool,
    ids: bool,
    last_end: f64,
}

impl<B: Backend> TranscriptStream<B> {
    /// Pushes samples at the model's rate, and reports what came out.
    pub fn write_read(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<()> {
        let events = self.ctx.write_read(samples)?;
        self.report(&events);
        Ok(())
    }

    /// Anchors the clock: the first of `samples` was captured at media time
    /// `time`, in seconds. Then as [`write_read`](Self::write_read).
    pub fn anchor_write_read(
        &mut self,
        time: f64,
        samples: &[f32],
    ) -> BunsenResult<()> {
        let events = self.ctx.anchor_write_read(time, samples)?;
        self.report(&events);
        Ok(())
    }

    /// Ends the stream, and reports whatever was left past the seek pointer.
    pub fn end_read(&mut self) -> BunsenResult<()> {
        let events = self.ctx.end_read()?;
        self.report(&events);
        Ok(())
    }

    /// Samples pushed so far.
    pub fn samples_seen(&self) -> usize {
        self.ctx.samples_seen()
    }

    /// Media time of the end of the last emission, in seconds; zero before
    /// any.
    pub fn last_end(&self) -> f64 {
        self.last_end
    }

    fn report(
        &mut self,
        events: &[TranscriptEvent],
    ) {
        // Detection runs on the first window decoded, so the language is
        // known once anything has been emitted; say so before the text.
        if !self.announced
            && self.detects_language
            && let Some(code) = self.ctx.language()
        {
            log::debug!("language: {code}");
            self.announced = true;
        }
        for event in events {
            self.last_end = event.segment().end;
            report(event, self.ids);
        }
        log::trace!(
            "stream: {} samples seen, seek {}, {} frames pending, speaking {}, {} regions pending",
            self.ctx.samples_seen(),
            self.ctx.seek(),
            self.ctx.pending_frames(),
            self.ctx.is_speaking(),
            self.ctx.regions_pending(),
        );
    }
}

/// One line per emission: a draft is marked `~`, a commit is not.
fn report(
    emission: &TranscriptEvent,
    ids: bool,
) {
    let segment = emission.segment();
    let mark = if emission.is_committed() { ' ' } else { '~' };

    log::info!("{mark}[{:>8.2} --> {:>8.2}]", segment.start, segment.end);
    println!("{}", segment.text.as_deref().unwrap_or("").trim());

    if ids {
        log::info!("ids: {:?}", segment.tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(160, 320), 320);
        assert_eq!(lcm(320, 512), 2560);
        assert_eq!(lcm(7, 1), 7);
    }

    #[test]
    fn test_parse_variant_any_case() {
        for (given, want) in [
            ("offline", PresetEmissionPolicy::Offline),
            ("Conservative", PresetEmissionPolicy::Conservative),
            ("RESPONSIVE", PresetEmissionPolicy::Responsive),
        ] {
            assert_eq!(parse_variant::<PresetEmissionPolicy>(given).unwrap(), want);
        }
        assert_eq!(
            parse_variant::<WhisperTask>("translate").unwrap(),
            WhisperTask::Translate
        );
        assert!(parse_variant::<PresetEmissionPolicy>("bogus").is_err());
        assert!(parse_variant::<PresetEmissionPolicy>("").is_err());
    }
}
