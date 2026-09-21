use std::{
    path::{
        Path,
        PathBuf,
    },
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
        silero_vad::pretrained::default_silero_factory,
        whisper::{
            WhisperFallbackConfig,
            driver::{
                PresetEmissionPolicy,
                WhisperBundle,
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

/// What `--model` names: a row of the factory, or a checkpoint on disk as
/// a one-resource map under [`CHECKPOINT`]. Either way a deferred model
/// with the kit's hook for it; nothing here builds one.
pub fn resolve_model(
    factory: &PretrainedFactory<WhisperConstruct>,
    spec: &str,
) -> BunsenResult<Deferred<WhisperConstruct>> {
    let path = Path::new(spec);
    if path.is_file() {
        return Deferred::from_map(ResourceMap::given(spec, CHECKPOINT, path));
    }
    factory.resolve(spec)
}

#[derive(clap::Args, Debug)]
pub struct WhisperDriverArgs {
    /// The model: `provider:ref` or a bare ref from `models list`
    /// (`well-known:openai/tiny.en`, `openai/tiny.en`, `large`), or a path
    /// to a checkpoint. The default is fetched into the cache on first use
    /// (145 MB, digest-checked), or found where a deployment put it ahead
    /// of time.
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
    #[arg(long, default_value_t = WhisperTask::Transcribe)]
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

    /// When to decode, and when a decode is final.
    #[arg(long, default_value_t = PresetEmissionPolicy::Offline)]
    preset: PresetEmissionPolicy,
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
        let mut model = resolve_model(&factory, &self.model)?;
        if let Some(path) = &self.vocab {
            model = model.with_overlay(ResourceMap::given("--vocab", VOCABULARY, path))?;
        }
        model.load_bundle::<B>(cache, device)
    }

    /// Initialize the cache.
    pub fn init_cache(&self) -> BunsenResult<PretrainedCache> {
        self.cache.init()
    }

    /// Load and setup the [`WhisperStreamDriver`].
    pub fn init_driver<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
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
            .with_emission(self.preset.into())
            .with_fallback(fallback)
            .init_from_bundle(bundle, device)?;

        if self.preset != PresetEmissionPolicy::Offline {
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
}
