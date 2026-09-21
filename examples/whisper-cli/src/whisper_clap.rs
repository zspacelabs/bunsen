use std::{
    path::PathBuf,
    sync::Arc,
};

use bunsen::{
    data::{
        cache::BunsenDiskCacheOptions,
        pretrained::{
            PretrainedCache,
            PretrainedCacheOptions,
            ResourceMap,
        },
    },
    errors::BunsenResult,
    kits::speech::{
        silero_vad::SileroVad,
        whisper::{
            WhisperFallbackConfig,
            WhisperMeta,
            blocks::{
                AudioEncoderMeta,
                TextDecoderMeta,
            },
            driver::{
                PresetEmissionPolicy,
                WhisperBundle,
                WhisperStreamDriver,
                WhisperStreamDriverConfig,
                WhisperTask,
            },
            pretrained::{
                OPENAI_LOCAL_DIR,
                PytorchWhisperScanner,
                VOCABULARY,
                WhisperConstruct,
                resolve_model,
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

/// How a checkpoint is read.
#[derive(clap::Args, Debug)]
pub struct ScannerArgs {
    /// The key the checkpoint keeps its tensors under; `model_state_dict`,
    /// as `OpenAI`'s do, when omitted. An empty string for a checkpoint whose
    /// tensors are at the top level.
    #[arg(long)]
    state_dict_key: Option<String>,
}

impl ScannerArgs {
    /// The scanner these flags describe.
    pub fn scanner(&self) -> PytorchWhisperScanner {
        let scanner = PytorchWhisperScanner::new();
        match self.state_dict_key.as_deref() {
            None => scanner,
            Some("") => scanner.with_top_level_key(None),
            Some(key) => scanner.with_top_level_key(Some(key.to_string())),
        }
    }
}

#[derive(clap::Args, Debug)]
pub struct WhisperDriverArgs {
    /// The model: `provider/name` or a bare name from `models list`
    /// (`openai/tiny.en`, `large`), or a path to a checkpoint. The default
    /// is fetched into the cache on first use (145 MB, digest-checked), or
    /// found where a deployment put it ahead of time.
    #[arg(long, default_value = "openai/base")]
    model: String,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    #[clap(flatten)]
    scanner: ScannerArgs,

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
    /// The name is resolved against the pretrained index, or taken as a
    /// path; every resource of its map comes from the cache, a local
    /// source, or a digest-checked download; the checkpoint is checked
    /// against the prefab its name promised before it is materialized; and
    /// the vocabulary is the one the checkpoint's layout selects, or the
    /// file `--vocab` names, which is trusted as given.
    ///
    /// `OpenAI`'s checkpoints are fp16 while the mel front end works in the
    /// backend's float, but the model casts at its own edges — mels in,
    /// logits out — so nothing here has to re-type it.
    pub fn load_bundle<B: Backend>(
        &self,
        cache: &PretrainedCache,
        device: &B::Device,
    ) -> BunsenResult<Arc<WhisperBundle<B>>> {
        let mut model = resolve_model(&self.model)?;
        if let Some(path) = &self.vocab {
            model = model.with_overlay(ResourceMap::given("--vocab", VOCABULARY, path))?;
        }
        let hook = WhisperConstruct::new().with_scanner(self.scanner.scanner());
        Ok(model.load::<B, _>(cache, &hook, device)?.handle)
    }

    /// Load and setup the [`WhisperStreamDriver`].
    pub fn init_driver<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
        let cache = self.cache.init()?;
        let bundle = self.load_bundle::<B>(&cache, device)?;
        log::info!(
            "model: {} n_mels, vocabulary {}, d_model {}, {} + {} layers",
            bundle.model.n_mels(),
            bundle.model.vocab_size(),
            bundle.model.d_model(),
            bundle.model.encoder().n_layers(),
            bundle.model.decoder().n_layers(),
        );

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
            driver = driver.with_vad(
                SileroVad::<B>::load_16khz_pretrained(device)?,
                Default::default(),
            )?;
        }
        if driver.detects_language() {
            log::info!("language: detected from the first window");
        } else {
            log::info!("prompt: {:?}", driver.prompt());
        }

        Ok(driver)
    }
}
