use std::{
    path::PathBuf,
    sync::Arc,
};

use bunsen::{
    errors::BunsenResult,
    kits::speech::{
        silero_vad::SileroVad,
        whisper::{
            Whisper,
            WhisperApiConfig,
            WhisperFallbackConfig,
            driver::{
                PresetEmissionPolicy,
                WhisperStreamDriver,
                WhisperStreamDriverConfig,
                WhisperTask,
            },
            logit_filters::default_filters,
            pretrained::bundled_vocabulary,
        },
    },
};
use burn::prelude::Backend;

use crate::models::{
    loader::{
        ModelRef,
        load_model,
    },
    weights_cache::{
        WeightsCache,
        WeightsCacheOptions,
    },
};

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

    /// `openai-whisper`'s download cache, read as a local source;
    /// `~/.cache/whisper` when omitted.
    #[arg(long)]
    upstream_cache_dir: Option<PathBuf>,
}

impl WeightsCacheArgs {
    /// Opens the cache.
    pub fn init(&self) -> BunsenResult<WeightsCache> {
        WeightsCache::new(WeightsCacheOptions {
            cache_dir: self.cache_dir.clone(),
            offline: self.offline,
            upstream_cache_dir: self.upstream_cache_dir.clone(),
        })
    }
}

#[derive(clap::Args, Debug)]
pub struct WhisperDriverArgs {
    /// The model: `provider/name` or a bare name from `models list`
    /// (`openai/tiny.en`, `large`), or a path to a checkpoint. The default
    /// is the checkpoint bunsen bundles, so it needs no network.
    #[arg(long, default_value = "openai/base")]
    model: String,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

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
    /// Loads `--model` at the precision it ships in.
    ///
    /// The name is resolved against the pretrained index, or taken as a
    /// path; the weights come from the cache, a local source, or a
    /// digest-checked download; and the checkpoint is checked against the
    /// prefab its name promised before it is materialized.
    ///
    /// `OpenAI`'s checkpoints are fp16 while the mel front end works in the
    /// backend's float, but the model casts at its own edges — mels in,
    /// logits out — so nothing here has to re-type it.
    pub fn load_model<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<(Whisper<B>, WhisperApiConfig)> {
        let model = ModelRef::resolve(&self.model)?;
        let mut cache = self.cache.init()?;
        load_model(&model, &mut cache, device)
    }

    /// Load and setup the [`WhisperStreamDriver`].
    pub fn init_driver<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
        let (model, cfg) = self.load_model(device)?;
        log::info!(
            "model: {} n_mels, vocabulary {}, d_model {}, {} + {} layers",
            cfg.n_mels,
            cfg.vocab_size,
            cfg.d_model,
            cfg.n_encoder_layers,
            cfg.n_decoder_layers,
        );

        // The token layout follows from the vocabulary size, and the bundled
        // vocabulary follows from the layout: nothing here is typed in.
        let policy = cfg.token_layout.policy_for_vocab(cfg.vocab_size)?;
        let ids = *policy.ids();
        let ranks = bundled_vocabulary(&ids)?;
        let detokenizer = policy.detokenizer(&ranks)?;
        let filters = default_filters::<B>(&ranks, &ids);

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
            .init_with_layout(model, policy, device)?
            .with_detokenizer(Arc::new(detokenizer))
            .with_logit_filters(filters);
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
