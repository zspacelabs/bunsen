//! The bundled checkpoint's old entry point, kept one release as a
//! deprecated alias of [`load_named`].

use burn::prelude::Backend;

use crate::{
    data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    },
    errors::BunsenResult,
    kits::speech::whisper::{
        blocks::{
            Whisper,
            WhisperApiConfig,
        },
        pretrained::load_named,
    },
};

impl<B: Backend> Whisper<B> {
    /// Loads `OpenAI`'s multilingual Whisper *base* checkpoint: `openai/base`
    /// through [`load_named`], with a default [`PretrainedCache`].
    ///
    /// The checkpoint and its vocabulary come through the default cache:
    /// fetched on first use, or found where a deployment put them ahead of
    /// time. That is what the name-to-model pathway does for every model,
    /// which is why this is deprecated in its favor:
    ///
    /// ```no_run
    /// # use burn::backend::Wgpu;
    /// # use bunsen::data::pretrained::{PretrainedCache, PretrainedCacheOptions};
    /// # use bunsen::kits::speech::whisper::{Whisper, pretrained::load_named};
    /// # use bunsen::support::testing::default_device;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = default_device();
    /// let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
    /// let (model, cfg) = load_named::<Wgpu>("openai/base", &cache, &device)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// As [`load_named`].
    #[deprecated(note = "use `pretrained::load_named(\"openai/base\", &cache, device)`")]
    pub fn load_pretrained_16khz_fp16_base(
        device: &B::Device
    ) -> BunsenResult<(Self, WhisperApiConfig)> {
        let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
        load_named::<B>("openai/base", &cache, device)
    }
}
