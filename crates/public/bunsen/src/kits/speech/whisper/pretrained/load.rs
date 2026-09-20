//! The bundled checkpoint's old entry point, kept one release as a
//! deprecated alias of [`load_named`].

use burn::prelude::Backend;

use crate::{
    data::pretrained::{
        WeightsCache,
        WeightsCacheOptions,
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
    /// through [`load_named`], with a default [`WeightsCache`].
    ///
    /// Under the `whisper-weights` feature the file `bunsen-bundled-whisper`
    /// fetched at build time is the first source of `openai/base`, so this
    /// reads it in place and reaches no network. It is what this function
    /// always did; the difference is that the name-to-model pathway now does
    /// it for every model, which is why this is deprecated in its favor:
    ///
    /// ```no_run
    /// # use burn::backend::Wgpu;
    /// # use bunsen::data::pretrained::{WeightsCache, WeightsCacheOptions};
    /// # use bunsen::kits::speech::whisper::{Whisper, pretrained::load_named};
    /// # use bunsen::support::testing::default_device;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = default_device();
    /// let cache = WeightsCache::new(WeightsCacheOptions::default())?;
    /// let (model, cfg) = load_named::<Wgpu>("openai/base", &cache, &device)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// As [`load_named`].
    #[deprecated(
        note = "use `pretrained::load_named(\"openai/base\", &cache, device)`; under `whisper-weights` the bundled file is its first source"
    )]
    pub fn load_pretrained_16khz_fp16_base(
        device: &B::Device
    ) -> BunsenResult<(Self, WhisperApiConfig)> {
        let cache = WeightsCache::new(WeightsCacheOptions::default())?;
        load_named::<B>("openai/base", &cache, device)
    }
}
