//! The Qwen3-ASR feature extractor.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::TensorData,
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::speech::whisper::blocks::WhisperFrontEndConfig,
    ops::signal::perceptive_audio::{
        PerceptiveAudioConverter,
        PerceptiveAudioConverterMeta,
    },
};

/// Common meta for [`Qwen3AsrFrontEnd`] and [`Qwen3AsrFrontEndConfig`].
pub trait Qwen3AsrFrontEndMeta {
    /// The sample rate, in Hz.
    fn sample_rate(&self) -> usize;

    /// The hop between frames, in samples.
    fn hop(&self) -> usize;

    /// The mel channel count.
    fn n_mels(&self) -> usize;

    /// The shortest window, in samples.
    fn min_samples(&self) -> usize;

    /// The valid frames of a `samples`-long utterance: `ceil(samples / hop)`.
    fn valid_frames(
        &self,
        samples: usize,
    ) -> usize {
        samples.div_ceil(self.hop())
    }
}

/// Config for [`Qwen3AsrFrontEnd`].
///
/// Implements [`Qwen3AsrFrontEndMeta`].
#[derive(Config, Debug)]
pub struct Qwen3AsrFrontEndConfig {
    /// The mel channel count.
    #[config(default = "128")]
    pub n_mels: usize,

    /// The shortest window, in seconds: shorter audio is zero-padded to it.
    #[config(default = "30")]
    pub min_seconds: usize,

    /// The Whisper front end this one is.
    #[config(default = "WhisperFrontEndConfig::new()")]
    pub whisper: WhisperFrontEndConfig,
}

impl Qwen3AsrFrontEndMeta for Qwen3AsrFrontEndConfig {
    fn sample_rate(&self) -> usize {
        self.whisper.sample_rate
    }

    fn hop(&self) -> usize {
        self.whisper.hop()
    }

    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn min_samples(&self) -> usize {
        self.min_seconds * self.whisper.sample_rate
    }
}

impl ModuleInit<Qwen3AsrFrontEnd> for Qwen3AsrFrontEndConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3AsrFrontEnd> {
        Ok(Qwen3AsrFrontEnd {
            converter: self.whisper.try_init_audio_converter(self.n_mels, device)?,
            config: self.clone(),
        })
    }
}

/// Qwen3-ASR's feature extractor: Whisper's, over windows of at least 30 s.
///
/// `transformers`' `Qwen3ASRFeatureExtractor` is Whisper's: the waveform
/// zero-padded to 30 s, a centred 25 ms Hann STFT every 10 ms with its last
/// frame dropped, log10 mels floored 8 dB under the window's maximum, then
/// `(x + 4) / 4`. A longer utterance is one window of its own length. The
/// first [`valid_frames`](Qwen3AsrFrontEndMeta::valid_frames) are valid; the
/// encoder reads only those.
///
/// It is [`WhisperFrontEndConfig`]'s converter and packaging, shared. The
/// converter's tables are bare tensors: moved by `to_device`, not recorded,
/// and skipped by `ModuleMapper` passes.
///
/// Built by [`Qwen3AsrFrontEndConfig`].
/// Implements [`Qwen3AsrFrontEndMeta`].
#[derive(Module, Debug)]
pub struct Qwen3AsrFrontEnd {
    /// The config this was built from.
    #[module(skip)]
    pub config: Qwen3AsrFrontEndConfig,

    /// The log-mel converter.
    pub converter: PerceptiveAudioConverter,
}

impl Qwen3AsrFrontEndMeta for Qwen3AsrFrontEnd {
    fn sample_rate(&self) -> usize {
        self.converter.sample_rate()
    }

    fn hop(&self) -> usize {
        self.converter.hop()
    }

    fn n_mels(&self) -> usize {
        self.converter.n_mels()
    }

    fn min_samples(&self) -> usize {
        self.config.min_samples()
    }
}

impl Qwen3AsrFrontEnd {
    /// Extracts one utterance's features.
    ///
    /// # Arguments
    /// * `samples` - mono audio at
    ///   [`sample_rate`](Qwen3AsrFrontEndMeta::sample_rate).
    ///
    /// # Returns
    /// `[n_mels, frames]` features, and the valid frame count.
    ///
    /// # Errors
    /// As the converter's context.
    pub fn forward(
        &self,
        samples: &[f32],
        device: &Device,
    ) -> BunsenResult<(Tensor<2>, usize)> {
        let n = samples.len().max(self.min_samples());
        let mut padded = samples.to_vec();
        padded.resize(n, 0.0);
        let wave = Tensor::<2>::from_data(TensorData::new(padded, [1, n]), device);
        let (mels, context) = self.converter.new_context(1).transform(wave)?;
        let joined = match context.finish() {
            Some(tail) => Tensor::cat(vec![mels, tail], 1),
            None => mels,
        };
        let features = self.config.whisper.package_mels(joined);
        let [_, n_mels, frames] = features.dims();
        let valid = self.valid_frames(samples.len()).min(frames);
        Ok((features.reshape([n_mels, frames]), valid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// A config non-default in every field, and the front end built from it,
    /// answer the meta alike; the converter's tables live on the front end's
    /// device; a short utterance becomes a full 30 s window.
    #[test]
    fn test_meta_devices_and_shapes() {
        let device = cpu_device();
        let config = Qwen3AsrFrontEndConfig::new()
            .with_n_mels(80)
            .with_min_seconds(2)
            .with_whisper(WhisperFrontEndConfig::new().with_range_clamp_db(6.0));
        let front: Qwen3AsrFrontEnd = config.init(&device);
        assert_eq!(front.sample_rate(), config.sample_rate());
        assert_eq!(front.hop(), config.hop());
        assert_eq!(front.n_mels(), config.n_mels());
        assert_eq!(front.min_samples(), config.min_samples());
        assert_eq!(front.devices(), vec![device.clone()]);

        let (features, valid) = front.forward(&vec![0.1; 1000], &device).unwrap();
        assert_eq!(features.dims(), [80, 200]);
        assert_eq!(valid, 7);
    }
}
