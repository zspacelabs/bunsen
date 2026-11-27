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
    kits::speech::support::log_mel::LogMelFrontEndConfig,
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

    /// The shortest window, in samples: the chunk, which shorter audio is
    /// zero-padded to.
    fn min_samples(&self) -> usize;

    /// The shortest audio whose frames are all valid, in samples: shorter
    /// audio is zero-padded to it, and the padding counted as valid.
    fn min_valid_samples(&self) -> usize;

    /// The valid frames of a `samples`-long utterance:
    /// `ceil(max(samples, min_valid_samples) / hop)`.
    fn valid_frames(
        &self,
        samples: usize,
    ) -> usize {
        samples.max(self.min_valid_samples()).div_ceil(self.hop())
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

    /// The chunk, in seconds (upstream's `chunk_length`): shorter audio is
    /// zero-padded to it, and a construct transcribes at most this much at
    /// once (the utterances the model was trained on).
    #[config(default = "30")]
    pub chunk_seconds: usize,

    /// The shortest audio whose frames are all valid, in samples (upstream's
    /// `min_length`; the original library's half second): shorter audio is
    /// zero-padded to it before the chunk, and the encoder reads the padding.
    #[config(default = "8000")]
    pub min_valid_samples: usize,

    /// The log-mel grid and clamp range this one shares with Whisper.
    #[config(default = "LogMelFrontEndConfig::new()")]
    pub log_mel: LogMelFrontEndConfig,
}

impl Qwen3AsrFrontEndMeta for Qwen3AsrFrontEndConfig {
    fn sample_rate(&self) -> usize {
        self.log_mel.sample_rate
    }

    fn hop(&self) -> usize {
        self.log_mel.hop()
    }

    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn min_samples(&self) -> usize {
        self.chunk_seconds * self.log_mel.sample_rate
    }

    fn min_valid_samples(&self) -> usize {
        self.min_valid_samples
    }
}

impl ModuleInit<Qwen3AsrFrontEnd> for Qwen3AsrFrontEndConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3AsrFrontEnd> {
        Ok(Qwen3AsrFrontEnd {
            converter: self.log_mel.try_init_audio_converter(self.n_mels, device)?,
            config: self.clone(),
        })
    }
}

/// Qwen3-ASR's feature extractor: Whisper's, over windows of at least a chunk
/// (the release's 30 s).
///
/// `transformers`' `Qwen3ASRFeatureExtractor` is Whisper's: the waveform
/// zero-padded to the chunk, a centred 25 ms Hann STFT every 10 ms with its
/// last frame dropped, log10 mels floored 8 dB under the window's maximum, then
/// `(x + 4) / 4`. A longer utterance is one window of its own length. The
/// first [`valid_frames`](Qwen3AsrFrontEndMeta::valid_frames) are valid,
/// audio under [`min_valid_samples`](Qwen3AsrFrontEndConfig::min_valid_samples)
/// counted as padded to it, as upstream pads it; the encoder reads only
/// those.
///
/// It is [`LogMelFrontEndConfig`]'s converter and packaging, shared. The
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

    fn min_valid_samples(&self) -> usize {
        self.config.min_valid_samples
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
        // The front end computes in f32, whatever the device's default.
        let wave = Tensor::<2>::from_data(
            TensorData::new(padded, [1, n]),
            (device, burn::tensor::DType::F32),
        );
        let joined = crate::kits::speech::support::log_mel::log_mels_of(&self.converter, wave)?;
        let features = self.config.log_mel.package_mels(joined);
        let [frames] = crate::contracts::unpack_shape_contract!(
            [1, "n_mels", "frames"],
            &features.dims(),
            &["frames"],
            &[("n_mels", self.n_mels())],
        );
        let n_mels = self.n_mels();
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
            .with_chunk_seconds(2)
            .with_min_valid_samples(4000)
            .with_log_mel(LogMelFrontEndConfig::new().with_range_clamp_db(6.0));
        let front: Qwen3AsrFrontEnd = config.init(&device);
        assert_eq!(front.sample_rate(), config.sample_rate());
        assert_eq!(front.hop(), config.hop());
        assert_eq!(front.n_mels(), config.n_mels());
        assert_eq!(front.min_samples(), config.min_samples());
        assert_eq!(front.min_valid_samples(), 4000);
        assert_eq!(front.devices(), vec![device.clone()]);

        // 1000 samples count as the 4000 they are padded to: 25 frames
        let (features, valid) = front.forward(&vec![0.1; 1000], &device).unwrap();
        assert_eq!(features.dims(), [80, 200]);
        assert_eq!(valid, 25);
        let (_, valid) = front.forward(&vec![0.1; 10_000], &device).unwrap();
        assert_eq!(valid, 63);
    }
}
