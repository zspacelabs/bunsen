//! The Voxtral Realtime feature extractor.
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
    kits::speech::whisper::{
        blocks::WhisperFrontEndConfig,
        driver::support::drop_last_frame,
    },
    ops::signal::perceptive_audio::{
        PerceptiveAudioConverter,
        PerceptiveAudioConverterMeta,
    },
};

/// Common meta for [`VoxtralFrontEnd`] and [`VoxtralFrontEndConfig`].
pub trait VoxtralFrontEndMeta {
    /// The sample rate, in Hz.
    fn sample_rate(&self) -> usize;

    /// The hop between frames, in samples.
    fn hop(&self) -> usize;

    /// The mel channel count.
    fn n_mels(&self) -> usize;

    /// The fixed level every log-mel is floored under, in `log10` units.
    fn global_log_mel_max(&self) -> f64;
}

/// Config for [`VoxtralFrontEnd`].
///
/// Implements [`VoxtralFrontEndMeta`].
#[derive(Config, Debug)]
pub struct VoxtralFrontEndConfig {
    /// The mel channel count.
    #[config(default = "128")]
    pub n_mels: usize,

    /// The fixed level every log-mel is floored `whisper.range_clamp_db`
    /// under, in `log10` units: `processor_config.json`'s
    /// `global_log_mel_max`.
    #[config(default = "1.5")]
    pub global_log_mel_max: f64,

    /// The Whisper front end this one is, but for the level.
    #[config(default = "WhisperFrontEndConfig::new()")]
    pub whisper: WhisperFrontEndConfig,
}

impl VoxtralFrontEndMeta for VoxtralFrontEndConfig {
    fn sample_rate(&self) -> usize {
        self.whisper.sample_rate
    }

    fn hop(&self) -> usize {
        self.whisper.hop()
    }

    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn global_log_mel_max(&self) -> f64 {
        self.global_log_mel_max
    }
}

impl ModuleInit<VoxtralFrontEnd> for VoxtralFrontEndConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<VoxtralFrontEnd> {
        Ok(VoxtralFrontEnd {
            converter: self.whisper.try_init_audio_converter(self.n_mels, device)?,
            config: self.clone(),
        })
    }
}

/// Voxtral Realtime's feature extractor: Whisper's log-mels, floored under a
/// fixed level rather than the utterance's own maximum.
///
/// `transformers`' `VoxtralRealtimeFeatureExtractor` is Whisper's large-v3
/// extractor (a centred 25 ms Hann STFT every 10 ms with its last frame
/// dropped, 128 slaney mels, `log10`) but for its floor: 8 under
/// `global_log_mel_max`, a constant, then `(x + 4) / 4`. No statistic of the
/// audio enters, so the features of a stream are the features of each of its
/// parts.
///
/// It is [`WhisperFrontEndConfig`]'s converter and packaging, shared. The
/// converter's tables are bare tensors: moved by `to_device`, not recorded,
/// and skipped by `ModuleMapper` passes.
///
/// Built by [`VoxtralFrontEndConfig`].
/// Implements [`VoxtralFrontEndMeta`].
#[derive(Module, Debug)]
pub struct VoxtralFrontEnd {
    /// The config this was built from.
    #[module(skip)]
    pub config: VoxtralFrontEndConfig,

    /// The log-mel converter.
    pub converter: PerceptiveAudioConverter,
}

impl VoxtralFrontEndMeta for VoxtralFrontEnd {
    fn sample_rate(&self) -> usize {
        self.converter.sample_rate()
    }

    fn hop(&self) -> usize {
        self.converter.hop()
    }

    fn n_mels(&self) -> usize {
        self.converter.n_mels()
    }

    fn global_log_mel_max(&self) -> f64 {
        self.config.global_log_mel_max
    }
}

impl VoxtralFrontEnd {
    /// Packages log-mels into encoder input under the fixed level.
    ///
    /// # Arguments
    /// * `window` - `[batch, frames, n_mels]` log-mels.
    ///
    /// # Returns
    /// `[batch, n_mels, frames]`.
    pub fn package(
        &self,
        window: Tensor<3>,
    ) -> Tensor<3> {
        let [batch] = crate::contracts::unpack_shape_contract!(
            ["batch", "frames", "n_mels"],
            &window.dims(),
            &["batch"],
            &[("n_mels", self.n_mels())]
        );
        let level = Tensor::<1>::full([batch], self.global_log_mel_max(), &window.device());
        self.config.whisper.package_window(window, level)
    }

    /// `frames` packaged frames of silence: the floor every channel of
    /// silent audio reaches.
    ///
    /// # Returns
    /// `[1, n_mels, frames]`.
    pub fn silence(
        &self,
        frames: usize,
        device: &Device,
    ) -> Tensor<3> {
        // Anything at or under the floor packages to the floor.
        let floor = self.global_log_mel_max() - self.config.whisper.range_clamp_db;
        self.package(Tensor::full(
            [1, frames, self.n_mels()],
            floor - 1.0,
            device,
        ))
    }

    /// Extracts the features of a whole recording.
    ///
    /// # Arguments
    /// * `samples` - mono audio at
    ///   [`sample_rate`](VoxtralFrontEndMeta::sample_rate), a non-zero multiple
    ///   of the hop long.
    ///
    /// # Returns
    /// `[1, n_mels, samples/hop]`.
    ///
    /// # Errors
    /// As the converter's context: an empty recording, or one that is not
    /// whole hops.
    pub fn forward(
        &self,
        samples: &[f32],
        device: &Device,
    ) -> BunsenResult<Tensor<3>> {
        let n = samples.len();
        let wave = Tensor::<2>::from_data(TensorData::new(samples.to_vec(), [1, n]), device);
        let (mels, context) = self.converter.new_context(1).transform(wave)?;
        let joined = match context.finish() {
            Some(tail) => Tensor::cat(vec![mels, tail], 1),
            None => mels,
        };
        let features = self.package(drop_last_frame(joined));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [1, "n_mels", "frames"],
            &features.dims(),
            &[("n_mels", self.n_mels()), ("frames", n / self.hop())],
        );
        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// A config non-default in every field, and the front end built from it,
    /// answer the meta alike; silence sits on the floor; a recording of `n`
    /// samples is `n / hop` frames.
    #[test]
    fn test_meta_floor_and_shapes() {
        let device = cpu_device();
        let config = VoxtralFrontEndConfig::new()
            .with_n_mels(80)
            .with_global_log_mel_max(2.0)
            .with_whisper(WhisperFrontEndConfig::new().with_range_clamp_db(6.0));
        let front: VoxtralFrontEnd = config.init(&device);
        assert_eq!(front.sample_rate(), config.sample_rate());
        assert_eq!(front.hop(), config.hop());
        assert_eq!(front.n_mels(), config.n_mels());
        assert_eq!(front.global_log_mel_max(), config.global_log_mel_max());

        let features = front.forward(&vec![0.0; 1600], &device).unwrap();
        assert_eq!(features.dims(), [1, 80, 10]);
        // Silence floors at (2 - 6 + 4) / 4 = 0.
        let max = features.abs().max().into_scalar::<f32>();
        assert!(max < 1e-6, "{max}");
        let silence = front.silence(3, &device);
        assert_eq!(silence.dims(), [1, 80, 3]);
        assert!(silence.abs().max().into_scalar::<f32>() < 1e-6);
    }
}
