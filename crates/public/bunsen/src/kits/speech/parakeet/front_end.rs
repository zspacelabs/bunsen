//! The Parakeet feature extractor.
use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::{
        signal::{
            StftWindowConfig,
            perceptive_audio::{
                PaddingMode,
                PerceptiveAudioConverter,
                PerceptiveAudioConverterMeta,
                PerceptiveAudioConverterOptions,
            },
        },
        transformers::attention::length_mask,
    },
};

/// Common meta for [`ParakeetFrontEnd`] and [`ParakeetFrontEndConfig`].
pub trait ParakeetFrontEndMeta {
    /// The sample rate the front end expects, in Hz.
    fn sample_rate(&self) -> usize;

    /// The hop between frames, in samples.
    fn hop(&self) -> usize;

    /// The mel channel count.
    fn n_mels(&self) -> usize;

    /// The frames a `samples`-long signal yields: one per hop, the last
    /// partial hop dropped.
    ///
    /// The extractor computes one frame more, centred past the end, and
    /// `transformers` and `NeMo` both count it as padding.
    fn frame_len(
        &self,
        samples: usize,
    ) -> usize {
        samples / self.hop()
    }
}

/// Config for [`ParakeetFrontEnd`]: `NeMo`'s
/// `AudioToMelSpectrogramPreprocessor` as `parakeet-tdt-0.6b-v3` configures it,
/// which `transformers`' `ParakeetFeatureExtractor` reproduces.
///
/// Implements [`ParakeetFrontEndMeta`].
#[derive(Config, Debug)]
pub struct ParakeetFrontEndConfig {
    /// The sample rate the front end expects, in Hz.
    #[config(default = "16000")]
    pub sample_rate: usize,

    /// The analysis window length, in samples (25 ms).
    #[config(default = "400")]
    pub win_length: usize,

    /// The hop between frames, in samples (10 ms).
    #[config(default = "160")]
    pub hop: usize,

    /// The mel channel count.
    #[config(default = "128")]
    pub n_mels: usize,

    /// The pre-emphasis coefficient `a` of `y[n] = x[n] - a * x[n - 1]`.
    #[config(default = "0.97")]
    pub preemphasis: f64,

    /// Added to every mel energy before the natural log: `2^-24`.
    #[config(default = "5.960464477539063e-8")]
    pub log_guard: f64,

    /// Added to each feature's standard deviation before dividing by it.
    #[config(default = "1e-5")]
    pub norm_epsilon: f64,
}

impl ParakeetFrontEndMeta for ParakeetFrontEndConfig {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn hop(&self) -> usize {
        self.hop
    }

    fn n_mels(&self) -> usize {
        self.n_mels
    }
}

impl ParakeetFrontEndConfig {
    /// The mel converter options this front end's spectrum stages use: a
    /// symmetric Hann window zero-padded to a power of two (400 to 512), a
    /// power spectrum, and Slaney mels over `[0, nyquist]`.
    ///
    /// Zero-padding the windowed frame at its end, rather than centring the
    /// window in the FFT as `torch.stft` does, circularly shifts the frame,
    /// which leaves the power spectrum unchanged.
    pub fn converter_options(&self) -> PerceptiveAudioConverterOptions {
        PerceptiveAudioConverterOptions::new()
            .with_sample_rate(self.sample_rate)
            .with_n_fft(self.win_length)
            .with_window(StftWindowConfig::Hann { periodic: false })
            .with_pad_to_pow2(true)
            .with_n_mels(self.n_mels)
            .with_hop(self.hop)
            .with_start_padding(PaddingMode::Zero)
            .with_end_padding(PaddingMode::Zero)
    }
}

impl ModuleInit<ParakeetFrontEnd> for ParakeetFrontEndConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ParakeetFrontEnd> {
        Ok(ParakeetFrontEnd {
            config: self.clone(),
            converter: self.converter_options().try_init(device)?,
        })
    }
}

/// Parakeet's feature extractor: waveforms to normalized log-mel features.
///
/// Pre-emphasis; a centred short-time power spectrum (zero padding, a
/// 25 ms symmetric Hann window, a 10 ms hop); 128 Slaney mels; `ln(x + 2^-24)`;
/// then each mel channel normalized to zero mean and unit (unbiased) standard
/// deviation over the item's valid frames, and padding frames zeroed.
///
/// The per-item normalization makes this an offline transform: a feature
/// depends on the whole item.
///
/// Built by [`ParakeetFrontEndConfig`].
/// Implements [`ParakeetFrontEndMeta`].
#[derive(Module, Debug)]
pub struct ParakeetFrontEnd {
    /// The config this was built from.
    #[module(skip)]
    pub config: ParakeetFrontEndConfig,

    /// The spectrum and mel stages.
    pub converter: PerceptiveAudioConverter,
}

impl ParakeetFrontEndMeta for ParakeetFrontEnd {
    fn sample_rate(&self) -> usize {
        self.converter.sample_rate()
    }

    fn hop(&self) -> usize {
        self.converter.hop()
    }

    fn n_mels(&self) -> usize {
        self.converter.n_mels()
    }
}

impl ParakeetFrontEnd {
    /// Extracts features.
    ///
    /// # Arguments
    /// * `waveforms` - `[batch, samples]`, at
    ///   [`sample_rate`](ParakeetFrontEndMeta::sample_rate).
    /// * `lengths` - each item's valid sample count; `None` means every item
    ///   fills `samples`.
    ///
    /// # Returns
    /// `[batch, n_mels, samples / hop + 1]` features, and each item's valid
    /// frame count, [`frame_len`](ParakeetFrontEndMeta::frame_len). Frames
    /// past it are zero.
    pub fn forward(
        &self,
        waveforms: Tensor<2>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        let [batch, samples] = waveforms.dims();
        let device = waveforms.device();
        let lengths: Vec<usize> = match lengths {
            Some(lengths) => {
                assert_eq!(lengths.len(), batch, "one length per batch item");
                lengths.to_vec()
            }
            None => vec![samples; batch],
        };

        // Pre-emphasis, then silence past each item's end.
        let a = self.config.preemphasis;
        let x = if a != 0.0 && samples > 1 {
            let head = waveforms.clone().slice_dim(1, 0..1);
            let rest = waveforms.clone().slice_dim(1, 1..)
                - waveforms.slice_dim(1, 0..samples - 1).mul_scalar(a);
            Tensor::cat(vec![head, rest], 1)
        } else {
            waveforms
        };
        let x = x * length_mask(&lengths, samples, &device).float();

        // Centre the frames: half a window of zeros on each side.
        let half = self.config.win_length / 2;
        let x = Tensor::cat(
            vec![
                Tensor::zeros([batch, half], &device),
                x,
                Tensor::zeros([batch, half], &device),
            ],
            1,
        );
        let mel = self
            .converter
            .mel(self.converter.spectrum(self.converter.frame(x)));
        let logmel = mel.add_scalar(self.config.log_guard).log();

        // Per-feature normalization over each item's valid frames.
        let [_, frames, _] = logmel.dims();
        let frame_lengths: Vec<usize> = lengths.iter().map(|&n| self.frame_len(n)).collect();
        let valid = length_mask(&frame_lengths, frames, &device)
            .float()
            .reshape([batch, frames, 1]);
        let counts: Vec<f32> = frame_lengths.iter().map(|&n| n as f32).collect();
        let counts = Tensor::<1>::from_floats(counts.as_slice(), &device).reshape([batch, 1, 1]);
        let mean = (logmel.clone() * valid.clone()).sum_dim(1) / counts.clone();
        let centered = (logmel - mean) * valid.clone();
        let variance = centered.clone().powi_scalar(2).sum_dim(1) / (counts - 1.0).clamp_min(1.0);
        let std = variance.sqrt().add_scalar(self.config.norm_epsilon);
        let features = (centered / std) * valid;
        let features = features.swap_dims(1, 2);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "n_mels", "frames"],
            &features.dims(),
            &[
                ("batch", batch),
                ("n_mels", self.n_mels()),
                ("frames", samples / self.hop() + 1),
            ],
        );

        (features, frame_lengths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = ParakeetFrontEndConfig::new()
            .with_hop(80)
            .with_n_mels(40)
            .with_sample_rate(8000);
        let module: ParakeetFrontEnd = config.init(&cpu_device());
        assert_eq!(module.sample_rate(), config.sample_rate());
        assert_eq!(module.hop(), config.hop());
        assert_eq!(module.n_mels(), config.n_mels());
        assert_eq!(module.frame_len(1000), 12);
    }

    #[test]
    fn test_shapes() {
        let module: ParakeetFrontEnd = ParakeetFrontEndConfig::new().init(&cpu_device());
        let (features, lengths) = module.forward(
            Tensor::zeros([2, 16000], &cpu_device()),
            Some(&[16000, 8000]),
        );
        assert_eq!(features.dims(), [2, 128, 101]);
        assert_eq!(lengths, [100, 50]);
    }

    /// The fixture `transformers`' `ParakeetFeatureExtractor` wrote: two
    /// waveforms of different lengths, padded as one batch.
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_matches_transformers() {
        use burn::{
            store::{
                ModuleStore,
                SafetensorsStore,
                bridge,
            },
            tensor::Tolerance,
        };

        use crate::support::testing::performance_device;

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parakeet/front_end.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
        let all = store.get_all_tensors().expect("read the fixture").clone();
        let device = performance_device();
        let get = |k: &str| bridge::to_data(&all[k]).expect("read a fixture tensor");

        let waveforms = Tensor::<2>::from_data(get("waveforms"), &device);
        let lengths: Vec<usize> = get("lengths")
            .convert::<f32>()
            .iter::<f32>()
            .map(|n| n as usize)
            .collect();
        let expected = Tensor::<3>::from_data(get("features"), &device);
        let mask = Tensor::<2>::from_data(get("mask"), &device);

        let module: ParakeetFrontEnd = ParakeetFrontEndConfig::new().init(&device);
        let (features, frame_lengths) = module.forward(waveforms, Some(&lengths));
        let expected_lengths: Vec<usize> = mask
            .sum_dim(1)
            .into_data()
            .iter::<f32>()
            .map(|n| n as usize)
            .collect();
        assert_eq!(frame_lengths, expected_lengths);

        // The features are standardized, so an absolute tolerance is a
        // fraction of a standard deviation.
        features
            .swap_dims(1, 2)
            .to_data_as::<f32>()
            .assert_approx_eq::<f32>(
                &expected.to_data_as::<f32>(),
                Tolerance::rel_abs(1e-3, 2e-3),
            );
    }
}
