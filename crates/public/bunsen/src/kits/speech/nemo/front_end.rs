//! `NeMo`'s mel front end, `AudioToMelSpectrogramPreprocessor`, at inference.
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
    kits::speech::asr_driver::StreamingFeatures,
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

/// Common meta for [`NemoMelFrontEnd`] and [`NemoMelFrontEndConfig`].
pub trait NemoMelFrontEndMeta {
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

/// Config for [`NemoMelFrontEnd`]: `NeMo`'s `AudioToMelSpectrogramPreprocessor`
/// at inference (no dither), with the knobs `NeMo`'s speech models set
/// differently.
///
/// The defaults are `parakeet-tdt-0.6b-v3`'s, which `transformers`'
/// `ParakeetFeatureExtractor` reproduces:
/// [`parakeet_tdt`](Self::parakeet_tdt).
/// [`nemotron_diarization`](Self::nemotron_diarization) is
/// `Nemotron-3-Diarization`'s and the streaming Sortformers'. `NeMo`'s
/// `pad_to` pads only past an item's valid frames, which this front end
/// zeroes and reports as padding, so it has no knob.
///
/// Implements [`NemoMelFrontEndMeta`].
#[derive(Config, Debug, PartialEq)]
pub struct NemoMelFrontEndConfig {
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

    /// How each item's log-mels are normalized.
    #[config(default = "FeatureNormalization::PerFeature")]
    pub normalize: FeatureNormalization,

    /// Round the analysis window and the mel filterbank to `bf16`, as a
    /// checkpoint that stores them in `bf16` computes with them.
    #[config(default = "false")]
    pub bf16_tables: bool,
}

/// How a [`NemoMelFrontEnd`] normalizes an item's log-mels: `NeMo`'s
/// `normalize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FeatureNormalization {
    /// Each mel channel to zero mean and unit (unbiased) standard deviation
    /// over the item's valid frames (`per_feature`). The feature then depends
    /// on the whole item.
    PerFeature,

    /// None (`NA`): each frame depends only on its own samples, so the front
    /// end streams exactly.
    None,
}

impl FeatureNormalization {
    /// Whether a frame depends only on its own window of samples.
    pub fn is_frame_local(self) -> bool {
        matches!(self, Self::None)
    }
}

impl NemoMelFrontEndMeta for NemoMelFrontEndConfig {
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

impl NemoMelFrontEndConfig {
    /// NVIDIA's `parakeet-tdt-0.6b-v3`: 128 mels, per-feature normalization.
    /// The defaults.
    pub fn parakeet_tdt() -> Self {
        Self::new()
    }

    /// NVIDIA's `Nemotron-3-Diarization`: 128 mels, no normalization, and the
    /// window and filterbank rounded to `bf16`, as its checkpoint stores them
    /// and `NeMo` computes with them.
    pub fn nemotron_diarization() -> Self {
        Self::new()
            .with_normalize(FeatureNormalization::None)
            .with_bf16_tables(true)
    }

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

/// Each value rounded to the nearest `bf16` (ties to even), on the host:
/// `bf16` itself need not exist on the device.
fn round_to_bf16<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let device = x.device();
    let shape = x.dims();
    let values: Vec<f32> = x
        .into_data()
        .try_into_vec_as::<f32>()
        .expect("read back f32 values")
        .into_iter()
        .map(bf16_round)
        .collect();
    Tensor::from_data(TensorData::new(values, shape), &device)
}

/// `x` rounded to the nearest `bf16`, ties to even.
fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let round = 0x7FFF + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round) & 0xFFFF_0000)
}

impl ModuleInit<NemoMelFrontEnd> for NemoMelFrontEndConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<NemoMelFrontEnd> {
        let mut converter: PerceptiveAudioConverter = self.converter_options().try_init(device)?;
        if self.bf16_tables {
            converter.window = round_to_bf16(converter.window);
            converter.mel_t = round_to_bf16(converter.mel_t);
        }
        Ok(NemoMelFrontEnd {
            config: self.clone(),
            converter,
        })
    }
}

/// `NeMo`'s mel front end: waveforms to log-mel features.
///
/// Pre-emphasis; a centred short-time power spectrum (zero padding, a
/// 25 ms symmetric Hann window, a 10 ms hop); Slaney mels; `ln(x + 2^-24)`;
/// then, under [`FeatureNormalization::PerFeature`], each mel channel
/// normalized to zero mean and unit (unbiased) standard deviation over the
/// item's valid frames; padding frames zeroed.
///
/// Per-feature normalization makes this an offline transform: a feature
/// depends on the whole item. Without it each frame depends only on its own
/// samples, and [`NemoMelFeatures`] streams it exactly.
///
/// Built by [`NemoMelFrontEndConfig`].
/// Implements [`NemoMelFrontEndMeta`].
#[derive(Module, Debug)]
pub struct NemoMelFrontEnd {
    /// The config this was built from.
    #[module(skip)]
    pub config: NemoMelFrontEndConfig,

    /// The spectrum and mel stages.
    pub converter: PerceptiveAudioConverter,
}

impl NemoMelFrontEndMeta for NemoMelFrontEnd {
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

impl NemoMelFrontEnd {
    /// Extracts features.
    ///
    /// # Arguments
    /// * `waveforms` - `[batch, samples]`, at
    ///   [`sample_rate`](NemoMelFrontEndMeta::sample_rate).
    /// * `lengths` - each item's valid sample count; `None` means every item
    ///   fills `samples`.
    ///
    /// # Returns
    /// `[batch, n_mels, samples / hop + 1]` features, and each item's valid
    /// frame count, [`frame_len`](NemoMelFrontEndMeta::frame_len). Frames
    /// past it are zero.
    pub fn forward(
        &self,
        waveforms: Tensor<2>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        let [batch, samples] = crate::contracts::unpack_shape_contract!(
            ["batch", "samples"],
            &waveforms.dims(),
            &["batch", "samples"],
        );
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

        let [_, frames, _] = logmel.dims();
        let frame_lengths: Vec<usize> = lengths.iter().map(|&n| self.frame_len(n)).collect();
        let valid = length_mask(&frame_lengths, frames, &device)
            .float()
            .reshape([batch, frames, 1]);
        let features = match self.config.normalize {
            FeatureNormalization::None => logmel * valid,
            FeatureNormalization::PerFeature => {
                // Over each item's valid frames.
                let counts: Vec<f32> = frame_lengths.iter().map(|&n| n as f32).collect();
                let counts =
                    Tensor::<1>::from_floats(counts.as_slice(), &device).reshape([batch, 1, 1]);
                let mean = (logmel.clone() * valid.clone()).sum_dim(1) / counts.clone();
                let centered = (logmel - mean) * valid.clone();
                let variance =
                    centered.clone().powi_scalar(2).sum_dim(1) / (counts - 1.0).clamp_min(1.0);
                let std = variance.sqrt().add_scalar(self.config.norm_epsilon);
                (centered / std) * valid
            }
        };
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

/// A [`NemoMelFrontEnd`], streamed: pre-emphasis carried across pushes, the
/// frames centred with half a window of zeros at each end, and
/// `ln(mel + 2^-24)` per frame, never normalized.
///
/// Every frame is the one-shot front end's before normalization, the last
/// included: its end padding is zeros, as the one-shot front end's is. So
/// under [`FeatureNormalization::None`] the frames are the front end's
/// exactly, however the samples are split; under
/// [`PerFeature`](FeatureNormalization::PerFeature) normalization is the
/// caller's, as a stream driver scales a window by its
/// [`MomentsLevel`](crate::kits::speech::asr_driver::MomentsLevel).
///
/// `[1, frames, n_mels]` per push.
#[derive(Clone, Debug)]
pub struct NemoMelFeatures {
    front_end: NemoMelFrontEnd,

    /// The last sample pushed, for pre-emphasis; `None` before the first.
    last: Option<f32>,

    /// Pre-emphasized, centred samples not yet consumed by a frame.
    pending: Vec<f32>,

    /// Set once the input ends.
    finished: bool,

    device: Device,
}

impl NemoMelFeatures {
    /// A stream through `front_end`, whose frames are built on `device`.
    pub fn new(
        front_end: NemoMelFrontEnd,
        device: Device,
    ) -> Self {
        let half = front_end.config.win_length / 2;
        Self {
            front_end,
            last: None,
            pending: vec![0.0; half],
            finished: false,
            device,
        }
    }

    /// The frames whose windows `pending` fills, consumed.
    fn frames(&mut self) -> Option<Tensor<3>> {
        let converter = &self.front_end.converter;
        let (n_fft, hop) = (converter.n_fft(), converter.hop());
        if self.pending.len() < n_fft {
            return None;
        }
        let count = (self.pending.len() - n_fft) / hop + 1;
        let span = (count - 1) * hop + n_fft;
        let x = Tensor::<2>::from_data(
            TensorData::new(self.pending[..span].to_vec(), [1, span]),
            &self.device,
        );
        self.pending.drain(..count * hop);
        let mel = converter.mel(converter.spectrum(converter.frame(x)));
        let logmel = mel.add_scalar(self.front_end.config.log_guard).log();

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [1, "frames", "n_mels"],
            &logmel.dims(),
            &[("frames", count), ("n_mels", self.front_end.n_mels())],
        );

        Some(logmel)
    }
}

impl StreamingFeatures for NemoMelFeatures {
    fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Option<Tensor<3>>> {
        let a = self.front_end.config.preemphasis as f32;
        for &x in samples {
            self.pending.push(match self.last {
                Some(last) => x - a * last,
                None => x,
            });
            self.last = Some(x);
        }
        Ok(self.frames())
    }

    fn finish(&mut self) -> BunsenResult<Option<Tensor<3>>> {
        if self.finished || self.last.is_none() {
            self.finished = true;
            return Ok(None);
        }
        self.finished = true;
        let half = self.front_end.config.win_length / 2;
        self.pending.extend(std::iter::repeat_n(0.0, half));
        Ok(self.frames())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = NemoMelFrontEndConfig::new()
            .with_hop(80)
            .with_n_mels(40)
            .with_sample_rate(8000);
        let module: NemoMelFrontEnd = config.init(&cpu_device());
        assert_eq!(module.sample_rate(), config.sample_rate());
        assert_eq!(module.hop(), config.hop());
        assert_eq!(module.n_mels(), config.n_mels());
        assert_eq!(module.frame_len(1000), 12);
    }

    #[test]
    fn test_presets() {
        let config = NemoMelFrontEndConfig::parakeet_tdt();
        assert_eq!(config, NemoMelFrontEndConfig::new());
        assert_eq!(config.normalize, FeatureNormalization::PerFeature);
        let config = NemoMelFrontEndConfig::nemotron_diarization();
        assert!(config.normalize.is_frame_local());
        assert!(config.bf16_tables);
        assert_eq!(config.n_mels, 128);
    }

    /// Rounding to `bf16` keeps 8 significant bits, ties to even, and leaves
    /// a value that is already `bf16` alone.
    #[test]
    fn test_bf16_round() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(0.0), 0.0);
        // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: to even, 1.
        assert_eq!(bf16_round(1.0 + 2f32.powi(-8)), 1.0);
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: to even.
        assert_eq!(bf16_round(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2f32.powi(-6));
        assert_eq!(bf16_round(1.0 + 2f32.powi(-7)), 1.0 + 2f32.powi(-7));
        let x = 0.123_456_7f32;
        assert_eq!(bf16_round(bf16_round(x)), bf16_round(x));
        assert!((bf16_round(x) - x).abs() <= x * 2f32.powi(-8));
    }

    /// With `bf16_tables`, the window and the filterbank are the `bf16`
    /// roundings of the `f32` ones.
    #[test]
    fn test_bf16_tables() {
        let device = cpu_device();
        let plain: NemoMelFrontEnd = NemoMelFrontEndConfig::new().init(&device);
        let rounded: NemoMelFrontEnd = NemoMelFrontEndConfig::new()
            .with_bf16_tables(true)
            .init(&device);
        for (a, b) in [
            (
                plain.converter.window.clone().into_data(),
                rounded.converter.window.clone().into_data(),
            ),
            (
                plain.converter.mel_t.clone().into_data(),
                rounded.converter.mel_t.clone().into_data(),
            ),
        ] {
            let a = a.try_into_vec_as::<f32>().unwrap();
            let b = b.try_into_vec_as::<f32>().unwrap();
            assert!(a.iter().zip(&b).all(|(&x, &y)| bf16_round(x) == y));
            assert!(a.iter().zip(&b).any(|(x, y)| x != y));
        }
    }

    /// Without normalization the streamed frames are the one-shot front
    /// end's exactly, however the samples are split.
    #[test]
    fn test_unnormalized_front_end_streams_exactly() {
        let device = cpu_device();
        let front_end: NemoMelFrontEnd =
            NemoMelFrontEndConfig::nemotron_diarization().init(&device);
        let samples: Vec<f32> = (0..4321)
            .map(|n| {
                ((n as f32) * 0.05).sin() * 0.3 + (((n * 7919) % 1009) as f32 / 504.5 - 1.0) * 0.05
            })
            .collect();
        let (whole, lengths) = front_end.forward(
            Tensor::from_data(
                TensorData::new(samples.clone(), [1, samples.len()]),
                &device,
            ),
            None,
        );
        let whole = whole.swap_dims(1, 2);

        for piece in [1, 77, 160, 999] {
            let mut stream = NemoMelFeatures::new(front_end.clone(), device.clone());
            let mut frames = Vec::new();
            for chunk in samples.chunks(piece) {
                frames.extend(stream.push(chunk).unwrap());
            }
            frames.extend(stream.finish().unwrap());
            let streamed = Tensor::cat(frames, 1);
            assert_eq!(streamed.dims(), whole.dims(), "{piece}");
            // The one-shot front end zeroes the frame centred past the end as
            // padding; the stream leaves the caller to drop it.
            let valid = lengths[0];
            streamed
                .slice_dim(1, 0..valid)
                .into_data()
                .assert_approx_eq::<f32>(
                    &whole.clone().slice_dim(1, 0..valid).into_data(),
                    burn::tensor::Tolerance::absolute(1e-4),
                );
        }
        assert_eq!(lengths, [samples.len() / 160]);
    }

    #[test]
    fn test_shapes() {
        let module: NemoMelFrontEnd = NemoMelFrontEndConfig::new().init(&cpu_device());
        let (features, lengths) = module.forward(
            Tensor::zeros([2, 16000], &cpu_device()),
            Some(&[16000, 8000]),
        );
        assert_eq!(features.dims(), [2, 128, 101]);
        assert_eq!(lengths, [100, 50]);
    }

    /// `testdata/nemo/front_end.safetensors`
    /// (`tools/gen_nemo_front_end_fixtures.py`).
    #[cfg(feature = "store_safetensors")]
    fn nemotron_fixture() -> std::collections::BTreeMap<String, burn::tensor::TensorData> {
        use burn::store::{
            ModuleStore,
            SafetensorsStore,
            bridge,
        };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/nemo/front_end.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
        let all = store.get_all_tensors().expect("read the fixture").clone();
        all.iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    bridge::to_data(v).expect("read a fixture tensor"),
                )
            })
            .collect()
    }

    /// `Nemotron-3-Diarization`'s front end, its tables in `f32`, is what
    /// `transformers`' `NemotronAsrStreamingFeatureExtractor` computes: two
    /// waveforms of different lengths, padded as one batch.
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_nemotron_matches_transformers() {
        use burn::tensor::Tolerance;

        use crate::support::testing::performance_device;

        let fixture = nemotron_fixture();
        let device = performance_device();
        let waveforms = Tensor::<2>::from_data(fixture["waveforms"].clone(), &device);
        let lengths: Vec<usize> = fixture["lengths"]
            .clone()
            .convert::<f32>()
            .iter::<f32>()
            .map(|n| n as usize)
            .collect();
        let expected = Tensor::<3>::from_data(fixture["features"].clone(), &device);
        let mask = Tensor::<2>::from_data(fixture["mask"].clone(), &device);

        let module: NemoMelFrontEnd = NemoMelFrontEndConfig::nemotron_diarization()
            .with_bf16_tables(false)
            .init(&device);
        let (features, frame_lengths) = module.forward(waveforms, Some(&lengths));
        let expected_lengths: Vec<usize> = mask
            .sum_dim(1)
            .into_data()
            .iter::<f32>()
            .map(|n| n as usize)
            .collect();
        assert_eq!(frame_lengths, expected_lengths);

        // Log-mels, unnormalized: the low-energy bins are where float32
        // spectra part; measured on wgpu, by 1.6e-4 on a bin of -16.
        features
            .swap_dims(1, 2)
            .to_data_as::<f32>()
            .assert_approx_eq::<f32>(
                &expected.to_data_as::<f32>(),
                Tolerance::rel_abs(1e-4, 2e-3),
            );
    }

    /// The `bf16` tables are the ones `Nemotron-3-Diarization`'s checkpoint
    /// stores, bit for bit: the window, then zeros to the FFT size, and the
    /// filterbank, transposed.
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_bf16_tables_are_the_checkpoints() {
        let fixture = nemotron_fixture();
        let module: NemoMelFrontEnd =
            NemoMelFrontEndConfig::nemotron_diarization().init(&cpu_device());

        let window: Vec<f32> = module
            .converter
            .window
            .clone()
            .into_data()
            .try_into_vec_as()
            .unwrap();
        let stored: Vec<f32> = fixture["window"].clone().try_into_vec_as().unwrap();
        assert_eq!(&window[..stored.len()], stored.as_slice());
        assert!(window[stored.len()..].iter().all(|&w| w == 0.0));

        let mel_t: Vec<f32> = module
            .converter
            .mel_t
            .clone()
            .swap_dims(0, 1)
            .into_data()
            .try_into_vec_as()
            .unwrap();
        let stored: Vec<f32> = fixture["filterbank"].clone().try_into_vec_as().unwrap();
        assert_eq!(mel_t.len(), stored.len());
        let differ = mel_t.iter().zip(&stored).filter(|(a, b)| a != b).count();
        assert_eq!(
            differ,
            0,
            "{differ} of {} filterbank values differ",
            stored.len()
        );
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

        let module: NemoMelFrontEnd = NemoMelFrontEndConfig::new().init(&device);
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
