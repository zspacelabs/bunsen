//! # ten-vad feature extraction.
//!
//! Turns `[-1, 1]` mono audio hops into the 41-dimensional feature vectors
//! [`TenVad`] consumes, reproducing the reference front end
//! (`ALGO_TRACE.md` §3.3 - §3.6) stage for stage:
//!
//! 1. scale to the reference's int16 range ([`INPUT_SCALE`]),
//! 2. [`PreEmphasisContext`] on the STFT branch only — the pitch branch keeps
//!    the raw samples,
//! 3. [`SlidingStftContext`] over a 768-sample queue, zero-padded to a
//!    1024-point FFT,
//! 4. bin power `re^2 + im^2`,
//! 5. pitch, from the [`TenVadPitchSource`], reading the raw hop and the
//!    **un-normalized** bin power,
//! 6. `1 / 32768^2` normalization, then the [`TenVadMelBank`], then `ln(x +
//!    1e-20)`,
//! 7. per-feature standardization against the reference mean / std tables.
//!
//! Two orderings here are load-bearing and easy to get backwards:
//!
//! * **Pitch runs before the power normalization**, and reads the raw,
//!   un-pre-emphasized hop.
//! * **The `1 / 32768^2` division happens before the filterbank matmul**, not
//!   after the log. The two commute algebraically but not in `f32`.
//!
//! The pieces are:
//! * [`TenVadFeatureConfig`] — the geometry.
//! * [`TenVadFeatures`] — the fixed analysis coefficients; stateless.
//! * [`TenVadFeatureContext`] — a streaming state bound to a batch; built by
//!   [`TenVadFeatureConfig::try_init_context`].
//!
//! [`TenVad`]: crate::kits::speech::ten_vad::TenVad

use burn::{
    config::Config,
    prelude::*,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    kits::speech::ten_vad::context::{
        coeff::{
            FEATURE_EPS,
            FEATURE_MEANS,
            FEATURE_STDS,
            INPUT_SCALE,
            N_FREQ,
            POWER_NORMAL,
            SAMPLE_RATE,
        },
        mel::{
            TenVadMelBank,
            TenVadMelConfig,
            TenVadMelMeta,
        },
        pitch::{
            TenVadPitchSource,
            TenVadPitchSourceInit,
            ZeroPitch,
        },
        pre_emphasis::{
            PreEmphasisConfig,
            PreEmphasisContext,
        },
    },
    ops::signal::{
        SlidingStft,
        SlidingStftConfig,
        SlidingStftContext,
        SlidingStftMeta,
    },
};

/// Common meta for [`TenVadFeatureConfig`], [`TenVadFeatures`], and
/// [`TenVadFeatureContext`].
pub trait TenVadFeatureMeta {
    /// The sample rate, in Hz, this front end expects.
    fn sample_rate(&self) -> usize;

    /// The hop size, in samples; one hop yields one feature frame.
    fn hop_size(&self) -> usize;

    /// The STFT analysis window length, in samples.
    fn win_len(&self) -> usize;

    /// The FFT size the analysis window is zero-padded to.
    fn fft_size(&self) -> usize;

    /// The number of frequency bins: `fft_size / 2 + 1`.
    fn n_bins(&self) -> usize {
        self.fft_size() / 2 + 1
    }

    /// The number of mel bands.
    fn n_mels(&self) -> usize;

    /// The feature width: [`n_mels`](Self::n_mels) log-mel bands plus one
    /// pitch bin.
    fn n_freq(&self) -> usize {
        self.n_mels() + 1
    }
}

/// Config for [`TenVadFeatures`] and [`TenVadFeatureContext`].
///
/// Defaults match the ten-vad reference front end: 16 kHz, hop 256, a
/// 768-sample periodic Hann window over a 1024-point FFT, and 40 mel bands.
///
/// Builds [`TenVadFeatures`] and [`TenVadFeatureContext`]. Implements
/// [`TenVadFeatureMeta`].
#[derive(Config, Debug)]
pub struct TenVadFeatureConfig {
    /// The sample rate, in Hz.
    #[config(default = "SAMPLE_RATE")]
    pub sample_rate: usize,

    /// The sliding STFT analyzer geometry.
    ///
    /// Its defaults are already the ten-vad analyzer
    /// (`win_len = 768`, `hop_size = 256`, `fft_size = 1024`, periodic Hann).
    #[config(default = "Default::default()")]
    pub stft: SlidingStftConfig,

    /// The mel filterbank geometry.
    #[config(default = "Default::default()")]
    pub mel: TenVadMelConfig,

    /// The pre-emphasis filter applied to the STFT branch.
    #[config(default = "Default::default()")]
    pub pre_emphasis: PreEmphasisConfig,
}

impl TenVadFeatureMeta for TenVadFeatureConfig {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn hop_size(&self) -> usize {
        self.stft.hop_size()
    }

    fn win_len(&self) -> usize {
        self.stft.win_len()
    }

    fn fft_size(&self) -> usize {
        self.stft.fft_size()
    }

    fn n_mels(&self) -> usize {
        self.mel.n_mels()
    }
}

impl TenVadFeatureConfig {
    /// Validates the front-end geometry.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the STFT or mel geometry is itself invalid,
    /// if the two disagree on the FFT size or bin count, if the mel config
    /// disagrees with the sample rate, or if the feature width exceeds the
    /// [`N_FREQ`] reference normalization entries.
    ///
    /// This deliberately does *not* pin the feature width to [`N_FREQ`]: the
    /// front end is geometry-generic, and only the driver has to match the
    /// pretrained model. Smaller geometries are useful for testing.
    pub fn validate(&self) -> BunsenResult<()> {
        self.stft.validate()?;
        self.mel.validate()?;

        if self.stft.fft_size() != self.mel.fft_size {
            return Err(BunsenError::Invalid(format!(
                "TenVadFeature stft fft_size ({}) != mel fft_size ({})",
                self.stft.fft_size(),
                self.mel.fft_size,
            )));
        }
        if self.stft.n_bins() != self.mel.n_bins() {
            return Err(BunsenError::Invalid(format!(
                "TenVadFeature stft n_bins ({}) != mel n_bins ({})",
                self.stft.n_bins(),
                self.mel.n_bins(),
            )));
        }
        if self.sample_rate != self.mel.sample_rate {
            return Err(BunsenError::Invalid(format!(
                "TenVadFeature sample_rate ({}) != mel sample_rate ({})",
                self.sample_rate, self.mel.sample_rate,
            )));
        }
        // The normalization tables are the reference's, so they bound the
        // feature width from above. The exact match against the model is
        // enforced where it belongs, in
        // [`TenVad::init_context_with`](crate::kits::speech::ten_vad::TenVad::init_context_with).
        if self.n_freq() > N_FREQ {
            return Err(BunsenError::Invalid(format!(
                "TenVadFeature n_freq ({}) exceeds the {N_FREQ} reference \
                 normalization entries",
                self.n_freq(),
            )));
        }
        Ok(())
    }

    /// Initializes the fixed analysis coefficients on `device`.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<TenVadFeatures<B>> {
        self.validate()?;

        let n_freq = self.n_freq();

        // Precompute the reciprocal so the per-frame path is a multiply.
        let stds_recip: Vec<f32> = FEATURE_STDS[..n_freq]
            .iter()
            .map(|&s| 1.0 / (s + FEATURE_EPS))
            .collect();

        Ok(TenVadFeatures {
            sample_rate: self.sample_rate,
            stft: self.stft.try_init(device)?,
            mel: self.mel.try_init(device)?,
            pre_emphasis: self.pre_emphasis,
            means: Tensor::from_data(TensorData::from(&FEATURE_MEANS[..n_freq]), device),
            stds_recip: Tensor::from_data(TensorData::new(stds_recip, [n_freq]), device),
        })
    }

    /// Initializes the fixed analysis coefficients, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> TenVadFeatures<B> {
        self.try_init(device).ok_or_panic()
    }

    /// Initializes a streaming [`TenVadFeatureContext`] over `batch_size`
    /// independent streams.
    ///
    /// # Arguments
    /// * `batch_size`: the number of independent streams; must be non-zero.
    /// * `pitch`: builds the pitch source; see [`TenVadPitchSourceInit`].
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate), plus anything the pitch source's
    /// [`try_init_source`](TenVadPitchSourceInit::try_init_source) reports.
    pub fn try_init_context<B: Backend, I: TenVadPitchSourceInit<B>>(
        &self,
        batch_size: usize,
        pitch: I,
        device: &B::Device,
    ) -> BunsenResult<TenVadFeatureContext<B, I::Source>> {
        self.try_init(device)?.try_init_state(batch_size, pitch)
    }
}

/// The fixed ten-vad feature-extraction coefficients.
///
/// Holds the STFT analysis window, the mel filterbank, and the normalization
/// tables. Stateless, so one instance can be shared by (or cheaply cloned
/// into) any number of streams. This is deliberately **not** a burn `Module`:
/// nothing here is a learnable parameter.
///
/// Built by [`TenVadFeatureConfig`]. Implements [`TenVadFeatureMeta`].
/// Streaming states are built by [`init_state`](Self::init_state).
#[derive(Debug, Clone)]
pub struct TenVadFeatures<B: Backend> {
    sample_rate: usize,
    pre_emphasis: PreEmphasisConfig,

    /// The sliding STFT analysis coefficients.
    pub stft: SlidingStft<B>,

    /// The mel filterbank.
    pub mel: TenVadMelBank<B>,

    /// The `[n_freq]` per-feature means.
    pub means: Tensor<B, 1>,

    /// The `[n_freq]` per-feature reciprocal standard deviations,
    /// `1 / (std + eps)`.
    pub stds_recip: Tensor<B, 1>,
}

impl<B: Backend> TenVadFeatureMeta for TenVadFeatures<B> {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn hop_size(&self) -> usize {
        self.stft.hop_size()
    }

    fn win_len(&self) -> usize {
        self.stft.win_len()
    }

    fn fft_size(&self) -> usize {
        self.stft.fft_size()
    }

    fn n_mels(&self) -> usize {
        self.mel.n_mels()
    }
}

impl<B: Backend> TenVadFeatures<B> {
    /// Builds a [`TenVadFeatureContext`] streaming state over these
    /// coefficients.
    ///
    /// # Arguments
    /// * `batch_size`: the number of independent streams; must be non-zero.
    /// * `pitch`: builds the pitch source; see [`TenVadPitchSourceInit`].
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if `batch_size` is zero, plus anything the
    /// pitch source's
    /// [`try_init_source`](TenVadPitchSourceInit::try_init_source) reports.
    pub fn try_init_state<I: TenVadPitchSourceInit<B>>(
        &self,
        batch_size: usize,
        pitch: I,
    ) -> BunsenResult<TenVadFeatureContext<B, I::Source>> {
        if batch_size == 0 {
            return Err(BunsenError::Invalid(
                "TenVadFeatures batch_size must be non-zero".to_string(),
            ));
        }
        let device = self.means.device();
        Ok(TenVadFeatureContext {
            stft: self.stft.init_state(batch_size),
            pre_emphasis: self.pre_emphasis.init(batch_size, &device),
            pitch: pitch.try_init_source(batch_size, &device)?,
            coef: self.clone(),
        })
    }

    /// Builds a [`TenVadFeatureContext`], panicking on error.
    ///
    /// See [`try_init_state`](Self::try_init_state).
    pub fn init_state<I: TenVadPitchSourceInit<B>>(
        &self,
        batch_size: usize,
        pitch: I,
    ) -> TenVadFeatureContext<B, I::Source> {
        self.try_init_state(batch_size, pitch).ok_or_panic()
    }
}

/// Streaming ten-vad feature-extraction state.
///
/// Binds the pre-emphasis carry, the sliding STFT queue, and one
/// [`TenVadPitchSource`] per stream to a [`TenVadFeatures`].
///
/// At stream start every buffer is zero, so — as in the reference — the first
/// couple of frames see partially zero-padded analysis windows.
///
/// Built by [`TenVadFeatures::init_state`]. Implements [`TenVadFeatureMeta`].
#[derive(Debug, Clone)]
pub struct TenVadFeatureContext<B: Backend, P: TenVadPitchSource<B> = ZeroPitch> {
    /// The fixed analysis coefficients.
    pub coef: TenVadFeatures<B>,

    /// The sliding STFT queue, over the pre-emphasized signal.
    pub stft: SlidingStftContext<B>,

    /// The pre-emphasis carry.
    pub pre_emphasis: PreEmphasisContext<B>,

    /// The pitch source, covering every stream in the batch.
    pub pitch: P,
}

impl<B: Backend, P: TenVadPitchSource<B>> TenVadFeatureMeta for TenVadFeatureContext<B, P> {
    fn sample_rate(&self) -> usize {
        self.coef.sample_rate()
    }

    fn hop_size(&self) -> usize {
        self.coef.hop_size()
    }

    fn win_len(&self) -> usize {
        self.coef.win_len()
    }

    fn fft_size(&self) -> usize {
        self.coef.fft_size()
    }

    fn n_mels(&self) -> usize {
        self.coef.n_mels()
    }
}

impl<B: Backend, P: TenVadPitchSource<B>> TenVadFeatureContext<B, P> {
    /// The batch size; each batch row is an independent stream.
    pub fn batch_size(&self) -> usize {
        self.stft.batch_size()
    }

    /// Resets every streaming buffer to the start-of-stream condition.
    pub fn reset(&mut self) {
        self.stft.reset();
        self.pre_emphasis.reset();
        self.pitch.reset();
    }

    /// Extracts the feature frame for one hop.
    ///
    /// # Arguments
    /// * `hop`: `[batch, hop_size]` mono audio in `[-1, 1]`.
    ///
    /// # Returns
    /// `[batch, n_freq]` normalized features.
    pub fn forward(
        &mut self,
        hop: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "hop_size"],
            &hop,
            &[("batch", self.batch_size()), ("hop_size", self.hop_size()),],
        );

        // The reference runs at int16 scale; bunsen takes unit-range audio.
        let raw = hop.mul_scalar(INPUT_SCALE);

        // Pre-emphasis feeds the STFT branch only; pitch sees `raw`.
        let emph = self.pre_emphasis.forward(raw.clone());

        // [batch, n_bins]
        let bin_power = self.stft.forward(emph).square().sum_dim(2).squeeze_dim(2);

        // Pitch reads the raw hop and the *un-normalized* bin power.
        // [batch, 1]
        let pitch = self.pitch.forward(raw, bin_power.clone());

        self.finish_frame(bin_power, pitch)
    }

    /// Extracts `steps` consecutive feature frames at once.
    ///
    /// Equivalent to `steps` calls of [`forward`](Self::forward), but the
    /// whole hop stream is pre-emphasized and analyzed in one pass — a single
    /// `stft` call for the sequence, and one matmul for the filterbank.
    ///
    /// # Arguments
    /// * `hops`: `[steps, batch, hop_size]` consecutive mono audio hops in
    ///   `[-1, 1]`.
    ///
    /// # Returns
    /// `[steps, batch, n_freq]` normalized features.
    pub fn forward_sequence(
        &mut self,
        hops: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        let [steps] = crate::contracts::unpack_shape_contract!(
            ["steps", "batch", "hop_size"],
            &hops,
            &["steps"],
            &[("batch", self.batch_size()), ("hop_size", self.hop_size()),],
        );
        #[cfg(not(any(test, debug_assertions)))]
        let steps = hops.dims()[0];

        let batch = self.batch_size();
        let n_freq = self.n_freq();

        let raw = hops.mul_scalar(INPUT_SCALE);
        let emph = self.pre_emphasis.forward_sequence(raw.clone());

        // [steps, batch, n_bins]
        let bin_power = self
            .stft
            .forward_sequence(emph)
            .square()
            .sum_dim(3)
            .squeeze_dim(3);

        // [steps, batch, 1]
        let pitch = self.pitch.forward_sequence(raw, bin_power.clone());

        // Fold the step axis into the row axis for the batched stages.
        let feat = self.finish_frame(
            bin_power.reshape([steps * batch, self.n_bins()]),
            pitch.reshape([steps * batch, 1]),
        );

        feat.reshape([steps, batch, n_freq])
    }

    /// Uploads host audio and extracts its feature frames.
    ///
    /// The convenience form of [`forward_sequence`](Self::forward_sequence)
    /// for callers whose audio is already host-side, which is the usual case
    /// for a file or a capture stream. The device is taken from the context,
    /// so the caller never names one.
    ///
    /// # Arguments
    /// * `audio`: one row per stream, each a whole number of hops of mono audio
    ///   in `[-1, 1]`. Every row must be the same length.
    ///
    /// # Returns
    /// `[steps, batch, n_freq]` normalized features.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the row count disagrees with the batch
    /// size, if the rows differ in length, or if a row is empty or not a whole
    /// number of hops.
    pub fn forward_audio_sequence(
        &mut self,
        audio: &[&[f32]],
    ) -> BunsenResult<Tensor<B, 3>> {
        let batch = self.batch_size();
        let hop_size = self.hop_size();

        if audio.len() != batch {
            return Err(BunsenError::Invalid(format!(
                "TenVadFeatureContext expected {batch} audio rows, got {}",
                audio.len(),
            )));
        }

        let samples = audio[0].len();
        if let Some(bad) = audio.iter().position(|row| row.len() != samples) {
            return Err(BunsenError::Invalid(format!(
                "TenVadFeatureContext audio rows must be equal length; row {bad} has {} \
                 samples, row 0 has {samples}",
                audio[bad].len(),
            )));
        }
        if samples == 0 || !samples.is_multiple_of(hop_size) {
            return Err(BunsenError::Invalid(format!(
                "TenVadFeatureContext audio length ({samples}) must be a non-zero multiple \
                 of the hop size ({hop_size})",
            )));
        }

        let steps = samples / hop_size;

        // Interleave into the `[steps, batch, hop_size]` the sequence path
        // wants; the caller's rows are per-stream contiguous.
        let mut flat = Vec::with_capacity(steps * batch * hop_size);
        for step in 0..steps {
            for row in audio {
                flat.extend_from_slice(&row[step * hop_size..(step + 1) * hop_size]);
            }
        }

        let device = self.coef.means.device();
        let hops = Tensor::from_data(TensorData::new(flat, [steps, batch, hop_size]), &device);

        Ok(self.forward_sequence(hops))
    }

    /// The shared tail of the per-frame pipeline: power normalization, mel
    /// filterbank, log, pitch concatenation, and standardization.
    ///
    /// # Arguments
    /// * `bin_power`: `[rows, n_bins]` un-normalized bin powers.
    /// * `pitch`: `[rows, 1]` pitch estimates, in Hz.
    ///
    /// # Returns
    /// `[rows, n_freq]` normalized features.
    fn finish_frame(
        &self,
        bin_power: Tensor<B, 2>,
        pitch: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // The division precedes the filterbank matmul, as in the reference;
        // the two commute algebraically but not in f32.
        let mel_power = self.coef.mel.forward(bin_power.div_scalar(POWER_NORMAL));

        // [rows, n_mels]
        let log_mel = mel_power.add_scalar(FEATURE_EPS).log();

        // [rows, n_freq]
        let feat = Tensor::cat(vec![log_mel, pitch], 1);

        (feat - self.coef.means.clone().unsqueeze::<2>())
            * self.coef.stds_recip.clone().unsqueeze::<2>()
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
        backend::BackendTypes,
    };

    use super::*;
    use crate::{
        kits::speech::ten_vad::context::{
            coeff::N_MELS,
            pitch::{
                HostPitch,
                HostPitchInit,
                TenVadPitchEstimator,
                TenVadPitchScalarSource,
            },
        },
        ops::signal::{
            SamplingWindowBuilder,
            StftWindowConfig,
        },
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;
    type F = <B as BackendTypes>::FloatElem;

    /// A small geometry whose naive-DFT reference is cheap to evaluate.
    ///
    /// The feature width (7) is below [`N_FREQ`], so it exercises the same
    /// code path with hand-checkable numbers.
    fn tiny_config() -> TenVadFeatureConfig {
        TenVadFeatureConfig::new()
            .with_stft(
                SlidingStftConfig::new()
                    .with_win_len(48)
                    .with_hop_size(16)
                    .with_fft_size(64),
            )
            .with_mel(TenVadMelConfig::new().with_n_mels(6).with_fft_size(64))
    }

    #[test]
    fn test_config_meta() {
        let cfg = TenVadFeatureConfig::new();

        assert_eq!(cfg.sample_rate(), 16000);
        assert_eq!(cfg.hop_size(), 256);
        assert_eq!(cfg.win_len(), 768);
        assert_eq!(cfg.fft_size(), 1024);
        assert_eq!(cfg.n_bins(), 513);
        assert_eq!(cfg.n_mels(), 40);
        assert_eq!(cfg.n_freq(), 41);
        assert_eq!(cfg.n_freq(), N_FREQ);

        cfg.validate().unwrap();
        tiny_config().validate().unwrap();
    }

    #[test]
    fn test_stft_window_matches_the_reference_table() {
        // The reference ships a 768-entry Hann table in `coeff.h`. We generate
        // it instead, from `StftWindowConfig::Hann { periodic: true }`; this
        // pins that the generated window really is that table.
        let window = StftWindowConfig::Hann { periodic: true }.to_vec_window(768);
        assert_eq!(window.len(), 768);

        // Anchors read straight out of the reference table.
        for (n, expected) in [
            (0usize, 0.0000000e+00f64),
            (1, 1.6733041e-05),
            (2, 6.6931045e-05),
            (96, 1.4644661e-01),
            (192, 5.0000000e-01),
            (384, 1.0000000e+00),
            (576, 5.0000000e-01),
            (700, 7.5398909e-02),
            (766, 6.6931045e-05),
            (767, 1.6733041e-05),
        ] {
            assert!(
                (window[n] - expected).abs() < 1e-7,
                "window[{n}]: {} vs {expected}",
                window[n],
            );
        }

        // A periodic window is symmetric about its midpoint, unlike the
        // symmetric variant; getting this wrong shifts every spectrum.
        for n in 1..768 {
            assert!((window[n] - window[768 - n]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        for bad in [
            // STFT and mel disagree on the FFT size.
            TenVadFeatureConfig::new().with_mel(TenVadMelConfig::new().with_fft_size(512)),
            // The mel config disagrees with the declared sample rate.
            TenVadFeatureConfig::new().with_sample_rate(8000),
            // Too many bands for the reference normalization tables.
            TenVadFeatureConfig::new().with_mel(TenVadMelConfig::new().with_n_mels(64)),
            // A structurally invalid STFT is rejected by delegation.
            TenVadFeatureConfig::new().with_stft(SlidingStftConfig::new().with_hop_size(0)),
        ] {
            assert!(
                matches!(bad.validate(), Err(BunsenError::Invalid(_))),
                "expected Invalid: {bad:?}",
            );
        }
    }

    #[test]
    fn test_init_meta_matches_config() {
        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();

        let coef: TenVadFeatures<B> = cfg.init(&device);
        assert_eq!(coef.sample_rate(), cfg.sample_rate());
        assert_eq!(coef.hop_size(), cfg.hop_size());
        assert_eq!(coef.win_len(), cfg.win_len());
        assert_eq!(coef.fft_size(), cfg.fft_size());
        assert_eq!(coef.n_bins(), cfg.n_bins());
        assert_eq!(coef.n_mels(), cfg.n_mels());
        assert_eq!(coef.n_freq(), cfg.n_freq());

        assert_eq!(coef.means.dims(), [41]);
        assert_eq!(coef.stds_recip.dims(), [41]);

        let ctx = coef.init_state(2, ZeroPitch);
        assert_eq!(ctx.batch_size(), 2);
        assert_eq!(ctx.n_freq(), 41);
        assert_eq!(ctx.stft.batch_size(), 2);
        assert_eq!(ctx.pre_emphasis.batch_size(), 2);
        assert_eq!(ctx.pitch, ZeroPitch);
    }

    #[test]
    fn test_stds_recip_is_the_normalization_divisor() {
        let device = Default::default();
        let coef: TenVadFeatures<B> = TenVadFeatureConfig::new().init(&device);

        let host: Vec<f32> = coef
            .stds_recip
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();
        for (i, &r) in host.iter().enumerate() {
            let expected = 1.0 / (FEATURE_STDS[i] + FEATURE_EPS);
            assert!((r - expected).abs() < 1e-6, "stds_recip[{i}]");
        }
    }

    #[test]
    #[should_panic(expected = "batch_size must be non-zero")]
    fn test_init_state_rejects_zero_batch() {
        let device = Default::default();
        let coef: TenVadFeatures<B> = TenVadFeatureConfig::new().init(&device);
        let _ = coef.init_state(0, ZeroPitch);
    }

    #[test]
    fn test_silence_hits_the_log_floor() {
        // Digital silence drives every mel band to zero, so the log floor
        // `ln(0 + FEATURE_EPS)` is what reaches the normalization. This pins
        // the epsilon, the standardization, and the pitch column's position
        // all at once.
        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();
        let mut ctx: TenVadFeatureContext<B, ZeroPitch> =
            cfg.try_init_context(1, ZeroPitch, &device).unwrap();

        let hop = Tensor::<B, 2>::zeros([1, cfg.hop_size()], &device);
        let feats = ctx.forward(hop);
        assert_eq!(feats.dims(), [1, 41]);

        let host: Vec<f32> = feats.to_data_as::<f32>().to_vec_as::<f32>().unwrap();

        let floor = FEATURE_EPS.ln();
        for i in 0..N_MELS {
            let expected = (floor - FEATURE_MEANS[i]) / (FEATURE_STDS[i] + FEATURE_EPS);
            assert!(
                (host[i] - expected).abs() < 1e-4,
                "silence feature {i}: {} vs {expected}",
                host[i],
            );
        }

        // Feature 40 is the pitch bin; under ZeroPitch it is the constant.
        assert!(
            (host[N_MELS] - ZeroPitch::normalized_feature()).abs() < 1e-5,
            "pitch feature: {} vs {}",
            host[N_MELS],
            ZeroPitch::normalized_feature(),
        );
    }

    #[test]
    fn test_input_gain_shifts_log_mel_by_two_log_k() {
        // The mel path is `ln(|k * X|^2 / c) = ln(|X|^2 / c) + 2 ln k`, so
        // scaling the input shifts every normalized mel feature by a known
        // amount. This pins that the log wraps the *power* and that the
        // standardization is a plain affine map applied afterwards.
        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();
        let k = 4.0f32;

        // Loud enough that the epsilon floor is irrelevant.
        let hop = Tensor::<B, 2>::random(
            [1, cfg.hop_size()],
            Distribution::Uniform(-0.5, 0.5),
            &device,
        );

        let mut base: TenVadFeatureContext<B, ZeroPitch> =
            cfg.try_init_context(1, ZeroPitch, &device).unwrap();
        let mut scaled: TenVadFeatureContext<B, ZeroPitch> =
            cfg.try_init_context(1, ZeroPitch, &device).unwrap();

        let a: Vec<f32> = base
            .forward(hop.clone())
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();
        let b: Vec<f32> = scaled
            .forward(hop.mul_scalar(k))
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        let shift = 2.0 * k.ln();
        for i in 0..N_MELS {
            let expected = shift / (FEATURE_STDS[i] + FEATURE_EPS);
            assert!(
                ((b[i] - a[i]) - expected).abs() < 1e-3,
                "band {i}: delta {} vs {expected}",
                b[i] - a[i],
            );
        }

        // The pitch bin is scale-invariant under ZeroPitch.
        assert!((b[N_MELS] - a[N_MELS]).abs() < 1e-6);
    }

    /// A fully independent host reference for the whole front end.
    ///
    /// Deliberately written from the reference ordering rather than from the
    /// tensor implementation: scale, pre-emphasize, slide the queue, window,
    /// zero-pad, naive DFT, power, normalize, filterbank, log, standardize.
    struct HostFeatures {
        cfg: TenVadFeatureConfig,
        window: Vec<f64>,
        weights: Vec<f32>,
        queue: Vec<f32>,
        prev: f32,
    }

    impl HostFeatures {
        fn new(cfg: &TenVadFeatureConfig) -> Self {
            Self {
                window: cfg.stft.window.to_vec_window(cfg.win_len()),
                weights: cfg.mel.to_vec_weights(),
                queue: vec![0.0; cfg.win_len()],
                prev: 0.0,
                cfg: cfg.clone(),
            }
        }

        fn push(
            &mut self,
            hop: &[f32],
        ) -> Vec<f32> {
            let n_bins = self.cfg.n_bins();
            let n_mels = self.cfg.n_mels();
            let fft_size = self.cfg.fft_size();
            let win_len = self.cfg.win_len();

            // 1. Scale to the reference's int16 range.
            let raw: Vec<f32> = hop.iter().map(|&x| x * INPUT_SCALE).collect();

            // 2. Pre-emphasis, on the STFT branch only.
            let mut emph = Vec::with_capacity(raw.len());
            for (i, &x) in raw.iter().enumerate() {
                let back = if i == 0 { self.prev } else { raw[i - 1] };
                emph.push(x - self.cfg.pre_emphasis.coeff * back);
            }
            self.prev = *raw.last().unwrap();

            // 3. Slide the analysis queue.
            self.queue.drain(..hop.len());
            self.queue.extend_from_slice(&emph);

            // 4. Window, zero-pad to fft_size, naive real DFT.
            let mut bin_power = Vec::with_capacity(n_bins);
            for k in 0..n_bins {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for n in 0..win_len {
                    let x = self.queue[n] as f64 * self.window[n];
                    let theta =
                        core::f64::consts::TAU * ((n * k) % fft_size) as f64 / fft_size as f64;
                    re += x * theta.cos();
                    im -= x * theta.sin();
                }
                bin_power.push((re * re + im * im) as f32);
            }

            // 5. Normalize the power, fold through the filterbank, log.
            let mut feats = Vec::with_capacity(n_mels + 1);
            for m in 0..n_mels {
                let mut acc = 0.0f32;
                for (j, &p) in bin_power.iter().enumerate() {
                    acc += (p / POWER_NORMAL) * self.weights[m * n_bins + j];
                }
                feats.push((acc + FEATURE_EPS).ln());
            }

            // 6. The pitch bin, then standardization.
            feats.push(0.0);
            for (i, f) in feats.iter_mut().enumerate() {
                *f = (*f - FEATURE_MEANS[i]) / (FEATURE_STDS[i] + FEATURE_EPS);
            }
            feats
        }
    }

    #[test]
    fn test_forward_matches_independent_host_pipeline() {
        let device = Default::default();
        let cfg = tiny_config();
        let hop_size = cfg.hop_size();
        let n_freq = cfg.n_freq();

        let mut ctx: TenVadFeatureContext<B, ZeroPitch> =
            cfg.try_init_context(1, ZeroPitch, &device).unwrap();
        let mut host = HostFeatures::new(&cfg);

        // Several hops, so the warm-up frames and the steady state are both
        // covered, and so the pre-emphasis carry is exercised.
        for step in 0..5 {
            let row: Vec<f32> = (0..hop_size)
                .map(|i| {
                    let t = (step * hop_size + i) as f32;
                    0.4 * (t * 0.11).sin() + 0.2 * (t * 0.37).cos()
                })
                .collect();

            let input =
                Tensor::<B, 2>::from_data(TensorData::new(row.clone(), [1, hop_size]), &device);
            let out = ctx.forward(input);
            assert_eq!(out.dims(), [1, n_freq]);

            let expected = host.push(&row);
            out.to_data_as::<F>().assert_approx_eq::<F>(
                &TensorData::new(expected, [1, n_freq]).convert::<F>(),
                Tolerance::permissive(),
            );
        }
    }

    #[test]
    fn test_forward_sequence_matches_stepwise() {
        let device = Default::default();

        for cfg in [tiny_config(), TenVadFeatureConfig::new()] {
            let steps = 6;
            let batch = 2;
            let hop_size = cfg.hop_size();

            let hops =
                Tensor::<B, 3>::random([steps, batch, hop_size], Distribution::Default, &device);

            let mut seq_ctx: TenVadFeatureContext<B, ZeroPitch> =
                cfg.try_init_context(batch, ZeroPitch, &device).unwrap();
            let mut step_ctx = seq_ctx.clone();

            let seq_out = seq_ctx.forward_sequence(hops.clone());
            assert_eq!(seq_out.dims(), [steps, batch, cfg.n_freq()]);

            let mut step_outs = Vec::with_capacity(steps);
            for step in 0..steps {
                step_outs.push(step_ctx.forward(hops.clone().select_dim::<2>(0, step)));
            }
            let step_out: Tensor<B, 3> = Tensor::stack(step_outs, 0);

            let tol = Tolerance::<F>::permissive();
            seq_out
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&step_out.to_data_as::<F>(), tol);

            // Residual state must agree, or the next call diverges.
            seq_ctx
                .stft
                .queue
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&step_ctx.stft.queue.to_data_as::<F>(), tol);
            seq_ctx
                .pre_emphasis
                .prev
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&step_ctx.pre_emphasis.prev.to_data_as::<F>(), tol);
        }
    }

    /// A 16 kHz pulse train at `f0`, in `[-1, 1]` — structured enough that the
    /// pitch branch actually tracks something.
    fn pulse_audio(
        f0: f32,
        samples: usize,
    ) -> Vec<f32> {
        let period = 16000.0 / f0;
        (0..samples)
            .map(|i| {
                let pos = i as f32 % period;
                0.25 * (-pos / (period * 0.08)).exp()
            })
            .collect()
    }

    #[test]
    fn test_forward_sequence_matches_stepwise_with_estimator() {
        // The ZeroPitch arm of `test_forward_sequence_matches_stepwise` cannot
        // see the pitch branch at all. This is the arm that does: the source
        // carries a per-stream recurrence, so the sequence path has to walk it
        // in exactly the stepwise order and leave identical residual state.
        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();
        let hop_size = cfg.hop_size();
        let steps = 8;
        let batch = 2;

        let mut flat = Vec::with_capacity(steps * batch * hop_size);
        let rows: Vec<Vec<f32>> = (0..batch)
            .map(|b| pulse_audio(140.0 + 30.0 * b as f32, steps * hop_size))
            .collect();
        for step in 0..steps {
            for row in &rows {
                flat.extend_from_slice(&row[step * hop_size..(step + 1) * hop_size]);
            }
        }
        let hops =
            Tensor::<B, 3>::from_data(TensorData::new(flat, [steps, batch, hop_size]), &device);

        let mut seq_ctx: TenVadFeatureContext<B, HostPitch<TenVadPitchEstimator>> = cfg
            .try_init_context(batch, TenVadPitchEstimator::new(), &device)
            .unwrap();
        let mut step_ctx = seq_ctx.clone();

        let seq_out = seq_ctx.forward_sequence(hops.clone());
        let mut step_outs = Vec::with_capacity(steps);
        for step in 0..steps {
            step_outs.push(step_ctx.forward(hops.clone().select_dim::<2>(0, step)));
        }
        let step_out: Tensor<B, 3> = Tensor::stack(step_outs, 0);

        seq_out
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_out.to_data_as::<F>(), Tolerance::permissive());

        // The pitch branch's residual state has to agree, not just its output.
        for b in 0..batch {
            let seq = &seq_ctx.pitch.sources[b];
            let step = &step_ctx.pitch.sources[b];
            assert_eq!(seq.exc_buf(), step.exc_buf(), "row {b} excitation history");
            assert_eq!(seq.path_score(), step.path_score(), "row {b} tracker state");
            assert_eq!(seq.best_period(), step.best_period(), "row {b} best period");
        }

        // Guard the guard: a tracker that never ran would compare equal too.
        assert!(
            seq_ctx.pitch.sources[0]
                .path_score()
                .iter()
                .any(|v| *v < -1e-6),
            "the tracker should have accumulated over 8 hops",
        );
    }

    #[test]
    fn test_batch_rows_are_independent_with_estimator() {
        // With `Vec<P>` per-row isolation was structural. With a batch-aware
        // source it is a property that has to be tested.
        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();
        let hop_size = cfg.hop_size();
        let steps = 6;
        let batch = 3;

        let rows: Vec<Vec<f32>> = (0..batch)
            .map(|b| pulse_audio(120.0 + 40.0 * b as f32, steps * hop_size))
            .collect();
        let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();

        let mut batched: TenVadFeatureContext<B, HostPitch<TenVadPitchEstimator>> = cfg
            .try_init_context(batch, TenVadPitchEstimator::new(), &device)
            .unwrap();
        let batched_out = batched.forward_audio_sequence(&row_refs).unwrap();

        for (b, row) in row_refs.iter().enumerate() {
            let mut solo: TenVadFeatureContext<B, HostPitch<TenVadPitchEstimator>> = cfg
                .try_init_context(1, TenVadPitchEstimator::new(), &device)
                .unwrap();
            let solo_out = solo.forward_audio_sequence(&[row]).unwrap();

            let expected = batched_out
                .clone()
                .slice_dim(1, b as isize..(b + 1) as isize);
            solo_out
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&expected.to_data_as::<F>(), Tolerance::permissive());
        }
    }

    #[test]
    fn test_forward_audio_sequence_matches_the_tensor_path() {
        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();
        let hop_size = cfg.hop_size();
        let steps = 5;

        let audio = pulse_audio(150.0, steps * hop_size);

        let mut via_audio: TenVadFeatureContext<B, HostPitch<TenVadPitchEstimator>> = cfg
            .try_init_context(1, TenVadPitchEstimator::new(), &device)
            .unwrap();
        let from_audio = via_audio.forward_audio_sequence(&[&audio]).unwrap();

        let mut via_tensor: TenVadFeatureContext<B, HostPitch<TenVadPitchEstimator>> = cfg
            .try_init_context(1, TenVadPitchEstimator::new(), &device)
            .unwrap();
        let hops =
            Tensor::<B, 1>::from_floats(audio.as_slice(), &device).reshape([steps, 1, hop_size]);
        let from_tensor = via_tensor.forward_sequence(hops);

        from_audio
            .to_data_as::<F>()
            .assert_eq(&from_tensor.to_data_as::<F>(), true);
    }

    #[test]
    fn test_forward_audio_sequence_rejects_bad_input() {
        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();
        let hop_size = cfg.hop_size();

        let mut ctx: TenVadFeatureContext<B, HostPitch<TenVadPitchEstimator>> = cfg
            .try_init_context(2, TenVadPitchEstimator::new(), &device)
            .unwrap();

        let good = vec![0.0f32; hop_size * 2];
        let short = vec![0.0f32; hop_size];
        let ragged = vec![0.0f32; hop_size + 7];

        // Wrong row count.
        assert!(ctx.forward_audio_sequence(&[&good]).is_err());
        // Rows of differing length.
        assert!(ctx.forward_audio_sequence(&[&good, &short]).is_err());
        // Not a whole number of hops.
        assert!(ctx.forward_audio_sequence(&[&ragged, &ragged]).is_err());
        // Empty.
        assert!(ctx.forward_audio_sequence(&[&[], &[]]).is_err());
        // And the good case still works.
        assert!(ctx.forward_audio_sequence(&[&good, &good]).is_ok());
    }

    #[test]
    fn test_batch_rows_are_independent() {
        let device = Default::default();
        let cfg = tiny_config();
        let hop_size = cfg.hop_size();
        let batch = 3;

        let hops = Tensor::<B, 3>::random([4, batch, hop_size], Distribution::Default, &device);

        let mut batched: TenVadFeatureContext<B, ZeroPitch> =
            cfg.try_init_context(batch, ZeroPitch, &device).unwrap();
        let batched_out = batched.forward_sequence(hops.clone());

        for b in 0..batch {
            let mut solo: TenVadFeatureContext<B, ZeroPitch> =
                cfg.try_init_context(1, ZeroPitch, &device).unwrap();
            // [steps, 1, hop_size]
            let row = hops.clone().slice_dim(1, b as isize..(b + 1) as isize);
            let solo_out = solo.forward_sequence(row);

            let expected = batched_out
                .clone()
                .slice_dim(1, b as isize..(b + 1) as isize);
            solo_out
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&expected.to_data_as::<F>(), Tolerance::permissive());
        }
    }

    #[test]
    fn test_reset() {
        let device = Default::default();
        let cfg = tiny_config();
        let hop_size = cfg.hop_size();

        let mut ctx: TenVadFeatureContext<B, ZeroPitch> =
            cfg.try_init_context(1, ZeroPitch, &device).unwrap();

        let hop = Tensor::<B, 2>::random([1, hop_size], Distribution::Default, &device);

        let first = ctx.forward(hop.clone());

        // Advance the state, then wind it back.
        ctx.forward(hop.clone());
        ctx.forward(hop.clone());
        ctx.reset();

        let again = ctx.forward(hop);
        again
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&first.to_data_as::<F>(), Tolerance::permissive());
    }

    #[test]
    fn test_custom_pitch_source_reaches_feature_40() {
        // A source that reports a fixed pitch must land, normalized, in the
        // last feature slot -- and must be driven once per stream per frame.
        #[derive(Clone, Default)]
        struct FixedPitch {
            hz: f32,
            calls: usize,
        }

        impl TenVadPitchScalarSource for FixedPitch {
            fn frame_pitch(
                &mut self,
                _raw_hop: &[f32],
                _bin_power: &[f32],
            ) -> f32 {
                self.calls += 1;
                self.hz
            }

            fn reset(&mut self) {
                self.calls = 0;
            }
        }

        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();
        let hz = 220.0f32;

        let mut ctx: TenVadFeatureContext<B, HostPitch<FixedPitch>> = cfg
            .try_init_context(1, HostPitchInit(FixedPitch { hz, calls: 0 }), &device)
            .unwrap();

        let steps = 3;
        let hops = Tensor::<B, 3>::zeros([steps, 1, cfg.hop_size()], &device);
        let out = ctx.forward_sequence(hops);

        let host: Vec<f32> = out.to_data_as::<f32>().to_vec_as::<f32>().unwrap();
        let expected = (hz - FEATURE_MEANS[N_MELS]) / (FEATURE_STDS[N_MELS] + FEATURE_EPS);

        for step in 0..steps {
            let got = host[step * cfg.n_freq() + N_MELS];
            assert!(
                (got - expected).abs() < 1e-4,
                "step {step} pitch feature: {got} vs {expected}",
            );
        }

        // One call per frame, in order.
        assert_eq!(ctx.pitch.sources[0].calls, steps);
    }
}
