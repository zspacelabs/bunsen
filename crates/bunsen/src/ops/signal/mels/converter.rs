//! # Waveform to log-mel conversion.
//!
//! [`MelConverterOptions`] configures the pipeline and builds a
//! [`MelConverter`]; the converter holds the precomputed constants and is
//! what a stream is driven through.
//!
//! Defaults reproduce Whisper / `librosa`: 16 kHz, 400-sample periodic Hann,
//! hop 160, 80 Slaney mels with Slaney area normalization, power spectrum,
//! `log10` over a `1e-10` floor, an 8 dB range clamp, and the `(log + 4) / 4`
//! affine tail.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Backend,
        TensorData,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::signal::{
        SamplingWindowBuilder,
        StftWindowConfig,
        mels::filterbank::{
            FilterNorm,
            MelScale,
            mel_filterbank,
        },
    },
};

/// How a chunk boundary is padded.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum PaddingMode {
    /// No padding; framing starts at the first sample.
    None,

    /// Pad with zeros.
    Zero,

    /// Mirror the signal about its edge sample, matching `librosa`'s
    /// `center=True` and `numpy`'s `"reflect"`.
    Reflect,
}

/// What the spectrum stage emits per bin.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumKind {
    /// `re² + im²`. The Whisper / `librosa` default.
    Power,

    /// `sqrt(re² + im²)`.
    Magnitude,
}

/// Which spectrum implementation to use.
///
/// One variant for now. `burn`'s `rfft` / `stft` are power-of-two only, so
/// they cannot reach the default `n_fft = 400` geometry at all; a `Stft`
/// variant is only worth adding alongside a power-of-two configuration that
/// exercises it. See `MEL_CONVERTER_PLAN.md`.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumImpl {
    /// Explicit DFT by matrix multiply against precomputed cos/sin tables.
    ///
    /// Works at any `n_fft`, and unlike `rfft` it is differentiable.
    DftMatmul,
}

/// The logarithm applied during compression.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum LogBase {
    /// `log10`. The Whisper / `librosa` default.
    Ten,

    /// Natural log, as used by Kaldi-flavoured frontends.
    E,
}

impl LogBase {
    /// Applies the logarithm.
    pub fn apply(
        &self,
        v: f64,
    ) -> f64 {
        match self {
            Self::Ten => v.log10(),
            Self::E => v.ln(),
        }
    }
}

/// A floor applied to the log-mels, relative to a reference maximum.
///
/// Values below `reference - db` are lifted to `reference - db`, which is
/// Whisper's `maximum(log_spec, log_spec.max() - 8.0)`.
#[derive(Config, Copy, Debug, PartialEq)]
pub enum RangeClamp {
    /// Reference is the maximum over the current call, per batch row.
    ///
    /// Note this is **not** a streaming homomorphism: the reference depends on
    /// how the signal was chunked, so `transform(a ++ b)` and
    /// `transform(a) ++ transform(b)` differ. Use [`Fixed`](Self::Fixed) when
    /// chunk-invariance matters.
    PerCall {
        /// The dynamic range to keep, in dB.
        db: f64,
    },

    /// Reference is a fixed value supplied by the caller.
    Fixed {
        /// The dynamic range to keep, in dB.
        db: f64,

        /// The reference maximum, in the post-log domain.
        reference: f64,
    },
}

/// The affine tail applied after compression: `(v + bias) / div`.
///
/// Whisper uses `(log_spec + 4.0) / 4.0`.
#[derive(Config, Copy, Debug, PartialEq)]
pub struct AffineCompress {
    /// Added before the division.
    pub bias: f64,

    /// Divides the shifted value; must be non-zero.
    pub div: f64,
}

impl Default for AffineCompress {
    fn default() -> Self {
        Self {
            bias: 4.0,
            div: 4.0,
        }
    }
}

/// Options for [`MelConverter`](super::MelConverter).
///
/// Defaults reproduce Whisper's `log_mel_spectrogram`. Validated by
/// [`validate`](Self::validate), which
/// [`try_init`](crate::burner::module::ModuleInit::try_init) runs before
/// building anything.
#[derive(Config, Debug)]
pub struct MelConverterOptions {
    // ---- spectral ----
    /// The signal sample rate, in Hz.
    #[config(default = "16000")]
    pub sample_rate: usize,

    /// The analysis window length, in samples.
    #[config(default = "400")]
    pub n_fft: usize,

    /// The analysis window.
    #[config(default = "StftWindowConfig::Hann { periodic: true }")]
    pub window: StftWindowConfig,

    /// Round the FFT length up to a power of two, zero-padding the windowed
    /// frame — Kaldi's `round_to_power_of_two`.
    ///
    /// This **changes the spectrum**: bin `k` moves from `k·sr/n_fft` to
    /// `k·sr/fft_len`, and the bin count changes with it. It is a distinct
    /// frontend flavour, not a way to make a non-power-of-two `n_fft` work
    /// with `burn`'s `rfft`.
    #[config(default = "false")]
    pub pad_to_pow2: bool,

    // ---- mel ----
    /// The number of mel triangles.
    #[config(default = "80")]
    pub n_mels: usize,

    /// The low edge of the mel span, in Hz.
    #[config(default = "0.0")]
    pub f_min: f64,

    /// The high edge of the mel span, in Hz; defaults to Nyquist.
    #[config(default = "None")]
    pub f_max: Option<f64>,

    /// The frequency-to-mel warping curve.
    #[config(default = "MelScale::Slaney")]
    pub mel_scale: MelScale,

    /// Triangle area normalization.
    #[config(default = "FilterNorm::Slaney")]
    pub filter_norm: FilterNorm,

    /// Whether the mel bank consumes power or magnitude.
    #[config(default = "SpectrumKind::Power")]
    pub spectrum: SpectrumKind,

    // ---- framing ----
    /// The hop between consecutive frames, in samples.
    #[config(default = "160")]
    pub hop: usize,

    /// Padding applied once, before the first frame of a stream.
    #[config(default = "PaddingMode::Reflect")]
    pub start_padding: PaddingMode,

    /// Padding applied by `finish`, after the last sample of a stream.
    #[config(default = "PaddingMode::Reflect")]
    pub end_padding: PaddingMode,

    // ---- preprocessing ----
    /// Pre-emphasis coefficient `a`, applied as `y[n] = x[n] - a·x[n-1]`.
    #[config(default = "None")]
    pub pre_emphasis: Option<f64>,

    /// Subtract each frame's mean before windowing.
    #[config(default = "false")]
    pub remove_dc: bool,

    // ---- compression ----
    /// The logarithm applied to the mel energies.
    #[config(default = "LogBase::Ten")]
    pub log_base: LogBase,

    /// Mel energies are clamped up to this before the log, so an all-zero
    /// frame yields a finite floor rather than `-inf`.
    #[config(default = "1e-10")]
    pub log_floor: f64,

    /// Optional dynamic-range floor applied after the log.
    #[config(default = "Some(RangeClamp::PerCall { db: 8.0 })")]
    pub range_clamp: Option<RangeClamp>,

    /// Optional affine tail applied last.
    #[config(default = "Some(AffineCompress { bias: 4.0, div: 4.0 })")]
    pub affine: Option<AffineCompress>,

    // ---- impl ----
    /// Which spectrum implementation to use.
    #[config(default = "SpectrumImpl::DftMatmul")]
    pub spectrum_impl: SpectrumImpl,
}

/// Common geometry for [`MelConverterOptions`] and [`MelConverter`].
///
/// Lets test and reflective code read the framing geometry uniformly from a
/// config or from a live module, without caring which it has.
///
/// Deliberately narrow. Only values needed for that uniform access live here;
/// everything else — the mel span, the compression settings — stays on
/// [`MelConverterOptions`], reachable from a module via
/// [`MelConverter::options`].
pub trait MelConverterMeta {
    /// The signal sample rate, in Hz.
    fn sample_rate(&self) -> usize;

    /// The analysis window length, in samples.
    fn n_fft(&self) -> usize;

    /// The hop between consecutive frames, in samples.
    fn hop(&self) -> usize;

    /// The number of mel triangles, and so of output channels.
    fn n_mels(&self) -> usize;

    /// Whether the FFT length is rounded up to a power of two.
    fn pad_to_pow2(&self) -> bool;

    /// Padding applied once, before the first frame of a stream.
    fn start_padding(&self) -> PaddingMode;

    /// Padding applied by `finish`, after the last sample of a stream.
    fn end_padding(&self) -> PaddingMode;

    /// The FFT length actually transformed.
    ///
    /// `n_fft`, or the next power of two when
    /// [`pad_to_pow2`](Self::pad_to_pow2) is set.
    fn fft_len(&self) -> usize {
        if self.pad_to_pow2() {
            self.n_fft().next_power_of_two()
        } else {
            self.n_fft()
        }
    }

    /// The number of `rfft` frequency bins: `fft_len / 2 + 1`.
    fn n_bins(&self) -> usize {
        self.fft_len() / 2 + 1
    }

    /// The padding prepended at the start of a stream, in samples.
    ///
    /// Zero unless [`start_padding`](Self::start_padding) pads.
    fn start_pad_len(&self) -> usize {
        match self.start_padding() {
            PaddingMode::None => 0,
            PaddingMode::Zero | PaddingMode::Reflect => self.n_fft() / 2,
        }
    }

    /// The smallest first chunk a `Reflect` start padding can mirror.
    ///
    /// Reflecting `n_fft / 2` samples reads `n_fft / 2 + 1` of them, so a
    /// shorter opening chunk cannot be padded.
    fn min_first_chunk(&self) -> usize {
        self.n_fft() / 2 + 1
    }
}

impl MelConverterMeta for MelConverterOptions {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn n_fft(&self) -> usize {
        self.n_fft
    }

    fn hop(&self) -> usize {
        self.hop
    }

    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn pad_to_pow2(&self) -> bool {
        self.pad_to_pow2
    }

    fn start_padding(&self) -> PaddingMode {
        self.start_padding
    }

    fn end_padding(&self) -> PaddingMode {
        self.end_padding
    }
}

impl Default for MelConverterOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl MelConverterOptions {
    /// The high edge of the mel span in Hz, resolving `None` to Nyquist.
    pub fn f_max_hz(&self) -> f64 {
        self.f_max.unwrap_or(self.sample_rate as f64 / 2.0)
    }

    /// Builds the host-side row-major `[n_mels, n_bins]` mel filterbank.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if any triangle covers no `rfft` bin — see
    /// [`mel_filterbank`].
    pub fn to_vec_filterbank(&self) -> BunsenResult<Vec<f64>> {
        mel_filterbank(
            self.sample_rate,
            self.fft_len(),
            self.n_mels,
            self.f_min,
            self.f_max_hz(),
            self.mel_scale,
            self.filter_norm,
        )
    }

    /// Builds the host-side mel filterbank **transposed**, row-major
    /// `[n_bins, n_mels]`.
    ///
    /// This is the orientation [`MelConverter`] stores, so a
    /// `[.., n_bins]` spectrum maps to `[.., n_mels]` by a plain matmul with
    /// no transpose in the hot path.
    ///
    /// # Errors
    ///
    /// See [`to_vec_filterbank`](Self::to_vec_filterbank).
    pub fn to_vec_filterbank_t(&self) -> BunsenResult<Vec<f64>> {
        let bank = self.to_vec_filterbank()?;
        let (n_mels, n_bins) = (self.n_mels, self.n_bins());

        let mut transposed = vec![0.0_f64; bank.len()];
        for i in 0..n_mels {
            for j in 0..n_bins {
                transposed[j * n_mels + i] = bank[i * n_bins + j];
            }
        }
        Ok(transposed)
    }

    /// Builds the host-side real-DFT tables, each row-major `[n_fft, n_bins]`.
    ///
    /// Returns `(cos, sin)` for the standard forward transform
    /// `X[k] = Σ x[n]·e^(-2πikn/fft_len)`, so the sine table carries the
    /// negative sign and `frame · cos` / `frame · sin` give `(re, im)`
    /// directly. This is the `numpy.fft.rfft` convention, matching the rest of
    /// [`ops::signal`](crate::ops::signal).
    ///
    /// Note the tables are `n_fft` rows, not `fft_len`: when `pad_to_pow2`
    /// widens the transform, the frame is conceptually zero-padded out to
    /// `fft_len`, and zeros contribute nothing to the sum. Folding the wider
    /// angle into `n_fft` rows gives the same spectrum without materializing
    /// the padding.
    ///
    /// Built in `f64` regardless of the tensor dtype — see the reduction note
    /// in the body.
    pub fn to_vec_dft_tables(&self) -> (Vec<f64>, Vec<f64>) {
        let (n_fft, n_bins, fft_len) = (self.n_fft, self.n_bins(), self.fft_len());
        let step = core::f64::consts::TAU / fft_len as f64;

        let mut cos_table = vec![0.0_f64; n_fft * n_bins];
        let mut sin_table = vec![0.0_f64; n_fft * n_bins];

        for n in 0..n_fft {
            for k in 0..n_bins {
                // Reduce `n * k` before scaling. The product reaches ~80_000
                // at the default geometry, and `(n * k) * step` there has
                // already lost the low bits that `((n * k) % fft_len) * step`
                // keeps exact.
                let theta = ((n * k) % fft_len) as f64 * step;

                cos_table[n * n_bins + k] = theta.cos();
                sin_table[n * n_bins + k] = -theta.sin();
            }
        }

        (cos_table, sin_table)
    }

    /// Validates the scalar geometry.
    ///
    /// Does **not** build the filterbank, so it cannot see an empty mel
    /// triangle; [`to_vec_filterbank`](Self::to_vec_filterbank) reports that.
    /// `try_init` runs both.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if `sample_rate`, `n_fft`, `hop`, or `n_mels`
    /// is zero, if `hop > n_fft`, if `f_min >= f_max`, if `f_max` exceeds
    /// Nyquist, or if `affine.div` is zero.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.sample_rate == 0 {
            return Err(BunsenError::Invalid(
                "MelConverter sample_rate must be non-zero".to_string(),
            ));
        }
        if self.n_fft == 0 {
            return Err(BunsenError::Invalid(
                "MelConverter n_fft must be non-zero".to_string(),
            ));
        }
        if self.hop == 0 {
            return Err(BunsenError::Invalid(
                "MelConverter hop must be non-zero".to_string(),
            ));
        }
        if self.hop > self.n_fft {
            return Err(BunsenError::Invalid(format!(
                "MelConverter hop ({}) must be <= n_fft ({})",
                self.hop, self.n_fft,
            )));
        }
        if self.n_mels == 0 {
            return Err(BunsenError::Invalid(
                "MelConverter n_mels must be non-zero".to_string(),
            ));
        }

        let nyquist = self.sample_rate as f64 / 2.0;
        let f_max = self.f_max_hz();
        if f_max > nyquist {
            return Err(BunsenError::Invalid(format!(
                "MelConverter f_max ({f_max}) must be <= Nyquist ({nyquist})",
            )));
        }
        if !(self.f_min < f_max) {
            return Err(BunsenError::Invalid(format!(
                "MelConverter f_min ({}) must be < f_max ({f_max})",
                self.f_min,
            )));
        }

        if let Some(affine) = self.affine
            && affine.div == 0.0
        {
            return Err(BunsenError::Invalid(
                "MelConverter affine.div must be non-zero".to_string(),
            ));
        }

        Ok(())
    }
}

impl<B: Backend> ModuleInit<B, MelConverter<B>> for MelConverterOptions {
    /// Initializes a [`MelConverter`] on `device`.
    ///
    /// # Errors
    ///
    /// See [`validate`](MelConverterOptions::validate) and
    /// [`to_vec_filterbank`](MelConverterOptions::to_vec_filterbank).
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<MelConverter<B>> {
        self.validate()?;

        let (n_fft, n_bins, n_mels) = (self.n_fft, self.n_bins(), self.n_mels);

        // An empty mel triangle is a configuration error, and
        // `to_vec_filterbank_t` is where it surfaces — at init, not as a
        // silently dead output channel at forward time.
        let mel_t = Tensor::from_data(
            TensorData::new(self.to_vec_filterbank_t()?, [n_bins, n_mels]),
            device,
        );

        let window = self.window.to_tensor_window(n_fft, device);

        let (cos_table, sin_table) = self.to_vec_dft_tables();
        let dft_cos = Tensor::from_data(TensorData::new(cos_table, [n_fft, n_bins]), device);
        let dft_sin = Tensor::from_data(TensorData::new(sin_table, [n_fft, n_bins]), device);

        Ok(MelConverter {
            options: self.clone(),
            window,
            mel_t,
            dft_cos,
            dft_sin,
        })
    }
}

/// Waveform to log-mel conversion module.
///
/// Built by [`MelConverterOptions`]. Holds the precomputed analysis constants;
/// like [`SlidingStft`](crate::ops::signal::SlidingStft) these are bare
/// tensors rather than `Param`s, so they ride `to_device` but are neither
/// recorded nor visited.
///
/// Implements [`MelConverterMeta`], so geometry reads the same here as on the
/// [`MelConverterOptions`] it was built from.
#[derive(Module, Debug)]
pub struct MelConverter<B: Backend> {
    /// The options this was built from.
    ///
    /// `skip`ped: it is configuration, not module state, and carries no
    /// tensors for the derive to traverse.
    #[module(skip)]
    options: MelConverterOptions,

    /// The `[n_fft]` analysis window.
    ///
    /// Applied when framing, **not** folded into the DFT tables, so every
    /// spectrum implementation sees the same windowed frames.
    pub window: Tensor<B, 1>,

    /// The `[n_bins, n_mels]` mel filterbank, stored transposed.
    ///
    /// A `[.., n_bins]` spectrum becomes `[.., n_mels]` by a plain matmul.
    pub mel_t: Tensor<B, 2>,

    /// The `[n_fft, n_bins]` real-DFT cosine table.
    pub dft_cos: Tensor<B, 2>,

    /// The `[n_fft, n_bins]` real-DFT sine table, carrying the forward
    /// transform's negative sign.
    pub dft_sin: Tensor<B, 2>,
}

impl<B: Backend> MelConverter<B> {
    /// The options this converter was built from.
    ///
    /// The escape hatch for everything [`MelConverterMeta`] deliberately
    /// leaves out — the mel span, the compression settings.
    pub fn options(&self) -> &MelConverterOptions {
        &self.options
    }
}

impl<B: Backend> MelConverterMeta for MelConverter<B> {
    fn sample_rate(&self) -> usize {
        self.options.sample_rate
    }

    fn n_fft(&self) -> usize {
        self.options.n_fft
    }

    fn hop(&self) -> usize {
        self.options.hop
    }

    fn n_mels(&self) -> usize {
        self.options.n_mels
    }

    fn pad_to_pow2(&self) -> bool {
        self.options.pad_to_pow2
    }

    fn start_padding(&self) -> PaddingMode {
        self.options.start_padding
    }

    fn end_padding(&self) -> PaddingMode {
        self.options.end_padding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::WithOkOrPanic,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    /// Reads a rank-1 or rank-2 tensor back as `f64` in row-major order.
    fn to_f64<const D: usize>(t: &Tensor<B, D>) -> Vec<f64> {
        t.clone()
            .cast(burn::tensor::DType::F64)
            .to_data()
            .to_vec()
            .unwrap()
    }

    /// Asserts every element matches, to an absolute tolerance.
    fn assert_all_close(
        actual: &[f64],
        expected: &[f64],
        tol: f64,
    ) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "element {i}: expected {e}, got {a} (tol {tol})",
            );
        }
    }

    #[test]
    fn test_converter_tensor_shapes() {
        let device = Default::default();
        let opts = MelConverterOptions::default();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        assert_eq!(conv.window.dims(), [400]);
        assert_eq!(conv.mel_t.dims(), [201, 80]);
        assert_eq!(conv.dft_cos.dims(), [400, 201]);
        assert_eq!(conv.dft_sin.dims(), [400, 201]);

        // `pad_to_pow2` widens the transform, so the bin axis grows — but the
        // DFT tables keep `n_fft` rows, because the padding is zeros that
        // contribute nothing to the sum.
        let pow2: MelConverter<B> = MelConverterOptions::default()
            .with_pad_to_pow2(true)
            .try_init(&device)
            .ok_or_panic();

        assert_eq!(pow2.window.dims(), [400]);
        assert_eq!(pow2.mel_t.dims(), [257, 80]);
        assert_eq!(pow2.dft_cos.dims(), [400, 257]);
    }

    #[test]
    fn test_converter_tensors_match_host_reference() {
        let device = Default::default();
        let opts = MelConverterOptions::default();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        // The window is the same one `StftWindowConfig` builds on the host.
        assert_all_close(
            &to_f64(&conv.window),
            &opts.window.to_vec_window(opts.n_fft),
            1e-6,
        );

        // `mel_t` is the Stage-2 bank, transposed.
        assert_all_close(
            &to_f64(&conv.mel_t),
            &opts.to_vec_filterbank_t().unwrap(),
            1e-6,
        );

        let (cos_table, sin_table) = opts.to_vec_dft_tables();
        assert_all_close(&to_f64(&conv.dft_cos), &cos_table, 1e-6);
        assert_all_close(&to_f64(&conv.dft_sin), &sin_table, 1e-6);
    }

    /// The DFT tables carry a sign and a layout convention that is easy to get
    /// backwards. Check them against `burn`'s own `rfft`.
    ///
    /// This can only run at a power-of-two `n_fft` — `rfft` rejects anything
    /// else, which is the whole reason `DftMatmul` exists — so it validates the
    /// convention at 512 and the default 400 geometry inherits it.
    #[test]
    fn test_dft_tables_match_rfft() {
        use burn::tensor::{
            Distribution,
            signal::rfft,
        };

        let device = Default::default();
        let (n_fft, n_bins, batch) = (512, 257, 3);

        let opts = MelConverterOptions::default()
            .with_n_fft(n_fft)
            .with_hop(128);
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();
        assert_eq!(conv.dft_cos.dims(), [n_fft, n_bins]);

        let frames: Tensor<B, 2> = Tensor::random([batch, n_fft], Distribution::Default, &device);

        // `X[k] = Σ x[n]·e^(-2πikn/N)`, so the tables give `(re, im)` directly.
        let re = frames.clone().matmul(conv.dft_cos.clone());
        let im = frames.clone().matmul(conv.dft_sin.clone());

        let (re_ref, im_ref) = rfft(frames, 1, Some(n_fft));

        let tol = 1e-3;
        assert_all_close(&to_f64(&re), &to_f64(&re_ref), tol);
        assert_all_close(&to_f64(&im), &to_f64(&im_ref), tol);
    }

    #[test]
    fn test_to_device_moves_every_tensor() {
        use burn::module::Module as _;

        let device = Default::default();
        let conv: MelConverter<B> = MelConverterOptions::default()
            .try_init(&device)
            .ok_or_panic();

        let before = to_f64(&conv.mel_t);

        // One device here, so this pins traversal rather than a real move: a
        // dropped derive or a stray `#[module(skip)]` on a tensor field drops
        // the entry.
        assert_eq!(conv.devices(), vec![device.clone()]);
        assert_eq!(conv.num_params(), 0);

        let moved = conv.clone().to_device(&device);
        assert_eq!(moved.devices(), vec![device]);
        assert_all_close(&to_f64(&moved.mel_t), &before, 0.0);
        assert_eq!(moved.window.dims(), conv.window.dims());
        assert_eq!(moved.options().n_mels, conv.options().n_mels);
    }

    #[test]
    fn test_defaults_are_whisper() {
        let opts = MelConverterOptions::default();

        assert_eq!(opts.sample_rate, 16000);
        assert_eq!(opts.n_fft, 400);
        assert_eq!(opts.hop, 160);
        assert_eq!(opts.n_mels, 80);
        assert_eq!(opts.window, StftWindowConfig::Hann { periodic: true });
        assert_eq!(opts.mel_scale, MelScale::Slaney);
        assert_eq!(opts.filter_norm, FilterNorm::Slaney);
        assert_eq!(opts.spectrum, SpectrumKind::Power);
        assert_eq!(opts.start_padding, PaddingMode::Reflect);
        assert_eq!(opts.end_padding, PaddingMode::Reflect);
        assert_eq!(opts.log_base, LogBase::Ten);
        assert_eq!(opts.log_floor, 1e-10);
        assert_eq!(opts.range_clamp, Some(RangeClamp::PerCall { db: 8.0 }));
        assert_eq!(
            opts.affine,
            Some(AffineCompress {
                bias: 4.0,
                div: 4.0,
            }),
        );
        assert_eq!(opts.spectrum_impl, SpectrumImpl::DftMatmul);

        opts.validate().unwrap();
    }

    #[test]
    fn test_derived_geometry() {
        let opts = MelConverterOptions::default();

        // No pow2 rounding by default: 400 stays 400, 201 bins.
        assert_eq!(opts.fft_len(), 400);
        assert_eq!(opts.n_bins(), 201);
        assert_eq!(opts.f_max_hz(), 8000.0);
        assert_eq!(opts.start_pad_len(), 200);
        assert_eq!(opts.min_first_chunk(), 201);

        // Kaldi flavour: 400 rounds to 512, giving 257 bins at different
        // centres. This is a different frontend, not the same one padded.
        let pow2 = MelConverterOptions::default().with_pad_to_pow2(true);
        assert_eq!(pow2.fft_len(), 512);
        assert_eq!(pow2.n_bins(), 257);

        // `None` start padding contributes nothing.
        let unpadded = MelConverterOptions::default().with_start_padding(PaddingMode::None);
        assert_eq!(unpadded.start_pad_len(), 0);
    }

    #[test]
    fn test_explicit_f_max_overrides_nyquist_default() {
        let opts = MelConverterOptions::default().with_f_max(Some(4000.0));
        assert_eq!(opts.f_max_hz(), 4000.0);
        opts.validate().unwrap();
    }

    #[test]
    fn test_validation_rejects_bad_geometry() {
        let base = MelConverterOptions::default;

        for bad in [
            base().with_sample_rate(0),
            base().with_n_fft(0),
            base().with_hop(0),
            // hop > n_fft
            base().with_hop(401),
            base().with_n_mels(0),
            // f_max past Nyquist
            base().with_f_max(Some(9000.0)),
            // f_min >= f_max
            base().with_f_min(8000.0),
            base().with_f_min(9000.0).with_f_max(Some(8000.0)),
            // a zero divisor would produce inf/NaN mels
            base().with_affine(Some(AffineCompress {
                bias: 4.0,
                div: 0.0,
            })),
        ] {
            assert!(
                matches!(bad.validate(), Err(BunsenError::Invalid(_))),
                "expected Invalid: {bad:?}",
            );
        }
    }

    #[test]
    fn test_filterbank_build_rejects_empty_triangles() {
        // Scalar validation passes; only building the bank catches this.
        let opts = MelConverterOptions::default()
            .with_n_fft(256)
            .with_n_mels(128);

        opts.validate().unwrap();

        let bank = opts.to_vec_filterbank();
        assert!(
            matches!(&bank, Err(BunsenError::Invalid(m)) if m.contains("covers no rfft bin")),
            "expected an empty-triangle error, got {bank:?}",
        );
    }

    #[test]
    fn test_filterbank_matches_derived_shape() {
        let opts = MelConverterOptions::default();
        let bank = opts.to_vec_filterbank().unwrap();
        assert_eq!(bank.len(), opts.n_mels * opts.n_bins());
    }

    #[test]
    fn test_config_roundtrips_through_serde() {
        let opts = MelConverterOptions::default()
            .with_n_mels(40)
            .with_f_max(Some(7600.0))
            .with_mel_scale(MelScale::Htk)
            .with_range_clamp(None);

        let json = serde_json::to_string(&opts).unwrap();
        let back: MelConverterOptions = serde_json::from_str(&json).unwrap();

        assert_eq!(back.n_mels, 40);
        assert_eq!(back.f_max, Some(7600.0));
        assert_eq!(back.mel_scale, MelScale::Htk);
        assert_eq!(back.range_clamp, None);
        assert_eq!(back.n_fft, opts.n_fft);
    }

    #[test]
    fn test_converter() {
        type B = PerformanceBackend;
        let device = Default::default();

        let options = MelConverterOptions::default();
        let _conv: MelConverter<B> = options.try_init(&device).ok_or_panic();
    }

    /// The point of [`MelConverterMeta`]: a config and the module built from
    /// it must answer identically, so test and reflective code can hold
    /// either.
    #[test]
    fn test_meta_agrees_between_config_and_module() {
        type B = PerformanceBackend;
        let device = Default::default();

        // Non-default across every meta field, so a delegation that read the
        // wrong one — or a default — shows up.
        let options = MelConverterOptions::default()
            .with_sample_rate(8000)
            .with_n_fft(300)
            .with_hop(120)
            .with_n_mels(40)
            .with_pad_to_pow2(true)
            .with_start_padding(PaddingMode::Zero)
            .with_end_padding(PaddingMode::None)
            .with_f_max(Some(4000.0));

        let conv: MelConverter<B> = options.try_init(&device).ok_or_panic();

        fn assert_same_meta(
            a: &impl MelConverterMeta,
            b: &impl MelConverterMeta,
        ) {
            assert_eq!(a.sample_rate(), b.sample_rate());
            assert_eq!(a.n_fft(), b.n_fft());
            assert_eq!(a.hop(), b.hop());
            assert_eq!(a.n_mels(), b.n_mels());
            assert_eq!(a.pad_to_pow2(), b.pad_to_pow2());
            assert_eq!(a.start_padding(), b.start_padding());
            assert_eq!(a.end_padding(), b.end_padding());
            assert_eq!(a.fft_len(), b.fft_len());
            assert_eq!(a.n_bins(), b.n_bins());
            assert_eq!(a.start_pad_len(), b.start_pad_len());
            assert_eq!(a.min_first_chunk(), b.min_first_chunk());
        }
        assert_same_meta(&options, &conv);

        // ...and the derived values are actually right, not just consistent.
        assert_eq!(conv.fft_len(), 512);
        assert_eq!(conv.n_bins(), 257);
        assert_eq!(conv.start_pad_len(), 150);
        assert_eq!(conv.min_first_chunk(), 151);

        // The full config stays reachable for everything Meta omits.
        assert_eq!(conv.options().f_max_hz(), 4000.0);
    }

    #[test]
    fn test_try_init_rejects_bad_options() {
        type B = PerformanceBackend;
        let device = Default::default();

        // Scalar geometry.
        let bad = MelConverterOptions::default().with_hop(0);
        assert!(matches!(
            ModuleInit::<B, MelConverter<B>>::try_init(&bad, &device),
            Err(BunsenError::Invalid(_)),
        ));

        // Only reachable by building the filterbank.
        let empty_rows = MelConverterOptions::default()
            .with_n_fft(256)
            .with_n_mels(128);
        assert!(matches!(
            ModuleInit::<B, MelConverter<B>>::try_init(&empty_rows, &device),
            Err(BunsenError::Invalid(_)),
        ));

        // A non-default but legal configuration still initializes.
        let ok = MelConverterOptions::default()
            .with_n_mels(40)
            .with_start_padding(PaddingMode::None);
        let _conv: MelConverter<B> = ok.try_init(&device).ok_or_panic();
    }
}
