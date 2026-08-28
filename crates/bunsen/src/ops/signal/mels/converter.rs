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

impl PaddingMode {
    /// The padding this mode contributes at a stream boundary, in samples.
    ///
    /// `n_fft / 2` when it pads at all, matching `librosa`'s `center=True`.
    pub fn pad_len(
        &self,
        n_fft: usize,
    ) -> usize {
        match self {
            Self::None => 0,
            Self::Zero | Self::Reflect => n_fft / 2,
        }
    }
}

/// What the spectrum stage emits per bin.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumKind {
    /// `re² + im²`. The Whisper / `librosa` default.
    Power,

    /// `sqrt(re² + im²)`.
    Magnitude,
}

impl SpectrumKind {
    /// Converts a power spectrum into this kind.
    ///
    /// Power is what the DFT stage produces, so [`Power`](Self::Power) is the
    /// identity and [`Magnitude`](Self::Magnitude) takes the square root.
    pub fn from_power<B: Backend, const D: usize>(
        &self,
        power: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self {
            Self::Power => power,
            Self::Magnitude => power.sqrt(),
        }
    }
}

/// Which spectrum implementation to use.
///
/// One variant for now. `burn`'s `rfft` / `stft` are power-of-two only, so
/// they cannot reach the default `n_fft = 400` geometry at all; a `Stft`
/// variant is only worth adding alongside a power-of-two configuration that
/// exercises it.
///
/// [`MelConverter::spectrum`] matches on this exhaustively, so adding a
/// variant does not compile until it has a path — it cannot be accepted and
/// then ignored.
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
    /// Applies the logarithm elementwise.
    ///
    /// `burn` exposes only the natural log, so base ten is `ln(x) / ln(10)`.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self {
            Self::Ten => x.log().div_scalar(core::f64::consts::LN_10),
            Self::E => x.log(),
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

impl RangeClamp {
    /// Applies the dynamic-range floor to already-logged values.
    ///
    /// Everything below `reference - db` is lifted to it, which is Whisper's
    /// `maximum(log_spec, log_spec.max() - 8.0)`.
    ///
    /// # Arguments
    /// * `x`: `[batch, frames, n_mels]` log-mels.
    pub fn apply<B: Backend>(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        match self {
            // A caller-supplied reference is a plain scalar floor, and the
            // only form that survives being chunked differently.
            Self::Fixed { db, reference } => x.clamp_min(reference - db),

            // Reduce over `[frames, n_mels]` but NOT over batch: each row is
            // an independent stream and must not see its neighbours' peaks.
            Self::PerCall { db } => {
                let dims = x.dims();
                let floor = x.clone().max_dims(&[1, 2]).sub_scalar(*db).expand(dims);
                x.max_pair(floor)
            }
        }
    }
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

impl AffineCompress {
    /// Applies `(x + bias) / div` elementwise.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        x.add_scalar(self.bias).div_scalar(self.div)
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
    ///
    /// **Not implemented yet** — [`validate`](Self::validate) rejects it
    /// rather than silently ignoring it. It needs one extra carried sample,
    /// which changes the streaming carry length.
    #[config(default = "None")]
    pub pre_emphasis: Option<f64>,

    /// Subtract each frame's mean before windowing.
    ///
    /// **Not implemented yet** — [`validate`](Self::validate) rejects it. It
    /// is per-frame, so it belongs inside framing rather than in the
    /// sample-domain preprocessing stage.
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
        self.start_padding().pad_len(self.n_fft())
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
        // Written out rather than `f_min >= f_max` so a NaN edge is rejected
        // too: `NaN >= x` is false, which would let it through to produce NaN
        // mel points.
        if self.f_min.is_nan() || f_max.is_nan() || self.f_min >= f_max {
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

        // Rejected rather than silently ignored. `t_stage_preproc` is the
        // identity today; these land there with the Kaldi-flavour work, where
        // they can be checked against `torchaudio.compliance.kaldi`.
        // Pre-emphasis also needs one extra carried sample, which changes the
        // streaming carry — not a change to make untested.
        if self.pre_emphasis.is_some() {
            return Err(BunsenError::Invalid(
                "MelConverter pre_emphasis is not implemented yet".to_string(),
            ));
        }
        if self.remove_dc {
            return Err(BunsenError::Invalid(
                "MelConverter remove_dc is not implemented yet".to_string(),
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

    /// The number of whole frames a `samples`-long signal yields.
    ///
    /// `(samples - n_fft) / hop + 1`, or zero when the signal is shorter than
    /// one window.
    pub fn frame_count(
        &self,
        samples: usize,
    ) -> usize {
        let n_fft = self.n_fft();
        if samples < n_fft {
            0
        } else {
            (samples - n_fft) / self.hop() + 1
        }
    }

    /// Frames a signal into consecutive windowed frames.
    ///
    /// Frame `f` covers `x[.., f * hop .. f * hop + n_fft]`, multiplied by the
    /// analysis window. Trailing samples that do not fill a whole frame are
    /// dropped — in a stream they belong to the carry, not to any frame.
    ///
    /// Stateless: the caller owns the padding and the carry.
    ///
    /// # Arguments
    /// * `x`: `[batch, samples]`, with `samples >= n_fft`.
    ///
    /// # Returns
    /// `[batch, frames, n_fft]` windowed frames, with
    /// `frames = (samples - n_fft) / hop + 1`.
    pub fn frame(
        &self,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        let [batch, samples] = crate::contracts::unpack_shape_contract!(
            ["batch", "samples"],
            &x,
            &["batch", "samples"],
        );
        #[cfg(any(test, debug_assertions))]
        assert!(
            samples >= self.n_fft(),
            "MelConverter samples ({samples}) must be >= n_fft ({})",
            self.n_fft(),
        );

        let (n_fft, hop) = (self.n_fft(), self.hop());
        let frames = self.frame_count(x.dims()[1]);

        // ── burn 0.21 `unfold` hazard ──────────────────────────────────────
        // Trim to exactly the span the frames cover before unfolding. On
        // `CubeCL` backends (wgpu / cuda / metal; `Flex` is correct) `unfold`
        // truncates the unfolded view's outer stride to the vectorization line
        // width `v` — the largest power of two dividing both `size` and `step`
        // — using `(len / v) * v` in place of `len`. Every batch row after the
        // first is then read `len % v` elements early. Row 0 is always right,
        // so a `batch == 1` test cannot see it.
        //
        // It fires when `size` and `step` are both even AND the uncovered tail
        // `len - ((frames - 1) * hop + n_fft)` is not a multiple of `v`. The
        // default Whisper geometry trips it at every chunk size: `n_fft` 400
        // and `hop` 160 give `v` 16, and the steady-state tail is 120.
        //
        // Trimming makes that tail zero, so `tail % v == 0` on every geometry
        // and the bug is unreachable rather than merely absent. It is also the
        // honest framing — the trailing samples belong to the carry.
        //
        // Fixed upstream in burn 0.22.0-dev. Keep the slice on the bump (it
        // states the contract), but delete this comment.
        // ──────────────────────────────────────────────────────────────────
        let covered = (frames - 1) * hop + n_fft;
        let x = x.slice_dim(1, 0..covered as isize);

        // [batch, frames, n_fft]
        let framed: Tensor<B, 3> = x.unfold(1, n_fft, hop);

        // Broadcast the window across batch and frame.
        let framed = framed.mul(self.window.clone().reshape([1, 1, n_fft]));

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "frames", "n_fft"],
            &framed,
            &[("batch", batch), ("frames", frames), ("n_fft", n_fft)],
        );

        framed
    }

    /// Transforms windowed frames into a power or magnitude spectrum.
    ///
    /// Uses the precomputed DFT tables: `re = frame · dft_cos`,
    /// `im = frame · dft_sin`, then `re² + im²` — or its square root when
    /// [`SpectrumKind::Magnitude`] is configured.
    ///
    /// # Arguments
    /// * `frames`: `[batch, frames, n_fft]` windowed frames, from
    ///   [`frame`](Self::frame).
    ///
    /// # Returns
    /// `[batch, frames, n_bins]`.
    pub fn spectrum(
        &self,
        frames: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        let [batch, n_frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "frames", "n_fft"],
            &frames,
            &["batch", "frames"],
            &[("n_fft", self.n_fft())],
        );
        #[cfg(not(any(test, debug_assertions)))]
        let [batch, n_frames] = [frames.dims()[0], frames.dims()[1]];

        let (n_fft, n_bins) = (self.n_fft(), self.n_bins());

        // Dispatched rather than assumed: this is the only thing that reads
        // `spectrum_impl`, so a new variant is a compile error here until it
        // is given a path, instead of being silently ignored.
        let power = match self.options.spectrum_impl {
            SpectrumImpl::DftMatmul => {
                // Fold batch and frame together: one
                // `[rows, n_fft] @ [n_fft, n_bins]` matmul beats broadcasting
                // the tables across a batch axis.
                let rows = batch * n_frames;
                let flat: Tensor<B, 2> = frames.reshape([rows, n_fft]);

                let re = flat.clone().matmul(self.dft_cos.clone());
                let im = flat.matmul(self.dft_sin.clone());

                // Squaring by multiply rather than `powi_scalar(2)`: same
                // result, and it is the form the fusion pass folds into the
                // matmul epilogue.
                re.clone().mul(re).add(im.clone().mul(im))
            }
        };

        let out = self.options.spectrum.from_power(power);

        let out: Tensor<B, 3> = out.reshape([batch, n_frames, n_bins]);

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "frames", "n_bins"],
            &out,
            &[("batch", batch), ("frames", n_frames), ("n_bins", n_bins)],
        );

        out
    }

    /// Maps a spectrum onto the mel scale.
    ///
    /// # Arguments
    /// * `spectrum`: `[batch, frames, n_bins]`, from
    ///   [`spectrum`](Self::spectrum).
    ///
    /// # Returns
    /// `[batch, frames, n_mels]` mel energies, before compression.
    pub fn mel(
        &self,
        spectrum: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        let [batch, n_frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "frames", "n_bins"],
            &spectrum,
            &["batch", "frames"],
            &[("n_bins", self.n_bins())],
        );
        #[cfg(not(any(test, debug_assertions)))]
        let [batch, n_frames] = [spectrum.dims()[0], spectrum.dims()[1]];

        let (n_bins, n_mels) = (self.n_bins(), self.n_mels());

        let rows = batch * n_frames;
        let flat: Tensor<B, 2> = spectrum.reshape([rows, n_bins]);

        // `mel_t` is stored `[n_bins, n_mels]`, so no transpose here.
        let out = flat.matmul(self.mel_t.clone());
        let out: Tensor<B, 3> = out.reshape([batch, n_frames, n_mels]);

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "frames", "n_mels"],
            &out,
            &[("batch", batch), ("frames", n_frames), ("n_mels", n_mels)],
        );

        out
    }

    /// Compresses mel energies into log-mels.
    ///
    /// Floors, takes the log, applies the optional dynamic-range clamp, then
    /// the optional affine tail. Shape is unchanged.
    ///
    /// The floor is applied *before* the log, so an all-zero frame yields
    /// `log(log_floor)` rather than `-inf`.
    ///
    /// # Arguments
    /// * `mels`: `[batch, frames, n_mels]` mel energies, from
    ///   [`mel`](Self::mel).
    pub fn compress(
        &self,
        mels: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "frames", "n_mels"],
            &mels,
            &[("n_mels", self.n_mels())],
        );

        let opts = &self.options;

        let x = mels.clamp_min(opts.log_floor);
        let x = opts.log_base.apply(x);

        let x = match opts.range_clamp {
            None => x,
            Some(clamp) => clamp.apply(x),
        };

        match opts.affine {
            None => x,
            Some(affine) => affine.apply(x),
        }
    }

    /// Converts a whole signal to log-mels in one call.
    ///
    /// Chains [`frame`](Self::frame), [`spectrum`](Self::spectrum),
    /// [`mel`](Self::mel) and [`compress`](Self::compress). Stateless and
    /// unpadded: no start or end padding is applied, and trailing samples that
    /// do not fill a frame are dropped. The streaming form, which owns the
    /// padding and the carry, arrives with `MelConversionContext`.
    ///
    /// Note the result is `[batch, frames, n_mels]`: frames stay on the middle
    /// axis because that is the axis streaming chunks concatenate along. A
    /// consumer wanting channels-first `[batch, n_mels, seq]` transposes with
    /// `.swap_dims(1, 2)` at that boundary.
    ///
    /// # Arguments
    /// * `x`: `[batch, samples]`, with `samples >= n_fft`.
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        self.compress(self.mel(self.spectrum(self.frame(x))))
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
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        burner::tensor::TensorDataToVecAsExt,
        errors::WithOkOrPanic,
        support::testing::{
            PerformanceBackend,
            assert_close_to_vec,
            assert_tensor_close_to_vec,
            assert_tensors_close,
        },
    };

    type B = PerformanceBackend;

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
        assert_tensor_close_to_vec(
            &conv.window,
            &opts.window.to_vec_window(opts.n_fft),
            Tolerance::absolute(1e-6),
        );

        // `mel_t` is the Stage-2 bank, transposed.
        assert_tensor_close_to_vec(
            &conv.mel_t,
            &opts.to_vec_filterbank_t().unwrap(),
            Tolerance::absolute(1e-6),
        );

        let (cos_table, sin_table) = opts.to_vec_dft_tables();
        assert_tensor_close_to_vec(&conv.dft_cos, &cos_table, Tolerance::absolute(1e-6));
        assert_tensor_close_to_vec(&conv.dft_sin, &sin_table, Tolerance::absolute(1e-6));
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
        assert_tensors_close(&re, &re_ref, Tolerance::absolute(tol));
        assert_tensors_close(&im, &im_ref, Tolerance::absolute(tol));
    }

    #[test]
    fn test_to_device_moves_every_tensor() {
        use burn::module::Module as _;

        let device = Default::default();
        let conv: MelConverter<B> = MelConverterOptions::default()
            .try_init(&device)
            .ok_or_panic();

        let before = conv.mel_t.to_data().to_vec_as::<f64>().unwrap();

        // One device here, so this pins traversal rather than a real move: a
        // dropped derive or a stray `#[module(skip)]` on a tensor field drops
        // the entry.
        assert_eq!(conv.devices(), vec![device.clone()]);
        assert_eq!(conv.num_params(), 0);

        let moved = conv.clone().to_device(&device);
        assert_eq!(moved.devices(), vec![device]);
        assert_tensor_close_to_vec(&moved.mel_t, &before, Tolerance::default());
        assert_eq!(moved.window.dims(), conv.window.dims());
        assert_eq!(moved.options().n_mels, conv.options().n_mels);
    }

    /// A deterministic sample in `[-1, 1]`, so f32 keeps ~1e-7.
    fn sample(
        row: usize,
        i: usize,
    ) -> f64 {
        (((row * 7919 + i * 104729) % 2003) as f64 / 1001.0) - 1.0
    }

    /// Host reference: frame `f` is `x[f*hop .. f*hop+n_fft]` times the window.
    fn host_frames(
        rows: &[Vec<f64>],
        n_fft: usize,
        hop: usize,
        window: &[f64],
        frames: usize,
    ) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * frames * n_fft);
        for row in rows {
            for f in 0..frames {
                for n in 0..n_fft {
                    out.push(row[f * hop + n] * window[n]);
                }
            }
        }
        out
    }

    /// Builds `[batch, samples]` from [`sample`], and the matching host rows.
    fn signal(
        device: &burn::prelude::Device<B>,
        batch: usize,
        samples: usize,
    ) -> (Tensor<B, 2>, Vec<Vec<f64>>) {
        let rows: Vec<Vec<f64>> = (0..batch)
            .map(|r| (0..samples).map(|i| sample(r, i)).collect())
            .collect();
        let flat: Vec<f64> = rows.concat();
        let t = Tensor::from_data(TensorData::new(flat, [batch, samples]), device);
        (t, rows)
    }

    #[test]
    fn test_frame_count() {
        let device = Default::default();
        let conv: MelConverter<B> = MelConverterOptions::default()
            .try_init(&device)
            .ok_or_panic();

        // Shorter than one window yields nothing.
        assert_eq!(conv.frame_count(0), 0);
        assert_eq!(conv.frame_count(399), 0);

        assert_eq!(conv.frame_count(400), 1);
        assert_eq!(conv.frame_count(559), 1);
        assert_eq!(conv.frame_count(560), 2);
        // A ragged tail does not add a frame.
        assert_eq!(conv.frame_count(520), 1);
    }

    /// Framing must agree with an explicit host loop, at `batch > 1`, for
    /// several signal lengths — including ones that leave a ragged tail.
    ///
    /// **`batch > 1` is load-bearing.** The burn 0.21 `unfold` defect this
    /// guards leaves row 0 correct and corrupts only later rows, so a
    /// `batch == 1` version of this test passes either way.
    #[test]
    fn test_frame_matches_host_reference() {
        let device = Default::default();
        let opts = MelConverterOptions::default();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();
        let window = opts.window.to_vec_window(opts.n_fft);

        let batch = 3;
        for samples in [
            400,  // exactly one window; tail 0
            560,  // two windows, hop-aligned; tail 0
            2000, // many windows, hop-aligned; tail 0
            // Ragged tails. `v` is 16 here, so `tail % v != 0` is what trips
            // the unfold defect when the covered-span slice is removed.
            520,  // tail 120 -> 120 % 16 == 8
            1000, // tail 120
            1234, // tail 34
        ] {
            let frames = conv.frame_count(samples);
            assert!(frames > 0, "samples {samples} yields no frames");

            let (x, rows) = signal(&device, batch, samples);
            let framed = conv.frame(x);

            assert_eq!(framed.dims(), [batch, frames, opts.n_fft]);
            assert_tensor_close_to_vec(
                &framed,
                &host_frames(&rows, opts.n_fft, opts.hop, &window, frames),
                Tolerance::absolute(1e-5),
            );
        }
    }

    #[test]
    fn test_frame_rows_are_independent() {
        let device = Default::default();
        let opts = MelConverterOptions::default();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        // A ragged length, so this also exercises the trimmed path.
        let samples = 1000;
        let (batched, _) = signal(&device, 3, samples);
        let together = conv
            .frame(batched.clone())
            .to_data()
            .to_vec_as::<f64>()
            .unwrap();

        let frames = conv.frame_count(samples);
        let per_row = frames * opts.n_fft;

        for row in 0..3 {
            let single: Tensor<B, 2> = batched
                .clone()
                .slice_dim(0, row as isize..(row + 1) as isize);
            let alone = conv.frame(single).to_data().to_vec_as::<f64>().unwrap();

            assert_close_to_vec(&alone, &together[row * per_row..(row + 1) * per_row], 1e-9);
        }
    }

    /// Host reference: the one-sided power spectrum of one frame.
    fn host_power(
        frame: &[f64],
        fft_len: usize,
        n_bins: usize,
    ) -> Vec<f64> {
        (0..n_bins)
            .map(|k| {
                let (mut re, mut im) = (0.0_f64, 0.0_f64);
                for (n, &x) in frame.iter().enumerate() {
                    let theta =
                        core::f64::consts::TAU * ((n * k) % fft_len) as f64 / fft_len as f64;
                    re += x * theta.cos();
                    im -= x * theta.sin();
                }
                re * re + im * im
            })
            .collect()
    }

    #[test]
    fn test_spectrum_matches_host_dft() {
        let device = Default::default();
        let (n_fft, hop, n_mels) = (64, 32, 8);
        let opts = MelConverterOptions::default()
            .with_n_fft(n_fft)
            .with_hop(hop)
            .with_n_mels(n_mels);
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        let (batch, samples) = (3, 256);
        let (x, rows) = signal(&device, batch, samples);

        let framed = conv.frame(x);
        let power = conv.spectrum(framed.clone());

        let frames = conv.frame_count(samples);
        assert_eq!(power.dims(), [batch, frames, opts.n_bins()]);

        // Re-derive from the framed tensor, so this tests the spectrum stage
        // alone rather than re-testing framing.
        let framed_host = framed.to_data().to_vec_as::<f64>().unwrap();
        let mut expected = Vec::with_capacity(batch * frames * opts.n_bins());
        for f in 0..batch * frames {
            let frame = &framed_host[f * n_fft..(f + 1) * n_fft];
            expected.extend(host_power(frame, opts.fft_len(), opts.n_bins()));
        }
        let _ = &rows;

        assert_tensor_close_to_vec(&power, &expected, Tolerance::absolute(1e-3));
    }

    /// A windowed sine at a bin centre must concentrate there.
    #[test]
    fn test_spectrum_peaks_at_bin_centre() {
        let device = Default::default();
        let opts = MelConverterOptions::default();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        let (n_fft, n_bins) = (opts.n_fft, opts.n_bins());
        let k = 40; // bin 40 -> 40 * 16000 / 400 = 1600 Hz

        let tone: Vec<f64> = (0..n_fft)
            .map(|n| (core::f64::consts::TAU * k as f64 * n as f64 / n_fft as f64).sin())
            .collect();
        let x = Tensor::from_data(TensorData::new(tone, [1, n_fft]), &device);

        let power = conv
            .spectrum(conv.frame(x))
            .to_data()
            .to_vec_as::<f64>()
            .unwrap();
        assert_eq!(power.len(), n_bins);

        let peak = power[k];
        assert!(
            power.iter().enumerate().all(|(j, &v)| j == k || v <= peak),
            "bin {k} is not the maximum",
        );

        // Hann leaks into the immediate neighbours, so check two bins out.
        for j in [k - 2, k + 2] {
            let ratio_db = 10.0 * (peak / power[j].max(f64::MIN_POSITIVE)).log10();
            assert!(
                ratio_db >= 20.0,
                "bin {j} is only {ratio_db:.1} dB below the peak",
            );
        }
    }

    #[test]
    fn test_mel_matches_host_matmul() {
        let device = Default::default();
        let (n_fft, n_mels) = (64, 8);
        let opts = MelConverterOptions::default()
            .with_n_fft(n_fft)
            .with_hop(32)
            .with_n_mels(n_mels);
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();
        let n_bins = opts.n_bins();

        let (batch, samples) = (2, 192);
        let (x, _) = signal(&device, batch, samples);

        let spectrum = conv.spectrum(conv.frame(x));
        let mels = conv.mel(spectrum.clone());

        let frames = conv.frame_count(samples);
        assert_eq!(mels.dims(), [batch, frames, n_mels]);

        // `[rows, n_bins] @ [n_bins, n_mels]`, on the host.
        let spec_host = spectrum.to_data().to_vec_as::<f64>().unwrap();
        let bank_t = opts.to_vec_filterbank_t().unwrap();
        let mut expected = Vec::with_capacity(batch * frames * n_mels);
        for r in 0..batch * frames {
            for m in 0..n_mels {
                let mut acc = 0.0;
                for b in 0..n_bins {
                    acc += spec_host[r * n_bins + b] * bank_t[b * n_mels + m];
                }
                expected.push(acc);
            }
        }

        assert_tensor_close_to_vec(&mels, &expected, Tolerance::absolute(1e-3));
    }

    #[test]
    fn test_compress_floors_zero_input() {
        let device = Default::default();
        let opts = MelConverterOptions::default();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        let zeros: Tensor<B, 3> = Tensor::zeros([2, 4, opts.n_mels], &device);
        let out = conv.compress(zeros).to_data().to_vec_as::<f64>().unwrap();

        // All-zero energy must land on a finite floor, not -inf or NaN.
        assert!(
            out.iter().all(|v| v.is_finite()),
            "compress produced a non-finite value on all-zero input",
        );

        // Everything is equal, so the per-call clamp does nothing and the
        // result is the affine of `log10(log_floor)`.
        let expected = (opts.log_floor.log10() + 4.0) / 4.0;
        assert_close_to_vec(&out, &vec![expected; out.len()], 1e-5);
    }

    /// The Whisper tail, on values chosen so every step is exact.
    ///
    /// Also pins that `PerCall` reduces per batch row: row 1's floor is set by
    /// row 1's own maximum, and a reduction across rows would clip it
    /// differently.
    #[test]
    fn test_compress_per_call_clamp_is_per_row() {
        let device = Default::default();
        let n_mels = 4;
        let opts = MelConverterOptions::default().with_n_mels(n_mels);
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        // log10: row 0 -> [0, -10, 0, 0]; row 1 -> [-2, -10, -2, -2].
        let energies = vec![
            1.0, 1e-10, 1.0, 1.0, //
            0.01, 1e-10, 0.01, 0.01,
        ];
        let x = Tensor::from_data(TensorData::new(energies, [2, 1, n_mels]), &device);

        let out = conv.compress(x).to_data().to_vec_as::<f64>().unwrap();

        // Row 0: max 0, floor -8, so -10 clips to -8. Affine (v + 4) / 4.
        // Row 1: max -2, floor -10, so -10 is untouched.
        let expected = vec![
            1.0, -1.0, 1.0, 1.0, // (0+4)/4, (-8+4)/4
            0.5, -1.5, 0.5, 0.5, // (-2+4)/4, (-10+4)/4
        ];
        assert_close_to_vec(&out, &expected, 1e-5);

        // A reduction across rows would have used row 0's max for row 1,
        // clipping its -10 to -8 and giving -1.0 instead of -1.5.
        assert!(
            (out[5] - (-1.5)).abs() < 1e-5,
            "row 1 was clamped against another row's maximum",
        );
    }

    #[test]
    fn test_compress_honours_log_base_and_disabled_stages() {
        let device = Default::default();
        let n_mels = 2;

        // Natural log, no clamp, no affine: just `ln(max(v, floor))`.
        let opts = MelConverterOptions::default()
            .with_n_mels(n_mels)
            .with_log_base(LogBase::E)
            .with_range_clamp(None)
            .with_affine(None);
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        let x = Tensor::from_data(
            TensorData::new(vec![1.0_f64, core::f64::consts::E], [1, 1, n_mels]),
            &device,
        );
        assert_tensor_close_to_vec(&conv.compress(x), &[0.0, 1.0], Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_forward_chains_the_stages() {
        let device = Default::default();
        let opts = MelConverterOptions::default();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        let (batch, samples) = (2, 4000);
        let (x, _) = signal(&device, batch, samples);

        let out = conv.forward(x.clone());
        let frames = conv.frame_count(samples);
        assert_eq!(out.dims(), [batch, frames, opts.n_mels]);

        let staged = conv.compress(conv.mel(conv.spectrum(conv.frame(x))));
        assert_tensors_close(&out, &staged, Tolerance::default());

        assert!(
            out.to_data()
                .to_vec_as::<f64>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
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
