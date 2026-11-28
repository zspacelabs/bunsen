//! Sliding-window STFT analysis.

use alloc::{
    string::ToString,
    vec,
    vec::Vec,
};

use burn::{
    prelude::*,
    tensor::signal::rfft,
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::signal::kernels::window::{
        SamplingWindowBuilder,
        StftWindowConfig,
    },
};

/// How a [`SlidingStft`] computes its spectrum.
///
/// [`SlidingStft::spectrum_parts`] matches on this exhaustively, so a new
/// variant does not compile until it has a path.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumImpl {
    /// burn's [`rfft`] over each zero-padded frame: an FFT, so `n_fft` must be
    /// a power of two. Not differentiable (burn's backward is unimplemented).
    Rfft,

    /// An explicit DFT, a matmul against precomputed cosine and sine tables:
    /// any `n_fft` (Whisper's 400), and differentiable.
    DftMatmul,
}

/// Common meta for [`SlidingStftConfig`] and [`SlidingStft`].
pub trait SlidingStftMeta {
    /// The analysis window length, in samples.
    fn win_len(&self) -> usize;

    /// The hop, in samples.
    fn hop(&self) -> usize;

    /// The FFT length; each windowed frame is zero-padded to it.
    fn n_fft(&self) -> usize;

    /// The number of frequency bins: `n_fft / 2 + 1`.
    fn n_bins(&self) -> usize {
        self.n_fft() / 2 + 1
    }
}

/// Config for [`SlidingStft`].
///
/// Defaults to a 768-sample periodic Hann window, hop 256, zero-padded to a
/// 1024-point FFT (513 bins), by [`Rfft`](SpectrumImpl::Rfft).
///
/// Implements [`SlidingStftMeta`].
#[derive(Config, Debug)]
pub struct SlidingStftConfig {
    /// The analysis window length, in samples.
    #[config(default = 768)]
    pub win_len: usize,

    /// The hop, in samples.
    #[config(default = 256)]
    pub hop: usize,

    /// The FFT length; each windowed frame is zero-padded to it.
    #[config(default = 1024)]
    pub n_fft: usize,

    /// The analysis window config.
    #[config(default = "StftWindowConfig::Hann { periodic: true }")]
    pub window: StftWindowConfig,

    /// How the spectrum is computed: [`Rfft`](SpectrumImpl::Rfft) for a
    /// power-of-two `n_fft`, [`DftMatmul`](SpectrumImpl::DftMatmul) for any.
    #[config(default = "SpectrumImpl::Rfft")]
    pub spectrum_impl: SpectrumImpl,
}

impl SlidingStftMeta for SlidingStftConfig {
    fn win_len(&self) -> usize {
        self.win_len
    }

    fn hop(&self) -> usize {
        self.hop
    }

    fn n_fft(&self) -> usize {
        self.n_fft
    }
}

impl SlidingStftConfig {
    /// Validates the analyzer geometry.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, if `hop` is zero, if `hop > win_len` or
    /// `win_len > n_fft`, or if [`Rfft`](SpectrumImpl::Rfft) is asked of an
    /// `n_fft` that is not a power of two.
    pub fn try_validate(&self) -> BunsenResult<()> {
        const OWNER: &str = "SlidingStftConfig";
        if self.hop == 0 {
            return Err(ConstraintError::zero_or_empty(OWNER, "hop").into());
        }
        if self.win_len < self.hop {
            return Err(ConstraintError::new(
                OWNER,
                "",
                Rule::Relation {
                    lhs: ("win_len".into(), self.win_len.to_string()),
                    op: ">=",
                    rhs: ("hop".into(), self.hop.to_string()),
                },
            )
            .into());
        }
        if self.n_fft < self.win_len {
            return Err(ConstraintError::new(
                OWNER,
                "",
                Rule::Relation {
                    lhs: ("n_fft".into(), self.n_fft.to_string()),
                    op: ">=",
                    rhs: ("win_len".into(), self.win_len.to_string()),
                },
            )
            .into());
        }
        if self.spectrum_impl == SpectrumImpl::Rfft && !self.n_fft.is_power_of_two() {
            return Err(ConstraintError::out_of_range(
                OWNER,
                "n_fft",
                self.n_fft,
                "the powers of two (Rfft); DftMatmul takes any",
            )
            .into());
        }
        Ok(())
    }

    /// The real-DFT tables of [`DftMatmul`](SpectrumImpl::DftMatmul),
    /// `[win_len, n_bins]` row-major: the cosines, and the sines carrying the
    /// forward transform's negative sign, on the `n_fft`-point grid (a frame
    /// zero-padded to `n_fft`).
    ///
    /// Built in `f64` regardless of the tensor dtype.
    pub fn to_vec_dft_tables(&self) -> (Vec<f64>, Vec<f64>) {
        let (win_len, n_bins, n_fft) = (self.win_len, self.n_bins(), self.n_fft);
        let step = core::f64::consts::TAU / n_fft as f64;

        let mut cos_table = vec![0.0_f64; win_len * n_bins];
        let mut sin_table = vec![0.0_f64; win_len * n_bins];

        for n in 0..win_len {
            for k in 0..n_bins {
                // Reduce `n * k` before scaling. The product reaches ~80_000
                // at Whisper's geometry, and `(n * k) * step` there has
                // already lost the low bits that `((n * k) % n_fft) * step`
                // keeps exact.
                let theta = ((n * k) % n_fft) as f64 * step;

                cos_table[n * n_bins + k] = theta.cos();
                sin_table[n * n_bins + k] = -theta.sin();
            }
        }

        (cos_table, sin_table)
    }
}

impl ModuleInit<SlidingStft> for SlidingStftConfig {
    /// Initializes a [`SlidingStft`] on `device`.
    ///
    /// # Errors
    ///
    /// See [`try_validate`](SlidingStftConfig::try_validate).
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SlidingStft> {
        self.try_validate()?;
        let (win_len, n_bins) = (self.win_len, self.n_bins());
        let (dft_cos, dft_sin) = match self.spectrum_impl {
            SpectrumImpl::Rfft => (None, None),
            SpectrumImpl::DftMatmul => {
                let (cos_table, sin_table) = self.to_vec_dft_tables();
                (
                    Some(Tensor::from_data(
                        TensorData::new(cos_table, [win_len, n_bins]),
                        device,
                    )),
                    Some(Tensor::from_data(
                        TensorData::new(sin_table, [win_len, n_bins]),
                        device,
                    )),
                )
            }
        };
        Ok(SlidingStft {
            hop: self.hop,
            win_len,
            n_fft: self.n_fft,
            spectrum_impl: self.spectrum_impl,
            window: self.window.to_tensor_window(win_len, device),
            dft_cos,
            dft_sin,
        })
    }
}

/// Fixed sliding-window STFT analysis coefficients.
///
/// Holds the analysis window, and for [`DftMatmul`](SpectrumImpl::DftMatmul)
/// the DFT tables; stateless, so one instance serves any number of streams.
///
/// Built by [`SlidingStftConfig`]. Implements [`SlidingStftMeta`].
/// [`analyze`](Self::analyze) is the one-shot form; its two steps are
/// [`frame`](Self::frame) and [`spectrum_parts`](Self::spectrum_parts). A
/// stream is a [`SlidingStftStreamContext`], from
/// [`new_stream_context`](Self::new_stream_context), carrying its sample
/// queue; [`analyze_stream`](Self::analyze_stream) and
/// [`analyze_hops_stream`](Self::analyze_hops_stream) advance it.
///
/// # Spectrum convention
///
/// The spectrum follows the standard real-DFT convention,
/// `X[k] = Σ_n x[n]·e^(-2πikn/n_fft)` (`numpy.fft.rfft`-compatible), with no
/// normalization, over each frame windowed and zero-padded at its end to
/// `n_fft`. A C reference built on FFTW's half-complex packing emits the same
/// spectrum with the imaginary parts negated; bin powers `re² + im²` agree.
///
/// burn's [`rfft`] does not yet support autodiff (the backward is
/// unimplemented upstream), so [`Rfft`](SpectrumImpl::Rfft) cannot be
/// differentiated through; [`DftMatmul`](SpectrumImpl::DftMatmul) can.
///
/// # Module semantics
///
/// This is a burn `Module` over **bare tensors, never `Param`s**. Nothing here
/// is learnable; `Module` is the traversal-and-device-mapping trait, not the
/// learnability marker. Deriving it is what makes `to_device` and `fork` reach
/// the window, and what lets an enclosing `Module` propagate a device move
/// without hand-written plumbing.
///
/// Two consequences of the bare-tensor choice:
///
/// * The window and tables are **not recorded**. `Module::Record` is
///   `EmptyRecord` and `num_params` is 0; the coefficients are derived from
///   [`SlidingStftConfig`] and rebuilt by
///   [`try_init`](SlidingStftConfig::try_init), never loaded from a checkpoint.
/// * `Module::visit` and `Module::map` **skip them**. `ModuleVisitor` only
///   exposes hooks for `Param`-wrapped tensors, so a `ModuleMapper` pass (dtype
///   casting, quantization) will silently leave them at their original dtype,
///   and they do not appear in the reflection module tree.
#[derive(Module, Debug)]
pub struct SlidingStft {
    hop: usize,
    win_len: usize,
    n_fft: usize,

    #[module(skip)]
    spectrum_impl: SpectrumImpl,

    /// The `[win_len]` analysis window, applied when framing.
    pub window: Tensor<1>,

    /// The `[win_len, n_bins]` real-DFT cosine table, for
    /// [`DftMatmul`](SpectrumImpl::DftMatmul).
    pub dft_cos: Option<Tensor<2>>,

    /// The `[win_len, n_bins]` real-DFT sine table, carrying the forward
    /// transform's negative sign, for [`DftMatmul`](SpectrumImpl::DftMatmul).
    pub dft_sin: Option<Tensor<2>>,
}

impl SlidingStftMeta for SlidingStft {
    fn win_len(&self) -> usize {
        self.win_len
    }

    fn hop(&self) -> usize {
        self.hop
    }

    fn n_fft(&self) -> usize {
        self.n_fft
    }
}

impl SlidingStft {
    /// How the spectrum is computed.
    pub fn spectrum_impl(&self) -> SpectrumImpl {
        self.spectrum_impl
    }

    /// The number of whole frames a `samples`-long signal yields:
    /// `(samples - win_len) / hop + 1`, or zero when it is shorter than one
    /// window.
    pub fn frame_count(
        &self,
        samples: usize,
    ) -> usize {
        match samples < self.win_len {
            true => 0,
            false => (samples - self.win_len) / self.hop + 1,
        }
    }

    /// Allocate a zeroed window.
    pub fn zero_window(
        &self,
        batch_size: usize,
    ) -> Tensor<2> {
        Tensor::zeros([batch_size, self.win_len()], &self.window.device())
    }

    /// Frames a signal into consecutive windowed frames.
    ///
    /// Frame `f` covers `x[.., f * hop .. f * hop + win_len]`, multiplied by
    /// the analysis window. Trailing samples that do not fill a whole frame
    /// are dropped (in a stream they belong to the carry, not to any frame).
    ///
    /// # Arguments
    /// * `x`: `[batch, samples]`, with `samples >= win_len`.
    ///
    /// # Returns
    /// `[batch, frames, win_len]` windowed frames, with
    /// `frames = (samples - win_len) / hop + 1`.
    pub fn frame(
        &self,
        x: Tensor<2>,
    ) -> Tensor<3> {
        let [batch, samples] =
            crate::contracts::unpack_shape_contract!(["batch", "samples"], &x.dims());
        assert!(
            samples >= self.win_len,
            "SlidingStft samples ({samples}) must be >= win_len ({})",
            self.win_len,
        );
        let (win_len, hop) = (self.win_len, self.hop);
        let frames = self.frame_count(samples);

        // ── `unfold` stride hazard ──────────────────────────────────────────
        // Trim to exactly the span the frames cover before unfolding. burn
        // 0.21 on `CubeCL` backends (wgpu / cuda / metal; `Flex` is correct)
        // truncated the unfolded view's outer stride to the vectorization line
        // width `v` — the largest power of two dividing both `size` and `step`
        // — using `(len / v) * v` in place of `len`, so every batch row after
        // the first was read `len % v` elements early (row 0 is always right,
        // so a `batch == 1` test cannot see it). It fires when `size` and
        // `step` are both even and the uncovered tail is not a multiple of
        // `v`; Whisper's geometry (400, 160, `v` 16, tail 120) trips it at
        // every chunk size. Trimming makes the tail zero, so the defect is
        // unreachable on any burn version; it is also the honest framing.
        // ────────────────────────────────────────────────────────────────────
        let covered = (frames - 1) * hop + win_len;
        let x = x.slice_dim(1, 0..covered as isize);

        // [batch, frames, win_len]
        let framed: Tensor<3> = x.unfold(1, win_len, hop);
        let framed = framed.mul(self.window.clone().reshape([1, 1, win_len]));

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "frames", "win_len"],
            &framed.dims(),
            &[("batch", batch), ("frames", frames), ("win_len", win_len)],
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;
        framed
    }

    /// The real and imaginary parts of windowed frames' spectra.
    ///
    /// # Arguments
    /// * `frames`: `[batch, frames, win_len]` windowed frames, from
    ///   [`frame`](Self::frame).
    ///
    /// # Returns
    /// `(re, im)`, each `[batch, frames, n_bins]`.
    pub fn spectrum_parts(
        &self,
        frames: Tensor<3>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [batch, n_frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "frames", "win_len"],
            &frames.dims(),
            &["batch", "frames"],
            &[("win_len", self.win_len)],
        );
        let n_bins = self.n_bins();
        let (re, im) = match self.spectrum_impl {
            SpectrumImpl::Rfft => rfft(frames, 2, Some(self.n_fft)),
            SpectrumImpl::DftMatmul => {
                // Fold batch and frame together: one
                // `[rows, win_len] @ [win_len, n_bins]` matmul beats
                // broadcasting the tables across a batch axis.
                let rows = batch * n_frames;
                let flat: Tensor<2> = frames.reshape([rows, self.win_len]);
                let table = |t: &Option<Tensor<2>>| {
                    t.clone()
                        .expect("a DftMatmul analyzer is built with its tables")
                };
                let re = flat.clone().matmul(table(&self.dft_cos));
                let im = flat.matmul(table(&self.dft_sin));
                (
                    re.reshape([batch, n_frames, n_bins]),
                    im.reshape([batch, n_frames, n_bins]),
                )
            }
        };
        #[cfg(any(debug_assertions, test))]
        for part in [&re, &im] {
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "frames", "n_bins"],
                &part.dims(),
                &[("batch", batch), ("frames", n_frames), ("n_bins", n_bins)],
            );
        }
        (re, im)
    }

    /// The power spectrum of windowed frames, `re² + im²`.
    ///
    /// # Arguments
    /// * `frames`: `[batch, frames, win_len]`, from [`frame`](Self::frame).
    ///
    /// # Returns
    /// `[batch, frames, n_bins]`.
    pub fn power(
        &self,
        frames: Tensor<3>,
    ) -> Tensor<3> {
        let (re, im) = self.spectrum_parts(frames);
        // Squaring by multiply rather than `powi_scalar(2)`: same result, and
        // the form the fusion pass folds into the matmul epilogue.
        re.clone().mul(re).add(im.clone().mul(im))
    }

    /// Analyzes a signal into consecutive aligned STFT frames.
    ///
    /// Frame `f` covers `signal[f * hop .. f * hop + win_len]`, windowed and
    /// zero-padded to `n_fft`. Trailing samples that do not fill a whole
    /// window are ignored.
    ///
    /// # Arguments
    /// * `signal`: `[batch, samples]`, with `samples >= win_len`.
    ///
    /// # Returns
    /// `[batch, frames, n_bins, 2]` spectra, with
    /// `frames = 1 + (samples - win_len) / hop`; the trailing axis is
    /// `(re, im)`.
    pub fn analyze(
        &self,
        signal: Tensor<2>,
    ) -> Tensor<4> {
        let (re, im) = self.spectrum_parts(self.frame(signal));
        Tensor::stack(vec![re, im], 3)
    }

    /// A stream of `batch_size` independent rows through these coefficients:
    /// its queue zeroed, on their device.
    ///
    /// # Panics
    /// If `batch_size` is zero.
    pub fn new_stream_context(
        &self,
        batch_size: usize,
    ) -> SlidingStftStreamContext {
        assert_ne!(batch_size, 0, "SlidingStft batch_size must be non-zero");
        SlidingStftStreamContext {
            queue: self.zero_window(batch_size),
        }
    }

    /// Pushes one hop into the stream and returns the spectrum of its
    /// updated window.
    ///
    /// # Arguments
    /// * `hop`: `[batch, hop]` new samples.
    /// * `ctx`: the stream, advanced past `hop`.
    ///
    /// # Returns
    /// `[batch, n_bins, 2]` spectrum; the trailing axis is `(re, im)`.
    pub fn analyze_stream(
        &self,
        hop: Tensor<2>,
        ctx: &mut SlidingStftStreamContext,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract!(
            ["batch", "hop"],
            &hop,
            &[("batch", ctx.batch_size()), ("hop", self.hop())],
        );

        let hop_len = self.hop();
        if self.win_len() == hop_len {
            ctx.queue = hop;
        } else {
            // Inplace, so we can drop the reference before the update.
            ctx.queue.inplace(|q| {
                let keep = q.slice_dim(1, hop_len as isize..);
                Tensor::cat(vec![keep, hop], 1)
            });
        }

        // One full window -> one frame.
        // [batch, 1, n_bins, 2]
        let x = self.analyze(ctx.queue.clone());
        // [batch, n_bins, 2]
        x.squeeze_dim(1)
    }

    /// Pushes `steps` consecutive hops into the stream at once.
    ///
    /// Equivalent to `steps` calls of [`analyze_stream`](Self::analyze_stream),
    /// but the whole extended stream is analyzed at once.
    ///
    /// # Arguments
    /// * `hops`: `[steps, batch, hop]` consecutive hops.
    /// * `ctx`: the stream, advanced past `hops`.
    ///
    /// # Returns
    /// `[steps, batch, n_bins, 2]` per-hop spectra; the trailing axis is
    /// `(re, im)`.
    pub fn analyze_hops_stream(
        &self,
        hops: Tensor<3>,
        ctx: &mut SlidingStftStreamContext,
    ) -> Tensor<4> {
        #[cfg(any(debug_assertions, test))]
        let [steps] = crate::contracts::unpack_shape_contract!(
            ["steps", "batch", "hop"],
            &hops,
            &["steps"],
            &[("batch", ctx.batch_size()), ("hop", self.hop())],
        );

        // [batch, steps * hop]
        let stream = hops.swap_dims(0, 1).flatten::<2>(1, 2);
        let win_len = self.win_len();

        let queue = ctx.queue.extract();
        let ext = Tensor::cat(vec![queue, stream], 1);
        ctx.queue = ext.clone().slice_dim(1, -(win_len as isize)..);

        // `analyze` yields `steps + 1` frames at `hop` offsets; frame
        // `s + 1` is the queue state after hop `s`, and frame 0 (the
        // pre-push queue) is dropped.
        //
        // [batch, steps + 1, n_bins, 2]
        let x = self.analyze(ext);
        // [steps, batch, n_bins, 2]
        let x = x.slice_dim(1, 1..).swap_dims(0, 1);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract!(
            ["steps", "batch", "n_bins", 2],
            &x,
            &[
                ("steps", steps),
                ("batch", ctx.batch_size()),
                ("n_bins", self.n_bins())
            ],
        );
        x
    }
}

/// One stream through a [`SlidingStft`]: its sliding sample queue.
///
/// Each push ([`analyze_stream`](SlidingStft::analyze_stream)) shifts the
/// queue left by `hop` and appends the new hop. At stream start the queue
/// is zero, so the first `win_len / hop - 1` spectra cover partially
/// zero-padded windows.
///
/// Built by [`SlidingStft::new_stream_context`]. Like [`SlidingStft`], this is
/// a `Module` over bare tensors (see that type's *Module semantics*): the
/// queue moves with `to_device`, and is neither recorded nor visited.
#[derive(Module, Debug)]
pub struct SlidingStftStreamContext {
    /// The sliding sample queue: `[batch, win_len]`.
    ///
    /// Each row holds the `win_len` most recent samples of that stream
    /// (zeros before the stream starts).
    pub queue: Tensor<2>,
}

impl SlidingStftStreamContext {
    /// The batch size; each batch row is an independent stream.
    pub fn batch_size(&self) -> usize {
        self.queue.dims()[0]
    }

    /// Resets the queue to zeros.
    pub fn reset(&mut self) {
        self.queue = Tensor::zeros_like(&self.queue);
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::{
            DeviceMemoryGuard,
            cpu_device,
            performance_device,
            tensor_to_vec_as,
        },
    };

    type F = f32;

    #[test]
    fn test_config_meta() {
        let cfg = SlidingStftConfig::new();
        assert_eq!(cfg.win_len, 768);
        assert_eq!(cfg.hop, 256);
        assert_eq!(cfg.n_fft, 1024);
        assert_eq!(cfg.window, StftWindowConfig::Hann { periodic: true });

        assert_eq!(cfg.win_len(), 768);
        assert_eq!(cfg.hop(), 256);
        assert_eq!(cfg.n_fft(), 1024);
        assert_eq!(cfg.n_bins(), 513);

        cfg.try_validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        for bad in [
            SlidingStftConfig::new().with_hop(0),
            // hop > win_len
            SlidingStftConfig::new().with_hop(1024),
            // win_len > n_fft
            SlidingStftConfig::new().with_win_len(2048),
            // Rfft at a non-power-of-two n_fft (odd)
            SlidingStftConfig::new().with_n_fft(1023).with_win_len(768),
            // Rfft at a non-power-of-two n_fft (even)
            SlidingStftConfig::new().with_n_fft(1536).with_win_len(768),
        ] {
            ErrorMatcher::illegal()
                .has_cause::<ConstraintError>()
                .assert_err(&bad.try_validate());
        }
    }

    #[test]
    fn test_init_meta_matches_config() {
        let device = cpu_device();
        let cfg = SlidingStftConfig::new()
            .with_win_len(48)
            .with_hop(16)
            .with_n_fft(64);

        let coef: SlidingStft = cfg.init(&device);

        assert_eq!(coef.win_len(), cfg.win_len());
        assert_eq!(coef.hop(), cfg.hop());
        assert_eq!(coef.n_fft(), cfg.n_fft());
        assert_eq!(coef.n_bins(), cfg.n_bins());

        // The stored window is the `win_len` coefficients; the frame's
        // zero-padding to `n_fft` is the transform's.
        assert_eq!(coef.window.dims(), [48]);
        let window: Vec<f64> = tensor_to_vec_as::<f64, _>(&coef.window);
        let host = cfg.window.to_vec_window(48);
        for (n, (&w, &h)) in window.iter().zip(&host).enumerate() {
            assert!((w - h).abs() <= 1e-6, "window[{n}]: {w} vs {h}");
        }

        let stft = coef.new_stream_context(3);
        assert_eq!(stft.batch_size(), 3);

        // The queue starts zeroed.
        assert_eq!(stft.queue.dims(), [3, 48]);
        stft.queue
            .to_data()
            .assert_eq(&Tensor::<2>::zeros([3, 48], &device).to_data(), true);
    }

    /// The `Module` derive is what carries the tensors across devices; assert
    /// the traversal reaches every field.
    ///
    /// A single-device test environment can't exercise a real cross-device
    /// move, but `devices()` still catches the regressions that matter: a
    /// dropped derive, a stray `#[module(skip)]`, or a nested field that stops
    /// being traversed.
    #[test]
    fn test_module_semantics() {
        let device = cpu_device();
        let cfg = SlidingStftConfig::new()
            .with_win_len(48)
            .with_hop(16)
            .with_n_fft(64);

        let coef: SlidingStft = cfg.init(&device);
        let ctx = coef.new_stream_context(2);

        // Traversal reaches the window and the queue. `collect_devices`
        // dedupes, so one device means one entry.
        assert_eq!(coef.devices(), vec![device.clone()]);
        assert_eq!(ctx.devices(), vec![device.clone()]);

        // Bare tensors, not `Param`s: nothing here is learnable.
        assert_eq!(coef.num_params(), 0);
        assert_eq!(ctx.num_params(), 0);

        let window_before = coef.window.to_data_as::<F>();
        let queue_before = ctx.queue.to_data_as::<F>();

        // `to_device` moves every field, preserving values and geometry.
        let moved = coef.clone().to_device(&device);
        assert_eq!(moved.devices(), vec![device.clone()]);
        moved
            .window
            .to_data_as::<F>()
            .assert_eq(&window_before, true);
        assert_eq!(moved.win_len(), coef.win_len());
        assert_eq!(moved.hop(), coef.hop());
        assert_eq!(moved.n_fft(), coef.n_fft());

        let moved_ctx = ctx.clone().to_device(&device);
        assert_eq!(moved_ctx.devices(), vec![device.clone()]);
        moved_ctx
            .queue
            .to_data_as::<F>()
            .assert_eq(&queue_before, true);
        assert_eq!(moved_ctx.batch_size(), ctx.batch_size());
    }

    #[test]
    #[should_panic(expected = "batch_size must be non-zero")]
    fn test_new_stream_context_rejects_zero_batch() {
        let device = cpu_device();
        let coef: SlidingStft = SlidingStftConfig::new().init(&device);
        coef.new_stream_context(0);
    }

    /// Host-side reference: a sliding queue and naive windowed real DFT.
    struct HostStft {
        win_len: usize,
        hop: usize,
        n_fft: usize,
        window: Vec<f64>,
        queue: Vec<f64>,
    }

    impl HostStft {
        fn new(cfg: &SlidingStftConfig) -> Self {
            Self {
                win_len: cfg.win_len,
                hop: cfg.hop,
                n_fft: cfg.n_fft,
                window: cfg.window.to_vec_window(cfg.win_len),
                queue: vec![0.0; cfg.win_len],
            }
        }

        /// Pushes one hop; returns interleaved `(re, im)` pairs per bin.
        fn push(
            &mut self,
            hop: &[f64],
        ) -> Vec<f64> {
            assert_eq!(hop.len(), self.hop);
            self.queue.drain(..self.hop);
            self.queue.extend(hop.iter().map(|&v| v as f64));

            let n_bins = self.n_fft / 2 + 1;
            let mut out = Vec::with_capacity(2 * n_bins);
            for k in 0..n_bins {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for n in 0..self.win_len {
                    let x = self.queue[n] * self.window[n] as f64;
                    let theta =
                        core::f64::consts::TAU * ((n * k) % self.n_fft) as f64 / self.n_fft as f64;
                    re += x * theta.cos();
                    im -= x * theta.sin();
                }
                out.push(re);
                out.push(im);
            }
            out
        }
    }

    /// A deterministic pseudo-random sample value.
    fn sample(
        batch: usize,
        step: usize,
        idx: usize,
    ) -> f64 {
        ((batch * 7919 + step * 104729 + idx * 1299709) % 1000) as f64 - 500.0
    }

    /// Host reference for [`SlidingStft::analyze`]: frame `f` covers
    /// `row[f * hop .. f * hop + win_len]`, windowed and
    /// zero-padded to `n_fft`, as interleaved `(re, im)` per bin.
    fn host_analyze(
        row: &[f64],
        cfg: &SlidingStftConfig,
    ) -> Vec<f64> {
        let window = cfg.window.to_vec_window(cfg.win_len);
        let frames = 1 + (row.len() - cfg.win_len) / cfg.hop;
        let n_bins = cfg.n_bins();

        let mut out = Vec::with_capacity(frames * n_bins * 2);
        for f in 0..frames {
            for k in 0..n_bins {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for n in 0..cfg.win_len {
                    let x = row[f * cfg.hop + n] * window[n];
                    let theta =
                        core::f64::consts::TAU * ((n * k) % cfg.n_fft) as f64 / cfg.n_fft as f64;
                    re += x * theta.cos();
                    im -= x * theta.sin();
                }
                out.push(re);
                out.push(im);
            }
        }
        out
    }

    /// `analyze` yields `1 + (samples - win_len) / hop` frames, for
    /// hop-aligned and ragged `samples` alike: trailing samples that do not
    /// fill a whole window are ignored.
    #[test]
    fn test_analyze_frame_count_and_shape() {
        let device = cpu_device();
        let cfg = SlidingStftConfig::new()
            .with_win_len(48)
            .with_hop(16)
            .with_n_fft(64);
        let coef: SlidingStft = cfg.init(&device);

        for (samples, frames) in [(48, 1), (64, 2), (80, 3), (88, 3), (95, 3), (96, 4)] {
            let x = Tensor::<2>::from_data(
                TensorData::new(vec![0.0f64; 2 * samples], [2, samples]),
                &device,
            );
            assert_eq!(
                coef.analyze(x).dims(),
                [2, frames, cfg.n_bins(), 2],
                "samples = {samples}",
            );
        }
    }

    /// `analyze` matches a naive per-frame DFT of the windowed, zero-padded
    /// signal. Uses a ragged `samples`, so the ignored tail is covered too.
    #[test]
    fn test_analyze_matches_naive_dft() {
        let device = cpu_device();
        // `Rfft` at a power of two; `DftMatmul` there and at 60, which `Rfft`
        // can't take.
        for cfg in [
            SlidingStftConfig::new()
                .with_win_len(48)
                .with_hop(16)
                .with_n_fft(64),
            SlidingStftConfig::new()
                .with_win_len(48)
                .with_hop(16)
                .with_n_fft(64)
                .with_spectrum_impl(SpectrumImpl::DftMatmul),
            SlidingStftConfig::new()
                .with_win_len(48)
                .with_hop(16)
                .with_n_fft(60)
                .with_spectrum_impl(SpectrumImpl::DftMatmul),
        ] {
            let coef: SlidingStft = cfg.init(&device);

            let batch = 2;
            let samples = 88;
            let rows: Vec<Vec<f64>> = (0..batch)
                .map(|b| (0..samples).map(|i| sample(b, 0, i)).collect())
                .collect();

            let got = coef.analyze(Tensor::<2>::from_data(
                TensorData::new(rows.concat(), [batch, samples]),
                &device,
            ));

            let expected: Vec<f64> = rows.iter().flat_map(|r| host_analyze(r, &cfg)).collect();
            let frames = 1 + (samples - cfg.win_len) / cfg.hop;

            got.to_data_as::<F>().assert_approx_eq::<F>(
                &TensorData::new(expected, [batch, frames, cfg.n_bins(), 2]).convert::<F>(),
                Tolerance::permissive(),
            );
        }
    }

    /// Batched analysis agrees with analyzing each row on its own.
    ///
    /// Runs on `performance_device()`, not the module's `cpu_device()`: a
    /// batching fault in the framing beneath `stft` would live in a `CubeCL`
    /// kernel and show only for rows after the first. `samples` is ragged,
    /// the shape that leaves an uncovered tail for the framing to mishandle.
    #[test]
    #[serial_test::serial]
    fn test_analyze_ragged_batch_matches_single_rows() {
        type PF = f32;

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        // The default geometry, with `samples = win_len + 8` leaving an
        // uncovered tail of 8. Small geometries do not vectorize, so they
        // cannot exercise the framing's vectorized path at all.
        let cfg = SlidingStftConfig::new();
        let coef: SlidingStft = cfg.init(&device);

        let batch = 3;
        let samples = cfg.win_len + 8;
        assert_ne!(
            (samples - cfg.win_len) % cfg.hop,
            0,
            "geometry must be ragged"
        );

        let rows: Vec<Vec<f64>> = (0..batch)
            .map(|b| (0..samples).map(|i| sample(b, 0, i)).collect())
            .collect();

        let batched = coef.analyze(Tensor::<2>::from_data(
            TensorData::new(rows.concat(), [batch, samples]),
            &device,
        ));

        for (b, row) in rows.iter().enumerate() {
            let alone = coef.analyze(Tensor::<2>::from_data(
                TensorData::new(row.clone(), [1, samples]),
                &device,
            ));

            let got = batched.clone().slice_dim(0, b as isize..b as isize + 1);
            assert_eq!(got.dims(), alone.dims());

            got.to_data_as::<PF>()
                .assert_approx_eq::<PF>(&alone.to_data_as::<PF>(), Tolerance::permissive());
        }
    }

    #[test]
    fn test_forward_matches_naive_dft() {
        let device = cpu_device();
        let cfg = SlidingStftConfig::new()
            .with_win_len(48)
            .with_hop(16)
            .with_n_fft(64);
        let batch = 2;
        let n_bins = cfg.n_bins();

        let coef: SlidingStft = cfg.init(&device);
        let mut stft = coef.new_stream_context(batch);
        let mut hosts: Vec<HostStft> = (0..batch).map(|_| HostStft::new(&cfg)).collect();

        // The first pushes cover the zero-padded warmup; the later ones a
        // full queue.
        for step in 0..5 {
            let rows: Vec<Vec<f64>> = (0..batch)
                .map(|b| (0..cfg.hop).map(|i| sample(b, step, i)).collect())
                .collect();

            let hop =
                Tensor::<2>::from_data(TensorData::new(rows.concat(), [batch, cfg.hop]), &device);
            let out = coef.analyze_stream(hop, &mut stft);
            assert_eq!(out.dims(), [batch, n_bins, 2]);

            let expected: Vec<f64> = hosts
                .iter_mut()
                .zip(&rows)
                .flat_map(|(host, row)| host.push(row))
                .collect();

            out.to_data_as::<F>().assert_approx_eq::<F>(
                &TensorData::new(expected, [batch, n_bins, 2]).convert::<F>(),
                Tolerance::permissive(),
            );
        }
    }

    #[test]
    fn test_analyze_hops_stream_matches_stepwise() {
        let device = cpu_device();
        let steps = 7;
        let batch = 2;

        for cfg in [
            SlidingStftConfig::new()
                .with_win_len(48)
                .with_hop(16)
                .with_n_fft(64),
            // win_len == hop: no overlap between frames.
            SlidingStftConfig::new()
                .with_win_len(16)
                .with_hop(16)
                .with_n_fft(16),
        ] {
            let hops = Tensor::<3>::random([steps, batch, cfg.hop], Distribution::Default, &device);

            let coef: SlidingStft = cfg.init(&device);
            let mut seq_stft = coef.new_stream_context(batch);
            let mut step_stft = seq_stft.clone();

            let seq_out = coef.analyze_hops_stream(hops.clone(), &mut seq_stft);
            assert_eq!(seq_out.dims(), [steps, batch, cfg.n_bins(), 2]);

            let mut step_outs = Vec::with_capacity(steps);
            for step in 0..steps {
                let hop = hops.clone().select_dim(0, step);
                step_outs.push(coef.analyze_stream(hop, &mut step_stft));
            }
            let step_out: Tensor<4> = Tensor::stack(step_outs, 0);

            let tol = Tolerance::<F>::permissive();
            seq_out
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&step_out.to_data_as::<F>(), tol);
            seq_stft
                .queue
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&step_stft.queue.to_data_as::<F>(), tol);
        }
    }

    #[test]
    fn test_reset() {
        let device = cpu_device();
        let cfg = SlidingStftConfig::new()
            .with_win_len(48)
            .with_hop(16)
            .with_n_fft(64);

        let coef: SlidingStft = cfg.init(&device);
        let mut stft = coef.new_stream_context(1);
        let hop = Tensor::random([1, 16], Distribution::Default, &device);
        coef.analyze_stream(hop, &mut stft);

        stft.reset();
        stft.queue.to_data_as::<F>().assert_eq(
            &Tensor::<2>::zeros([1, 48], &device).to_data_as::<F>(),
            true,
        );
    }
}
