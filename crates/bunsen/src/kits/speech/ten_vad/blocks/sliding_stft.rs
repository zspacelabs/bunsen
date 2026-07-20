//! # Sliding-window STFT analyzer.
//!
//! A burn port of the ten-vad reference STFT analyzer (`AUP_Analyzer`,
//! `src/stft.cc` in the reference sources; see also `ALGO_TRACE.md` §3.4):
//! a persistent `win_len`-sample queue is shifted left by `hop_size` and the
//! new hop appended, so each spectrum covers the current hop plus the
//! preceding `win_len - hop_size` samples. The queue is multiplied by the
//! analysis window, zero-padded to `fft_size`, and projected through a real
//! DFT.
//!
//! The analyzer is split into:
//! * [`SlidingStft`] — the fixed analysis coefficients (window, geometry);
//!   stateless, shareable across streams.
//! * [`SlidingStftContext`] — a streaming state (the sample queue) bound to a
//!   [`SlidingStft`]; built by [`SlidingStft::init_state`].
//!
//! The spectrum follows the standard real-DFT convention,
//! `X[k] = Σ_n x[n]·e^(-2πikn/fft_size)` (`numpy.fft.rfft`-compatible),
//! with no normalization; it is computed with [`burn::tensor::signal::rfft`].
//! The C reference emits the same spectrum with the imaginary parts negated
//! (an artifact of its FFTW half-complex packing); bin powers `re² + im²`
//! agree.
//!
//! Note: burn's `rfft` does not yet support autodiff (the backward is
//! unimplemented upstream), so this analyzer cannot currently be
//! differentiated through.

use burn::{
    prelude::*,
    tensor::signal::rfft,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    ops::arange::arange_start_step,
};

/// Common meta for [`SlidingStftConfig`], [`SlidingStft`], and
/// [`SlidingStftContext`].
pub trait SlidingStftMeta {
    /// The analysis window length, in samples.
    fn win_len(&self) -> usize;

    /// The hop size, in samples.
    fn hop_size(&self) -> usize;

    /// The FFT size; the windowed queue is zero-padded to this length.
    fn fft_size(&self) -> usize;

    /// The number of frequency bins: `fft_size / 2 + 1`.
    fn n_bins(&self) -> usize {
        self.fft_size() / 2 + 1
    }
}

/// Analysis window function for [`SlidingStft`].
#[derive(Config, Debug, PartialEq)]
pub enum SlidingStftWindow {
    /// All-ones (rectangular) window.
    ///
    /// The reference analyzer default when no coefficient table is provided.
    Ones,

    /// Periodic Hann window: `0.5 - 0.5 * cos(2π n / win_len)`.
    ///
    /// Matches the reference `AUP_AED_STFTWindow_Hann768` table (`coeff.h`).
    Hann,

    /// Periodic Hamming window: `0.54 - 0.46 * cos(2π n / win_len)`.
    Hamming,

    /// Custom period.
    /// Window: `alpha - beta * cos(2π n / win_len)`.
    Custom(f64, f64),
}

// The `Config` derive does not preserve the `#[default]` helper attribute,
// so `Default` cannot be derived alongside it.
#[allow(clippy::derivable_impls)]
impl Default for SlidingStftWindow {
    fn default() -> Self {
        SlidingStftWindow::Hann
    }
}

impl SlidingStftWindow {
    /// The `(alpha, beta)` parameters of the periodic raised-cosine family,
    /// `w[n] = alpha - beta * cos(2π n / win_len)`; `None` for [`Self::Ones`].
    fn window_params(&self) -> Option<(f64, f64)> {
        match self {
            Self::Ones => Some((1.0, 0.0)),
            Self::Hann => Some((0.5, 0.5)),
            Self::Hamming => Some((0.54, 0.46)),
            Self::Custom(alpha, beta) => Some((*alpha, *beta)),
        }
    }

    /// The window coefficient table for a `win_len`-sample window.
    pub fn coefficients(
        &self,
        win_len: usize,
    ) -> Vec<f32> {
        match self.window_params() {
            None => vec![1.0; win_len],
            Some((alpha, beta)) => {
                let scale = core::f64::consts::TAU / win_len as f64;
                (0..win_len)
                    .map(|n| {
                        // (2π | τ) * n / win_len
                        let theta = (n as f64) * scale;

                        (alpha - beta * theta.cos()) as f32
                    })
                    .collect()
            }
        }
    }

    /// The window coefficient table for a `win_len`-sample window,
    /// materialized on `device`.
    pub fn coefficients_tensor<B: Backend>(
        &self,
        win_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        match self.window_params() {
            None => Tensor::ones([win_len], device),
            Some((alpha, beta)) => {
                // (2π | τ) * n / win_len
                let step = core::f64::consts::TAU / win_len as f64;
                let theta = arange_start_step(win_len, 0.0, Some(step), device);

                theta.cos().mul_scalar(-beta).add_scalar(alpha)
            }
        }
    }
}

/// Config for [`SlidingStft`].
///
/// Defaults match the ten-vad analyzer: a 768-sample periodic Hann window,
/// hop 256, zero-padded to a 1024-point FFT (513 bins).
///
/// Implements [`SlidingStftMeta`].
#[derive(Config, Debug)]
pub struct SlidingStftConfig {
    /// The analysis window length, in samples.
    #[config(default = "768")]
    pub win_len: usize,

    /// The hop size, in samples.
    #[config(default = "256")]
    pub hop_size: usize,

    /// The FFT size; the windowed queue is zero-padded to this length.
    #[config(default = "1024")]
    pub fft_size: usize,

    /// The analysis window function.
    #[config(default = "SlidingStftWindow::Hann")]
    pub window: SlidingStftWindow,
}

impl SlidingStftMeta for SlidingStftConfig {
    fn win_len(&self) -> usize {
        self.win_len
    }

    fn hop_size(&self) -> usize {
        self.hop_size
    }

    fn fft_size(&self) -> usize {
        self.fft_size
    }
}

impl SlidingStftConfig {
    /// Validates the analyzer geometry.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if `hop_size` is zero, if
    /// `hop_size > win_len` or `win_len > fft_size`, or if `fft_size` is not
    /// a power of two.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.hop_size == 0 {
            return Err(BunsenError::Invalid(
                "SlidingStft hop_size must be non-zero".to_string(),
            ));
        }
        if self.win_len < self.hop_size {
            return Err(BunsenError::Invalid(format!(
                "SlidingStft win_len ({}) must be >= hop_size ({})",
                self.win_len, self.hop_size,
            )));
        }
        if self.fft_size < self.win_len {
            return Err(BunsenError::Invalid(format!(
                "SlidingStft fft_size ({}) must be >= win_len ({})",
                self.fft_size, self.win_len,
            )));
        }
        if !self.fft_size.is_power_of_two() {
            return Err(BunsenError::Invalid(format!(
                "SlidingStft fft_size ({}) must be a power of two",
                self.fft_size,
            )));
        }
        Ok(())
    }

    /// Initializes a [`SlidingStft`] on `device`.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<SlidingStft<B>> {
        self.validate()?;

        let win_len = self.win_len;

        let window = self.window.coefficients_tensor(win_len, device);

        Ok(SlidingStft {
            hop_size: self.hop_size,
            fft_size: self.fft_size,
            window,
        })
    }

    /// Initializes a [`SlidingStft`] on `device`, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> SlidingStft<B> {
        self.try_init(device).ok_or_panic()
    }
}

/// Fixed sliding-window STFT analysis coefficients.
///
/// Holds the analysis window and geometry; stateless, so one instance can be
/// shared by (or cheaply cloned into) any number of streams. This is
/// deliberately **not** a burn `Module`: nothing here is a learnable
/// parameter.
///
/// Built by [`SlidingStftConfig`]. Implements [`SlidingStftMeta`].
/// Streaming states are built by [`init_state`](Self::init_state).
#[derive(Debug, Clone)]
pub struct SlidingStft<B: Backend> {
    hop_size: usize,
    fft_size: usize,

    /// The analysis window coefficients: `[win_len]`.
    pub window: Tensor<B, 1>,
}

impl<B: Backend> SlidingStftMeta for SlidingStft<B> {
    fn win_len(&self) -> usize {
        self.window.dims()[0]
    }

    fn hop_size(&self) -> usize {
        self.hop_size
    }

    fn fft_size(&self) -> usize {
        self.fft_size
    }
}

impl<B: Backend> SlidingStft<B> {
    /// Builds a [`SlidingStftContext`] streaming state over these coefficients.
    ///
    /// The queue is zeroed, on the same device as the coefficients.
    ///
    /// # Arguments
    /// * `batch_size`: the number of independent streams; must be non-zero.
    pub fn init_state(
        &self,
        batch_size: usize,
    ) -> SlidingStftContext<B> {
        assert_ne!(batch_size, 0, "SlidingStft batch_size must be non-zero");
        let queue = self.zero_window(batch_size);
        SlidingStftContext {
            coef: self.clone(),
            queue,
        }
    }

    /// Allocate a zeroed window.
    pub fn zero_window(
        &self,
        batch_size: usize,
    ) -> Tensor<B, 2> {
        Tensor::zeros([batch_size, self.win_len()], &self.window.device())
    }

    /// Analyzes full analysis frames: windows each frame, zero-pads it to
    /// `fft_size`, and projects it through the real DFT.
    ///
    /// Uses [`burn::tensor::signal::rfft`], which does not yet support
    /// autodiff.
    ///
    /// # Arguments
    /// * `frames`: `[batch, steps, win_len]` analysis frames.
    ///
    /// # Returns
    /// `[batch, steps, n_bins, 2]` spectra; the trailing axis is `(re, im)`.
    pub fn analyze(
        &self,
        frames: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        #[cfg(any(test, debug_assertions))]
        let [batch, steps] = crate::contracts::unpack_shape_contract!(
            ["batch", "steps", "win_len"],
            &frames,
            &["batch", "steps"],
            &[("win_len", self.win_len())],
        );

        // [batch, steps, win_len]
        let frames = frames * self.window.clone().unsqueeze::<3>();

        // [batch, steps, n_bins] each; rfft zero-pads to fft_size.
        let (re, im) = rfft(frames, 2, Some(self.fft_size()));

        // [batch, steps, n_bins, 2]
        let x = Tensor::stack(vec![re, im], 3);

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "steps", "n_bins", 2],
            &x,
            &[
                ("batch", batch),
                ("steps", steps),
                ("n_bins", self.n_bins())
            ],
        );
        x
    }
}

/// Streaming sliding-window STFT analyzer state.
///
/// Binds a sliding sample queue to a [`SlidingStft`].
///
/// Each [`forward`](Self::forward) shifts the queue left by `hop_size`,
/// appends the new hop, and returns the real-DFT spectrum of the windowed,
/// zero-padded queue (see the module docs for the convention). At stream
/// start the queue is zero, so the first `win_len / hop_size - 1` spectra
/// cover partially zero-padded windows.
///
/// Built by [`SlidingStft::init_state`]. Implements [`SlidingStftMeta`].
#[derive(Debug, Clone)]
pub struct SlidingStftContext<B: Backend> {
    /// The fixed analysis coefficients.
    pub coef: SlidingStft<B>,

    /// The sliding sample queue: `[batch, win_len]`.
    ///
    /// Each row holds the `win_len` most recent samples of that stream
    /// (zeros before the stream starts).
    pub queue: Tensor<B, 2>,
}

impl<B: Backend> SlidingStftMeta for SlidingStftContext<B> {
    fn win_len(&self) -> usize {
        self.coef.win_len()
    }

    fn hop_size(&self) -> usize {
        self.coef.hop_size()
    }

    fn fft_size(&self) -> usize {
        self.coef.fft_size()
    }
}

impl<B: Backend> SlidingStftContext<B> {
    /// The batch size; each batch row is an independent stream.
    pub fn batch_size(&self) -> usize {
        self.queue.dims()[0]
    }

    /// Resets the streaming queue to zeros.
    pub fn reset(&mut self) {
        self.queue = Tensor::zeros_like(&self.queue);
    }

    /// Pushes one hop and returns the spectrum of the updated window.
    ///
    /// # Arguments
    /// * `hop`: `[batch, hop_size]` new samples.
    ///
    /// # Returns
    /// `[batch, n_bins, 2]` spectrum; the trailing axis is `(re, im)`.
    pub fn forward(
        &mut self,
        hop: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "hop_size"],
            &hop,
            &[("batch", self.batch_size()), ("hop_size", self.hop_size())],
        );

        let hop_size = self.hop_size();

        self.queue = if self.win_len() == hop_size {
            hop
        } else {
            let keep = self.queue.clone().slice(s![.., hop_size as isize..]);
            Tensor::cat(vec![keep, hop], 1)
        };

        // [batch, 1, win_len] -> [batch, 1, n_bins, 2] -> [batch, n_bins, 2]
        self.coef
            .analyze(self.queue.clone().unsqueeze_dim(1))
            .squeeze_dim(1)
    }

    /// Pushes `steps` consecutive hops at once.
    ///
    /// Equivalent to `steps` calls of [`forward`](Self::forward), but the
    /// frames are assembled with one unfold and analyzed with one batched
    /// FFT.
    ///
    /// # Arguments
    /// * `hops`: `[steps, batch, hop_size]` consecutive hops.
    ///
    /// # Returns
    /// `[steps, batch, n_bins, 2]` per-hop spectra; the trailing axis is
    /// `(re, im)`.
    pub fn forward_sequence(
        &mut self,
        hops: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        #[cfg(any(test, debug_assertions))]
        let [steps] = crate::contracts::unpack_shape_contract!(
            ["steps", "batch", "hop_size"],
            &hops,
            &["steps"],
            &[("batch", self.batch_size()), ("hop_size", self.hop_size())],
        );

        let win_len = self.win_len();
        let hop_size = self.hop_size();

        // [batch, steps * hop_size]
        let stream = hops.swap_dims(0, 1).flatten::<2>(1, 2);

        // [batch, win_len + steps * hop_size]
        let ext = Tensor::cat(vec![self.queue.clone(), stream], 1);

        self.queue = ext.clone().slice(s![.., -(win_len as isize)..]);

        // Unfolding yields `steps + 1` windows starting at multiples of
        // `hop_size`; window `s + 1` is the queue state after hop `s`, and
        // window 0 (the pre-push queue) is dropped.
        // [batch, steps, win_len]
        let frames: Tensor<B, 3> = ext.unfold(1, win_len, hop_size).slice(s![.., 1..]);

        // [batch, steps, n_bins, 2] -> [steps, batch, n_bins, 2]
        let x = self.coef.analyze(frames).swap_dims(0, 1);

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["steps", "batch", "n_bins", 2],
            &x,
            &[
                ("steps", steps),
                ("batch", self.batch_size()),
                ("n_bins", self.n_bins())
            ],
        );
        x
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
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    #[test]
    fn test_config_meta() {
        let cfg = SlidingStftConfig::new();
        assert_eq!(cfg.win_len, 768);
        assert_eq!(cfg.hop_size, 256);
        assert_eq!(cfg.fft_size, 1024);
        assert_eq!(cfg.window, SlidingStftWindow::Hann);

        assert_eq!(cfg.win_len(), 768);
        assert_eq!(cfg.hop_size(), 256);
        assert_eq!(cfg.fft_size(), 1024);
        assert_eq!(cfg.n_bins(), 513);

        cfg.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        for bad in [
            SlidingStftConfig::new().with_hop_size(0),
            // hop_size > win_len
            SlidingStftConfig::new().with_hop_size(1024),
            // win_len > fft_size
            SlidingStftConfig::new().with_win_len(2048),
            // non-power-of-two fft_size (odd)
            SlidingStftConfig::new()
                .with_fft_size(1023)
                .with_win_len(768),
            // non-power-of-two fft_size (even)
            SlidingStftConfig::new()
                .with_fft_size(1536)
                .with_win_len(768),
        ] {
            assert!(
                matches!(bad.validate(), Err(BunsenError::Invalid(_))),
                "expected Invalid: {bad:?}",
            );
        }
    }

    #[test]
    fn test_window_coefficients() {
        assert_eq!(SlidingStftWindow::Ones.coefficients(4), vec![1.0; 4]);

        let hann = SlidingStftWindow::Hann.coefficients(768);

        // Spot-check against the reference `AUP_AED_STFTWindow_Hann768`
        // table (`coeff.h`): 0.0, 1.6733041e-05, 6.6931045e-05, 1.5059065e-04.
        let expected = [0.0000000e+00f32, 1.6733041e-05, 6.693104e-05, 1.5059065e-04];
        for (n, e) in expected.into_iter().enumerate() {
            assert!((hann[n] - e).abs() <= 1e-8, "hann[{n}]: {} vs {e}", hann[n]);
        }

        // The periodic window satisfies w[n] == w[win_len - n].
        for n in 1..768 {
            assert!((hann[n] - hann[768 - n]).abs() <= 1e-7);
        }
    }

    #[test]
    fn test_hamming_coefficients() {
        let hamming = SlidingStftWindow::Hamming.coefficients(768);

        // Periodic Hamming anchors: `0.54 - 0.46` at n = 0, and the peak
        // `0.54 + 0.46` at the (even) midpoint.
        assert!((hamming[0] - 0.08).abs() <= 1e-7);
        assert!((hamming[384] - 1.0).abs() <= 1e-7);

        // The periodic window satisfies w[n] == w[win_len - n].
        for n in 1..768 {
            assert!((hamming[n] - hamming[768 - n]).abs() <= 1e-7);
        }

        // Anchor against the standard table value at win_len = 8, n = 1:
        // 0.54 - 0.46 * cos(π / 4).
        let hamming8 = SlidingStftWindow::Hamming.coefficients(8);
        assert!((hamming8[1] - 0.21473088).abs() <= 1e-7);
    }

    #[test]
    fn test_coefficients_tensor_matches_host() {
        let device = Default::default();
        for window in [
            SlidingStftWindow::Ones,
            SlidingStftWindow::Hann,
            SlidingStftWindow::Hamming,
        ] {
            let host = window.coefficients(48);
            let tensor: Tensor<B, 1> = window.coefficients_tensor(48, &device);
            tensor
                .to_data()
                .assert_approx_eq::<F>(&TensorData::new(host, [48]), Tolerance::permissive());
        }
    }

    #[test]
    fn test_init_meta_matches_config() {
        let device = Default::default();
        let cfg = SlidingStftConfig::new()
            .with_win_len(48)
            .with_hop_size(16)
            .with_fft_size(64);

        let coef: SlidingStft<B> = cfg.init(&device);

        assert_eq!(coef.win_len(), cfg.win_len());
        assert_eq!(coef.hop_size(), cfg.hop_size());
        assert_eq!(coef.fft_size(), cfg.fft_size());
        assert_eq!(coef.n_bins(), cfg.n_bins());

        assert_eq!(coef.window.dims(), [48]);

        let stft = coef.init_state(3);

        assert_eq!(stft.win_len(), cfg.win_len());
        assert_eq!(stft.hop_size(), cfg.hop_size());
        assert_eq!(stft.fft_size(), cfg.fft_size());
        assert_eq!(stft.n_bins(), cfg.n_bins());
        assert_eq!(stft.batch_size(), 3);

        // The queue starts zeroed.
        assert_eq!(stft.queue.dims(), [3, 48]);
        stft.queue
            .to_data()
            .assert_eq(&Tensor::<B, 2>::zeros([3, 48], &device).to_data(), true);
    }

    #[test]
    #[should_panic(expected = "batch_size must be non-zero")]
    fn test_init_state_rejects_zero_batch() {
        let device = Default::default();
        let coef: SlidingStft<B> = SlidingStftConfig::new().init(&device);
        coef.init_state(0);
    }

    /// Host-side reference: a sliding queue and naive windowed real DFT.
    struct HostStft {
        win_len: usize,
        hop_size: usize,
        fft_size: usize,
        window: Vec<f32>,
        queue: Vec<f64>,
    }

    impl HostStft {
        fn new(cfg: &SlidingStftConfig) -> Self {
            Self {
                win_len: cfg.win_len,
                hop_size: cfg.hop_size,
                fft_size: cfg.fft_size,
                window: cfg.window.coefficients(cfg.win_len),
                queue: vec![0.0; cfg.win_len],
            }
        }

        /// Pushes one hop; returns interleaved `(re, im)` pairs per bin.
        fn push(
            &mut self,
            hop: &[f32],
        ) -> Vec<f32> {
            assert_eq!(hop.len(), self.hop_size);
            self.queue.drain(..self.hop_size);
            self.queue.extend(hop.iter().map(|&v| v as f64));

            let n_bins = self.fft_size / 2 + 1;
            let mut out = Vec::with_capacity(2 * n_bins);
            for k in 0..n_bins {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for n in 0..self.win_len {
                    let x = self.queue[n] * self.window[n] as f64;
                    let theta = core::f64::consts::TAU * ((n * k) % self.fft_size) as f64
                        / self.fft_size as f64;
                    re += x * theta.cos();
                    im -= x * theta.sin();
                }
                out.push(re as f32);
                out.push(im as f32);
            }
            out
        }
    }

    /// A deterministic pseudo-random sample value.
    fn sample(
        batch: usize,
        step: usize,
        idx: usize,
    ) -> f32 {
        ((batch * 7919 + step * 104729 + idx * 1299709) % 1000) as f32 - 500.0
    }

    #[test]
    fn test_forward_matches_naive_dft() {
        let device = Default::default();
        let cfg = SlidingStftConfig::new()
            .with_win_len(48)
            .with_hop_size(16)
            .with_fft_size(64);
        let batch = 2;
        let n_bins = cfg.n_bins();

        let mut stft = cfg.init::<B>(&device).init_state(batch);
        let mut hosts: Vec<HostStft> = (0..batch).map(|_| HostStft::new(&cfg)).collect();

        // The first pushes cover the zero-padded warmup; the later ones a
        // full queue.
        for step in 0..5 {
            let rows: Vec<Vec<f32>> = (0..batch)
                .map(|b| (0..cfg.hop_size).map(|i| sample(b, step, i)).collect())
                .collect();

            let hop = Tensor::<B, 2>::from_data(
                TensorData::new(rows.concat(), [batch, cfg.hop_size]),
                &device,
            );
            let out = stft.forward(hop);
            assert_eq!(out.dims(), [batch, n_bins, 2]);

            let expected: Vec<f32> = hosts
                .iter_mut()
                .zip(&rows)
                .flat_map(|(host, row)| host.push(row))
                .collect();

            out.to_data().assert_approx_eq::<F>(
                &TensorData::new(expected, [batch, n_bins, 2]),
                Tolerance::permissive(),
            );
        }
    }

    #[test]
    fn test_forward_sequence_matches_stepwise() {
        let device = Default::default();
        let steps = 7;
        let batch = 2;

        for cfg in [
            SlidingStftConfig::new()
                .with_win_len(48)
                .with_hop_size(16)
                .with_fft_size(64),
            // win_len == hop_size: no overlap between frames.
            SlidingStftConfig::new()
                .with_win_len(16)
                .with_hop_size(16)
                .with_fft_size(16),
        ] {
            let hops = Tensor::<B, 3>::random(
                [steps, batch, cfg.hop_size],
                Distribution::Default,
                &device,
            );

            let mut seq_stft = cfg.init::<B>(&device).init_state(batch);
            let mut step_stft = seq_stft.clone();

            let seq_out = seq_stft.forward_sequence(hops.clone());
            assert_eq!(seq_out.dims(), [steps, batch, cfg.n_bins(), 2]);

            let mut step_outs = Vec::with_capacity(steps);
            for step in 0..steps {
                let hop = hops.clone().slice_dim(0, step).squeeze_dim::<2>(0);
                step_outs.push(step_stft.forward(hop));
            }
            let step_out: Tensor<B, 4> = Tensor::stack(step_outs, 0);

            let tol = Tolerance::<F>::permissive();
            seq_out
                .to_data()
                .assert_approx_eq::<F>(&step_out.to_data(), tol);
            seq_stft
                .queue
                .to_data()
                .assert_approx_eq::<F>(&step_stft.queue.to_data(), tol);
        }
    }

    #[test]
    fn test_reset() {
        let device = Default::default();
        let cfg = SlidingStftConfig::new()
            .with_win_len(48)
            .with_hop_size(16)
            .with_fft_size(64);

        let mut stft = cfg.init::<B>(&device).init_state(1);
        let hop = Tensor::random([1, 16], Distribution::Default, &device);
        stft.forward(hop);

        stft.reset();
        stft.queue
            .to_data()
            .assert_eq(&Tensor::<B, 2>::zeros([1, 48], &device).to_data(), true);
    }
}
