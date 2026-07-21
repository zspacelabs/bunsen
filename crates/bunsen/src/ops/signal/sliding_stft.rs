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
//! with no normalization; it is computed with [`burn::tensor::signal::stft`].
//! The C reference emits the same spectrum with the imaginary parts negated
//! (an artifact of its FFTW half-complex packing); bin powers `re² + im²`
//! agree.
//!
//! Note: `stft` center-pads windows shorter than `n_fft`, while the ten-vad
//! layout puts the window at the frame start with zero-padding at the end;
//! the analyzer therefore carries its window pre-padded to `fft_size` and
//! feeds `stft` full-frame windows.
//!
//! Note: burn's `stft` (via the `rfft` beneath it) does not yet support
//! autodiff (the backward is unimplemented upstream), so this analyzer
//! cannot currently be differentiated through.

use burn::{
    prelude::*,
    tensor::{
        ops::PadMode,
        signal::{
            StftOptions,
            stft,
        },
    },
};

use crate::{
    burner::tensor::inplace_res,
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    ops::signal::{
        SamplingWindowBuilder,
        StftWindowConfig,
    },
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

    /// The analysis window config.
    #[config(default = "StftWindowConfig::Hann { periodic: true }")]
    pub window: StftWindowConfig,
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

        let window = self.window.to_tensor_window(win_len, device);

        // Right-pad the window to `fft_size`: `stft` consumes full-frame
        // windows, and the ten-vad layout puts the window at the frame
        // start with zero-padding at the end.
        let pad = self.fft_size - win_len;
        let window = if pad > 0 {
            window.pad([(0, pad)], PadMode::Constant(0.0))
        } else {
            window
        };

        Ok(SlidingStft {
            hop_size: self.hop_size,
            win_len,
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
    win_len: usize,

    /// The analysis window, right-padded with zeros from `win_len` to
    /// `[fft_size]`.
    ///
    /// This is the ten-vad frame layout (window at the frame start,
    /// zero-padding at the end), and the full-frame window layout consumed
    /// by [`stft`].
    pub window: Tensor<B, 1>,
}

impl<B: Backend> SlidingStftMeta for SlidingStft<B> {
    fn win_len(&self) -> usize {
        self.win_len
    }

    fn hop_size(&self) -> usize {
        self.hop_size
    }

    fn fft_size(&self) -> usize {
        self.window.dims()[0]
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

    /// Build the matching [`StftOptions`].
    pub fn to_options(&self) -> StftOptions {
        StftOptions {
            n_fft: self.fft_size(),
            hop_length: self.hop_size(),
            win_length: None,
            center: false,
            onesided: true,
        }
    }

    /// Allocate a zeroed window.
    pub fn zero_window(
        &self,
        batch_size: usize,
    ) -> Tensor<B, 2> {
        Tensor::zeros([batch_size, self.win_len()], &self.window.device())
    }

    /// Analyzes a signal into consecutive ten-vad-aligned STFT frames.
    ///
    /// Frame `f` covers `signal[f * hop_size .. f * hop_size + win_len]`,
    /// windowed and zero-padded to `fft_size`. Trailing samples that do not
    /// fill a whole window are ignored.
    ///
    /// Uses [`stft`], which does not yet support autodiff.
    ///
    /// # Arguments
    /// * `signal`: `[batch, samples]`, with `samples >= win_len`.
    ///
    /// # Returns
    /// `[batch, frames, n_bins, 2]` spectra, with
    /// `frames = 1 + (samples - win_len) / hop_size`; the trailing axis is
    /// `(re, im)`.
    pub fn analyze(
        &self,
        signal: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        #[cfg(any(test, debug_assertions))]
        let [batch, samples] = crate::contracts::unpack_shape_contract!(
            ["batch", "samples"],
            &signal,
            &["batch", "samples"],
        );
        #[cfg(any(test, debug_assertions))]
        assert!(
            samples >= self.win_len(),
            "SlidingStft samples ({samples}) must be >= win_len ({})",
            self.win_len(),
        );

        // Right-pad so the final window fills a whole `fft_size` stft
        // frame; the padding only ever lands under the zeroed window tail.
        let pad = self.fft_size() - self.win_len();
        let signal = if pad > 0 {
            signal.pad([(0, pad)], PadMode::Constant(0.0))
        } else {
            signal
        };

        // [batch, frames, n_bins, 2]
        let x = stft(signal, Some(self.window.clone()), self.to_options());

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "frames", "n_bins", 2],
            &x,
            &[
                ("batch", batch),
                ("frames", 1 + (samples - self.win_len()) / self.hop_size()),
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
        if self.win_len() == hop_size {
            self.queue = hop;
        } else {
            // Inplace, so we can drop the reference before the update.
            self.queue.inplace(|q| {
                let keep = q.slice_dim(1, hop_size as isize..);
                Tensor::cat(vec![keep, hop], 1)
            });
        }

        // One full window -> one frame.
        // [batch, 1, n_bins, 2] -> [batch, n_bins, 2]
        self.coef.analyze(self.queue.clone()).squeeze_dim(1)
    }

    /// Pushes `steps` consecutive hops at once.
    ///
    /// Equivalent to `steps` calls of [`forward`](Self::forward), but the
    /// whole extended stream is analyzed with a single [`stft`] call.
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

        // [batch, steps * hop_size]
        let stream = hops.swap_dims(0, 1).flatten::<2>(1, 2);

        // Inplace, so we can drop the reference before the update.
        let ext = inplace_res(&mut self.queue, |q| {
            let ext = Tensor::cat(vec![q, stream], 1);
            let q = ext.clone().slice_dim(1, -(win_len as isize)..);
            (q, ext)
        });

        // `analyze` yields `steps + 1` frames at `hop_size` offsets; frame
        // `s + 1` is the queue state after hop `s`, and frame 0 (the
        // pre-push queue) is dropped.
        // [batch, steps + 1, n_bins, 2] -> [steps, batch, n_bins, 2]
        let x = self.coef.analyze(ext).slice_dim(1, 1..).swap_dims(0, 1);

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
        DType,
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
        assert_eq!(cfg.window, StftWindowConfig::Hann { periodic: true });

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

        // The stored window is right-padded from win_len to fft_size, with
        // the coefficients at the frame start and zeros in the tail.
        assert_eq!(coef.window.dims(), [64]);
        let window: Vec<f64> = coef
            .window
            .clone()
            .cast(DType::F64)
            .to_data()
            .to_vec()
            .unwrap();
        let host = cfg.window.to_vec_window(48);
        for (n, (&w, &h)) in window.iter().zip(&host).enumerate() {
            assert!((w - h).abs() <= 1e-6, "window[{n}]: {w} vs {h}");
        }
        assert!(window[48..].iter().all(|&v| v == 0.0));

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
        window: Vec<f64>,
        queue: Vec<f64>,
    }

    impl HostStft {
        fn new(cfg: &SlidingStftConfig) -> Self {
            Self {
                win_len: cfg.win_len,
                hop_size: cfg.hop_size,
                fft_size: cfg.fft_size,
                window: cfg.window.to_vec_window(cfg.win_len),
                queue: vec![0.0; cfg.win_len],
            }
        }

        /// Pushes one hop; returns interleaved `(re, im)` pairs per bin.
        fn push(
            &mut self,
            hop: &[f64],
        ) -> Vec<f64> {
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
            let rows: Vec<Vec<f64>> = (0..batch)
                .map(|b| (0..cfg.hop_size).map(|i| sample(b, step, i)).collect())
                .collect();

            let hop = Tensor::<B, 2>::from_data(
                TensorData::new(rows.concat(), [batch, cfg.hop_size]),
                &device,
            );
            let out = stft.forward(hop);
            assert_eq!(out.dims(), [batch, n_bins, 2]);

            let expected: Vec<f64> = hosts
                .iter_mut()
                .zip(&rows)
                .flat_map(|(host, row)| host.push(row))
                .collect();

            out.cast(DType::F64).to_data().assert_approx_eq::<F>(
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
