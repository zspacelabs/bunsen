//! # Decimating FIR filtering, as one GEMM.
//!
//! Filters and downsamples in a single contraction:
//!
//! ```text
//! y[m] = sum(h[k] * u[decimation * m - k] for k in 0..taps)
//! ```
//!
//! The decimation is folded into the kernel rather than applied afterwards, so
//! the discarded samples are never computed. Blocks are processed with a
//! carried history, so a stream may be filtered in hop-sized pieces and get the
//! same answer as filtering it whole.
//!
//! ## Why this shape, and not the obvious ones
//!
//! **Not an IIR recurrence.** When the filter you actually want is an IIR
//! cascade, realizing it as a truncated impulse response is usually the better
//! move on a device — see [`BiquadCascade`](super::BiquadCascade), whose
//! [`to_vec_impulse_response`](super::BiquadCascade::to_vec_impulse_response)
//! feeds straight into [`DecimatingFirConfig::try_init`]. A cascade is a
//! sample-rate recurrence, so it is sequential in the sample axis; the FIR form
//! is one parallel contraction.
//!
//! That is not a speed-for-accuracy trade, which is the surprising part. A
//! truncated FIR is a *better-conditioned realization of the same LTI system*,
//! not an approximation of it: measured against an `f64` ground truth, an FIR
//! at 2048 taps was 25-50% **more** accurate than the `f32` Direct-Form-II
//! cascade it replaced, while truncation contributed three orders of magnitude
//! less error than the thing it replaced. Pick `taps` from the impulse
//! response's L1 tail, not from its peak decay.
//!
//! **Not a long `conv1d`.** `burn-flex`'s depthwise convolution path fires when
//! `channels == groups == 1` and parallelizes over `batch * channels`, which is
//! then a single task — a long convolution would run single-threaded. On
//! cubecl, `groups != 1` disables every GEMM kernel outright. Reshaping the
//! windows and contracting against a constant matrix hits a threaded,
//! vectorized GEMM instead.
//!
//! ## Cost
//!
//! The Toeplitz matrix is `[window_len, hop_size / decimation]` and constant.
//! The transient is the materialized window stack, `batch * steps * window_len`
//! floats, which `matmul` makes contiguous — so chunk long inputs rather than
//! handing over an entire stream at once. Kernel selection is keyed on shape,
//! so a fixed chunk size is also tuned once instead of once per input length.

use burn::{
    config::Config,
    prelude::*,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    WithOkOrPanic,
};

/// Config for [`DecimatingFir`].
///
/// Describes the geometry only; the impulse response itself is supplied to
/// [`try_init`](Self::try_init), since it is data rather than configuration.
#[derive(Config, Debug, Copy)]
pub struct DecimatingFirConfig {
    /// Samples of input consumed per call, per batch row.
    pub hop_size: usize,

    /// Length of the impulse response.
    pub taps: usize,

    /// Downsampling factor; `1` filters without decimating.
    #[config(default = "1")]
    pub decimation: usize,
}

impl DecimatingFirConfig {
    /// Validates the geometry.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the hop does not decimate evenly, if the
    /// decimation is zero, or if the response is too short to be meaningful
    /// against it.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.decimation == 0 {
            return Err(BunsenError::Invalid(
                "DecimatingFir decimation must be non-zero".to_string(),
            ));
        }
        if self.hop_size == 0 || !self.hop_size.is_multiple_of(self.decimation) {
            return Err(BunsenError::Invalid(format!(
                "DecimatingFir hop_size ({}) must be a non-zero multiple of the \
                 decimation ({})",
                self.hop_size, self.decimation,
            )));
        }
        if self.taps <= self.decimation {
            return Err(BunsenError::Invalid(format!(
                "DecimatingFir taps ({}) must exceed the decimation ({})",
                self.taps, self.decimation,
            )));
        }
        Ok(())
    }

    /// Outputs produced per hop.
    pub fn out_per_hop(&self) -> usize {
        self.hop_size / self.decimation
    }

    /// Input samples carried between calls.
    ///
    /// This is `taps - 1`, **not** `window_len - hop_size`. Those differ by
    /// `decimation - 1`, and using the latter would both drop the oldest taps
    /// and desync the carry from the kernel's offset. Both quantities are
    /// derived from `taps` here so they cannot drift apart.
    pub fn carry_len(&self) -> usize {
        self.taps - 1
    }

    /// The window length one hop's outputs are contracted from.
    ///
    /// Output `m` of a hop reads `u[decimation * m - k]` for `k < taps`, so the
    /// window spans from `taps - 1` samples before the hop to
    /// `decimation * (out_per_hop - 1) + 1` after its start.
    pub fn window_len(&self) -> usize {
        self.hop_size + self.taps - self.decimation
    }

    /// The `[window_len, out_per_hop]` decimating Toeplitz matrix, row-major.
    ///
    /// Column `m` is the impulse response positioned so the contraction lands
    /// output `m` at input phase `decimation * m`.
    ///
    /// # Arguments
    /// * `response`: the impulse response, [`taps`](Self::taps) long.
    ///
    /// # Panics
    /// If `response` is not `taps` long.
    pub fn to_vec_toeplitz(
        &self,
        response: &[f32],
    ) -> Vec<f32> {
        assert_eq!(
            response.len(),
            self.taps,
            "DecimatingFir expects a {}-tap response",
            self.taps,
        );

        let carry = self.carry_len();
        let rows = self.window_len();
        let cols = self.out_per_hop();

        let mut out = vec![0.0f32; rows * cols];
        for m in 0..cols {
            let head = self.decimation * m + carry;
            for j in 0..rows {
                if head >= j && head - j < self.taps {
                    out[j * cols + m] = response[head - j];
                }
            }
        }
        out
    }

    /// Builds the filter around an impulse response.
    ///
    /// # Errors
    /// See [`validate`](Self::validate); also
    /// [`BunsenError::Invalid`] if `response` is not [`taps`](Self::taps) long.
    pub fn try_init<B: Backend>(
        &self,
        response: &[f32],
        device: &B::Device,
    ) -> BunsenResult<DecimatingFir<B>> {
        self.validate()?;
        if response.len() != self.taps {
            return Err(BunsenError::Invalid(format!(
                "DecimatingFir response has {} taps, config declares {}",
                response.len(),
                self.taps,
            )));
        }

        let toeplitz = Tensor::from_data(
            TensorData::new(
                self.to_vec_toeplitz(response),
                [self.window_len(), self.out_per_hop()],
            ),
            device,
        );

        Ok(DecimatingFir {
            cfg: *self,
            toeplitz,
        })
    }

    /// Builds the filter, panicking on error.
    pub fn init<B: Backend>(
        &self,
        response: &[f32],
        device: &B::Device,
    ) -> DecimatingFir<B> {
        self.try_init(response, device).ok_or_panic()
    }
}

/// A decimating FIR filter as a constant Toeplitz matrix.
///
/// Stateless; the carried history is passed through
/// [`forward`](Self::forward). Built by [`DecimatingFirConfig::try_init`].
#[derive(Debug, Clone)]
pub struct DecimatingFir<B: Backend> {
    cfg: DecimatingFirConfig,

    /// `[window_len, out_per_hop]` decimating kernel.
    pub toeplitz: Tensor<B, 2>,
}

impl<B: Backend> DecimatingFir<B> {
    /// The geometry this filter was built for.
    pub fn config(&self) -> &DecimatingFirConfig {
        &self.cfg
    }

    /// A zeroed `[batch, carry_len]` start-of-stream history.
    pub fn init_history(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        Tensor::zeros([batch_size, self.cfg.carry_len()], device)
    }

    /// Filters and decimates a run of hops.
    ///
    /// # Arguments
    /// * `input`: `[batch, steps * hop_size]` samples.
    /// * `history`: `[batch, carry_len]` from the previous call, or
    ///   [`init_history`](Self::init_history).
    ///
    /// # Returns
    /// `([batch, steps * out_per_hop]` outputs, the next history`)`.
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        history: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, len] = input.dims();
        let hop = self.cfg.hop_size;

        #[cfg(any(test, debug_assertions))]
        {
            assert_eq!(
                len % hop,
                0,
                "DecimatingFir input must be a whole number of hops",
            );
            crate::contracts::assert_shape_contract!(
                ["batch", "carry"],
                &history,
                &[("batch", batch), ("carry", self.cfg.carry_len())],
            );
        }

        let steps = len / hop;
        let window = self.cfg.window_len();

        // The carried history in front, so window `s` starts `carry_len`
        // samples before hop `s`.
        let extended = Tensor::cat(vec![history, input], 1);

        // On CubeCL, a vectorized `unfold` truncates its outer stride to a
        // multiple of the line width, so every row after the first is read
        // early whenever the unfolded axis is not a whole number of lines --
        // which it is not here, by `carry_len - (window - hop)` samples.
        // Trimming to the span the windows actually cover fixes it, because
        // the line width divides both `window` and `hop` and therefore divides
        // that span. The trimmed tail is not lost: it is still part of the
        // carry below.
        //
        // `burner::tensor::burn_behavior` pins the upstream behavior; if a
        // future burn honours the stride, that test fails loudly rather than
        // leaving dead defensive code here.
        let covered = (steps - 1) * hop + window;

        // [batch, steps, window]
        let windows = extended
            .clone()
            .slice_dim(1, 0..covered as isize)
            .unfold::<3, _>(1, window, hop);

        let out = windows
            .reshape([batch * steps, window])
            .matmul(self.toeplitz.clone())
            .reshape([batch, steps * self.cfg.out_per_hop()]);

        let next_history = extended.slice_dim(1, -(self.cfg.carry_len() as isize)..);

        (out, next_history)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    const HOP: usize = 32;
    const TAPS: usize = 12;
    const DECIM: usize = 4;

    fn cfg() -> DecimatingFirConfig {
        DecimatingFirConfig::new(HOP, TAPS).with_decimation(DECIM)
    }

    /// A deterministic, non-symmetric impulse response.
    fn response(taps: usize) -> Vec<f32> {
        (0..taps)
            .map(|k| {
                let t = k as f32;
                (-(t) / 5.0).exp() * (0.7 + (t * 0.9).sin())
            })
            .collect()
    }

    fn signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                let t = n as f32;
                (t * 0.31).sin() + 0.4 * (t * 1.17).cos()
            })
            .collect()
    }

    /// `y[m] = sum(h[k] * u[decim * m - k])`, computed directly on the host.
    ///
    /// The definition the GEMM has to reproduce. Indices before the start of
    /// the stream read zero, matching a zeroed initial history.
    fn direct(
        u: &[f32],
        h: &[f32],
        decimation: usize,
    ) -> Vec<f32> {
        let outs = u.len() / decimation;
        (0..outs)
            .map(|m| {
                let head = decimation * m;
                h.iter()
                    .enumerate()
                    .filter(|(k, _)| *k <= head)
                    .map(|(k, hk)| hk * u[head - k])
                    .sum()
            })
            .collect()
    }

    fn to_vec(t: Tensor<B, 2>) -> Vec<f32> {
        t.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic()
    }

    #[test]
    fn test_config_meta() {
        let c = cfg();
        assert_eq!(c.out_per_hop(), HOP / DECIM);
        assert_eq!(c.carry_len(), TAPS - 1);
        assert_eq!(c.window_len(), HOP + TAPS - DECIM);
        c.validate().unwrap();

        // Decimation of 1 is a plain convolution, and still valid.
        DecimatingFirConfig::new(HOP, TAPS).validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        for bad in [
            cfg().with_decimation(0),
            DecimatingFirConfig::new(0, TAPS).with_decimation(DECIM),
            // 30 is not a multiple of 4.
            DecimatingFirConfig::new(30, TAPS).with_decimation(DECIM),
            // A response no longer than the decimation cannot span a phase.
            DecimatingFirConfig::new(HOP, DECIM).with_decimation(DECIM),
        ] {
            assert!(
                matches!(bad.validate(), Err(BunsenError::Invalid(_))),
                "expected Invalid: {bad:?}",
            );
        }
    }

    #[test]
    fn test_init_rejects_a_mismatched_response() {
        let device = Default::default();
        assert!(matches!(
            cfg().try_init::<B>(&response(TAPS + 1), &device),
            Err(BunsenError::Invalid(_)),
        ));
    }

    #[test]
    fn test_matches_direct_convolution() {
        // The anchor: the GEMM must equal the definition, for an arbitrary
        // response. Nothing here refers to any particular filter.
        let device = Default::default();
        let steps = 3;
        let h = response(TAPS);
        let u = signal(steps * HOP);

        let fir = cfg().init::<B>(&h, &device);
        let input =
            Tensor::<B, 2>::from_data(TensorData::new(u.clone(), [1, steps * HOP]), &device);
        let (out, _) = fir.forward(input, fir.init_history(1, &device));

        let got = to_vec(out);
        let want = direct(&u, &h, DECIM);
        assert_eq!(got.len(), want.len());

        let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (m, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5 * peak, "output {m}: got {g}, want {w}",);
        }
    }

    #[test]
    fn test_decimation_of_one_is_plain_convolution() {
        let device = Default::default();
        let h = response(TAPS);
        let u = signal(2 * HOP);

        let c = DecimatingFirConfig::new(HOP, TAPS);
        let fir = c.init::<B>(&h, &device);
        let input = Tensor::<B, 2>::from_data(TensorData::new(u.clone(), [1, 2 * HOP]), &device);
        let (out, _) = fir.forward(input, fir.init_history(1, &device));

        let got = to_vec(out);
        let want = direct(&u, &h, 1);
        assert_eq!(got.len(), 2 * HOP);

        let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (m, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5 * peak, "output {m}: {g} vs {w}");
        }
    }

    #[test]
    fn test_unit_kernel_selects_every_nth_sample() {
        // With `h = [1, 0, 0, ...]` the filter is the identity, so the output
        // is exactly the decimated input. A phase error shows up immediately.
        let device = Default::default();
        let mut h = vec![0.0f32; TAPS];
        h[0] = 1.0;

        let u = signal(2 * HOP);
        let fir = cfg().init::<B>(&h, &device);
        let input = Tensor::<B, 2>::from_data(TensorData::new(u.clone(), [1, 2 * HOP]), &device);
        let (out, _) = fir.forward(input, fir.init_history(1, &device));

        let got = to_vec(out);
        let want: Vec<f32> = u.iter().step_by(DECIM).copied().collect();
        assert_eq!(got.len(), want.len());
        for (m, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "output {m}: {g} vs {w}");
        }
    }

    #[test]
    fn test_streaming_matches_a_single_call() {
        // The property that makes the carry correct: block boundaries must not
        // be observable.
        let device = Default::default();
        let steps = 4;
        let h = response(TAPS);
        let u = signal(steps * HOP);
        let fir = cfg().init::<B>(&h, &device);

        let whole =
            Tensor::<B, 2>::from_data(TensorData::new(u.clone(), [1, steps * HOP]), &device);
        let (out_whole, _) = fir.forward(whole, fir.init_history(1, &device));

        let mut history = fir.init_history(1, &device);
        let mut pieces = Vec::new();
        for chunk in u.chunks(HOP) {
            let t = Tensor::<B, 2>::from_data(TensorData::new(chunk.to_vec(), [1, HOP]), &device);
            let (y, next) = fir.forward(t, history);
            history = next;
            pieces.push(y);
        }
        let out_split = Tensor::cat(pieces, 1);

        out_whole
            .to_data_as::<f32>()
            .assert_approx_eq::<f32>(&out_split.to_data_as::<f32>(), Tolerance::permissive());
    }

    #[test]
    fn test_batch_rows_are_independent() {
        // Also the regression for burn's `unfold` row-stride bug: before the
        // trim, row 0 was exact while every later row was displaced. The filter
        // is linear, so a scaled row must give a scaled result.
        let device = Default::default();
        let steps = 3;
        let h = response(TAPS);
        let a = signal(steps * HOP);
        let b: Vec<f32> = a.iter().map(|v| -0.5 * v).collect();

        let mut flat = a.clone();
        flat.extend_from_slice(&b);
        let input = Tensor::<B, 2>::from_data(TensorData::new(flat, [2, steps * HOP]), &device);

        let fir = cfg().init::<B>(&h, &device);
        let (out, _) = fir.forward(input, fir.init_history(2, &device));
        let got = to_vec(out);

        let per = got.len() / 2;
        let peak = got[..per].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for i in 0..per {
            let want = -0.5 * got[i];
            assert!(
                (got[per + i] - want).abs() < 1e-5 * peak,
                "row 1 sample {i}: {} vs {want}",
                got[per + i],
            );
        }
    }

    #[test]
    fn test_history_carries_the_last_taps_minus_one_samples() {
        let device = Default::default();
        let h = response(TAPS);
        let u = signal(HOP);
        let fir = cfg().init::<B>(&h, &device);

        let input = Tensor::<B, 2>::from_data(TensorData::new(u.clone(), [1, HOP]), &device);
        let (_, history) = fir.forward(input, fir.init_history(1, &device));

        assert_eq!(history.dims(), [1, TAPS - 1]);
        let got = to_vec(history);
        let want = &u[HOP - (TAPS - 1)..];
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "carry {i}: {g} vs {w}");
        }
    }

    #[test]
    #[should_panic(expected = "whole number of hops")]
    fn test_partial_hop_is_rejected() {
        let device = Default::default();
        let fir = cfg().init::<B>(&response(TAPS), &device);
        let input = Tensor::<B, 2>::zeros([1, HOP + 1], &device);
        fir.forward(input, fir.init_history(1, &device));
    }
}
