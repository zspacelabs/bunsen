//! # Stage 2: whitening and decimation, on device.
//!
//! Turns raw hops plus this hop's whitening filter into the decimated
//! excitation history the lag search reads:
//!
//! ```text
//! raw [steps, batch, hop] + lpc [steps, batch, 16]  ->  exc [steps, batch, exc_len]
//! ```
//!
//! ## What the reference does
//!
//! ```text
//! aligned = raw delayed by XCORR_TRAINING_OFFSET
//! w[n]    = aligned[n] + Σ_{j<16} lpc[j]·aligned[n-1-j]     # whitening FIR
//! u[n]    = w[n] + 0.7·w[n-1]                               # 2-tap smoother
//! exc     = decimate(antialias(u), 4)
//! ```
//!
//! Whitening is what leaves a clean impulse train at the pitch period; the
//! smoother keeps the excitation from going fully impulsive.
//!
//! ## Carried state
//!
//! Four things, and one of them is smaller than it looks:
//!
//! * `input_q` — the raw FIFO. **This subsumes the reference's `pitch_mem`**:
//!   the whitening is a pure order-16 FIR over the *aligned* input rather than
//!   a recurrence, so the 16 samples it needs are already in the FIFO, sixteen
//!   positions behind the aligned window.
//! * `smoother` — the previous whitened sample.
//! * the anti-alias filter's own state, whose shape depends on which
//!   formulation is selected ([`super::antialias`]).
//! * `exc_buf` — the decimated history the lag search windows over.
//!
//! ## Both FIRs are shift-and-accumulate
//!
//! The whitening taps change every hop, which rules out a plain `conv1d`.
//! Written as `[hop, 17]` windows against a per-hop tap vector it would
//! materialize a `rows × 256 × 17` intermediate; written as 16 broadcast
//! multiply-adds over `[rows, hop]` slices it is far smaller *and* accumulates
//! in the reference's order, which makes it bit-exact rather than merely
//! close.
//!
//! ## Window alignment
//!
//! With the FIFO holding `max(offset, hop) + hop` samples, the aligned window
//! for hop `t` sits at `t·hop + 2·hop - offset` in the extended stream, and the
//! FIR needs 16 samples before it. Both the aligned windows and the excitation
//! windows are trimmed to their covered span before `unfold`, for the row
//! stride reason documented in [`super::antialias`].

use burn::{
    config::Config,
    prelude::*,
};

use super::{
    super::coeff::{
        LPC_ORDER,
        MAX_PERIOD_16KHZ,
        PROC_RESAMPLE_RATE,
        XCORR_TRAINING_OFFSET,
    },
    antialias::{
        PitchAntiAlias,
        PitchAntiAliasConfig,
        PitchAntiAliasState,
    },
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    kits::speech::ten_vad::context::coeff::HOP_SIZE,
    ops::signal::lpc_residual_batched,
};

/// The 2-tap smoother's feedback coefficient.
pub const SMOOTHER_COEFF: f32 = 0.7;

/// Config for [`PitchExcitation`].
#[derive(Config, Debug)]
pub struct PitchExcitationConfig {
    /// The hop size, in samples at 16 kHz.
    #[config(default = "HOP_SIZE")]
    pub hop_size: usize,

    /// How the anti-alias filter before decimation is realized.
    ///
    /// This is where the reference and optimized tiers diverge; see
    /// [`PitchAntiAliasConfig`].
    #[config(default = "PitchAntiAliasConfig::default()")]
    pub anti_alias: PitchAntiAliasConfig,
}

impl PitchExcitationConfig {
    /// The raw FIFO length.
    pub fn fifo_len(&self) -> usize {
        XCORR_TRAINING_OFFSET.max(self.hop_size) + self.hop_size
    }

    /// Where the aligned window starts, relative to a hop in the extended
    /// stream.
    pub fn aligned_offset(&self) -> usize {
        2 * self.hop_size - XCORR_TRAINING_OFFSET
    }

    /// The decimated excitation history length the lag search reads.
    pub fn exc_len(&self) -> usize {
        MAX_PERIOD_16KHZ / PROC_RESAMPLE_RATE + self.hop_size.div_ceil(PROC_RESAMPLE_RATE) + 1
    }

    /// How many decimated samples one hop contributes.
    pub fn exc_stride(&self) -> usize {
        self.hop_size / PROC_RESAMPLE_RATE
    }

    /// Validates the geometry.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the hop is too short to carry the
    /// correlation delay, or if the anti-alias geometry is invalid.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.hop_size < XCORR_TRAINING_OFFSET {
            return Err(BunsenError::Invalid(format!(
                "PitchExcitation hop_size ({}) must be at least the correlation delay ({})",
                self.hop_size, XCORR_TRAINING_OFFSET,
            )));
        }
        if self.exc_len() <= self.exc_stride() {
            return Err(BunsenError::Invalid(
                "PitchExcitation excitation history must exceed one hop".to_string(),
            ));
        }
        self.anti_alias.validate(self.hop_size)
    }

    /// Builds the stage.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<PitchExcitation<B>> {
        self.validate()?;
        Ok(PitchExcitation {
            hop_size: self.hop_size,
            fifo_len: self.fifo_len(),
            aligned_offset: self.aligned_offset(),
            exc_len: self.exc_len(),
            exc_stride: self.exc_stride(),
            anti_alias: self.anti_alias.try_init(self.hop_size, device)?,
        })
    }

    /// Builds the stage, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PitchExcitation<B> {
        self.try_init(device).ok_or_panic()
    }
}

/// The whitening and decimation stage.
///
/// Stateless coefficients; the carried buffers live in
/// [`PitchExcitationState`]. Built by [`PitchExcitationConfig::try_init`].
#[derive(Debug, Clone)]
pub struct PitchExcitation<B: Backend> {
    hop_size: usize,
    fifo_len: usize,
    aligned_offset: usize,
    exc_len: usize,
    exc_stride: usize,

    /// The anti-alias filter before decimation.
    pub anti_alias: PitchAntiAlias<B>,
}

/// The state [`PitchExcitation`] carries between calls.
#[derive(Debug, Clone)]
pub struct PitchExcitationState<B: Backend> {
    /// `[batch, fifo_len]` raw sample FIFO. Also supplies the whitening FIR's
    /// sample history, which is why there is no separate buffer for it.
    pub fifo: Tensor<B, 2>,

    /// `[batch, 1]` previous whitened sample, for the 2-tap smoother.
    pub smoother: Tensor<B, 2>,

    /// The anti-alias filter's carried state.
    pub anti_alias: PitchAntiAliasState<B>,

    /// `[batch, exc_len - exc_stride]` decimated history not yet consumed.
    pub exc_carry: Tensor<B, 2>,
}

impl<B: Backend> PitchExcitation<B> {
    /// The hop size, in samples at 16 kHz.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// The decimated excitation history length this stage emits.
    pub fn exc_len(&self) -> usize {
        self.exc_len
    }

    /// A zeroed start-of-stream state.
    pub fn init_state(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> PitchExcitationState<B> {
        PitchExcitationState {
            fifo: Tensor::zeros([batch_size, self.fifo_len], device),
            smoother: Tensor::zeros([batch_size, 1], device),
            anti_alias: self.anti_alias.init_state(batch_size, device),
            exc_carry: Tensor::zeros([batch_size, self.exc_len - self.exc_stride], device),
        }
    }

    /// Extracts the excitation history for a run of hops.
    ///
    /// # Arguments
    /// * `raw`: `[steps, batch, hop_size]` samples at the reference's int16
    ///   scale — **raw**, not pre-emphasized.
    /// * `lpc`: `[steps, batch, LPC_ORDER]` whitening filters, one per hop.
    /// * `state`: carried state, from a previous call or
    ///   [`init_state`](Self::init_state).
    ///
    /// # Returns
    /// `[steps, batch, exc_len]` excitation histories — the state after each
    /// hop, which is what the lag search consumes — and the state to carry
    /// forward.
    pub fn forward(
        &self,
        raw: Tensor<B, 3>,
        lpc: Tensor<B, 3>,
        state: PitchExcitationState<B>,
    ) -> (Tensor<B, 3>, PitchExcitationState<B>) {
        let [steps, batch, hop] = raw.dims();
        assert_eq!(hop, self.hop_size, "PitchExcitation hop mismatch");
        assert_eq!(
            lpc.dims(),
            [steps, batch, LPC_ORDER],
            "PitchExcitation lpc shape mismatch",
        );

        let PitchExcitationState {
            fifo,
            smoother,
            anti_alias,
            exc_carry,
        } = state;

        // Per-stream contiguous: [batch, steps * hop].
        let stream = raw.swap_dims(0, 1).flatten::<2>(1, 2);
        let extended = Tensor::cat(vec![fifo, stream], 1);

        let whitened = self.whiten(&extended, lpc, steps, batch);

        // [batch, steps * hop]
        let whitened_stream = whitened
            .reshape([steps, batch, hop])
            .swap_dims(0, 1)
            .flatten::<2>(1, 2);

        let (smoothed, smoother) = Self::smooth(whitened_stream, smoother);
        let (decimated, anti_alias) = self.anti_alias.forward(smoothed, anti_alias);

        let (windows, exc_carry) = self.window_excitation(decimated, exc_carry, steps, batch);

        let fifo = extended.slice_dim(1, -(self.fifo_len as isize)..);

        (
            windows,
            PitchExcitationState {
                fifo,
                smoother,
                anti_alias,
                exc_carry,
            },
        )
    }

    /// The order-16 whitening FIR, with per-hop taps.
    ///
    /// Accumulated one tap at a time in increasing `j`, matching the
    /// reference's order rather than a tree reduce's.
    fn whiten(
        &self,
        extended: &Tensor<B, 2>,
        lpc: Tensor<B, 3>,
        steps: usize,
        batch: usize,
    ) -> Tensor<B, 2> {
        let hop = self.hop_size;
        let rows = steps * batch;
        // The aligned window, plus the LPC_ORDER samples of history the FIR
        // reaches back into.
        let window = LPC_ORDER + hop;
        let start = self.aligned_offset - LPC_ORDER;
        let covered = (steps - 1) * hop + window;

        // [batch, steps, window] -> [steps, batch, window] -> [rows, window]
        let aligned = extended
            .clone()
            .slice_dim(1, start as isize..(start + covered) as isize)
            .unfold::<3, _>(1, window, hop)
            .swap_dims(0, 1)
            .reshape([rows, window]);

        lpc_residual_batched(aligned, lpc.reshape([rows, LPC_ORDER]))
    }

    /// The 2-tap smoother, `y[n] = w[n] + 0.7·w[n-1]`, carrying `w[-1]`.
    fn smooth(
        whitened: Tensor<B, 2>,
        carry: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let len = whitened.dims()[1] as isize;

        let previous = Tensor::cat(vec![carry, whitened.clone().slice_dim(1, 0..len - 1)], 1);
        let out = whitened.clone() + previous.mul_scalar(SMOOTHER_COEFF);
        let next_carry = whitened.slice_dim(1, len - 1..len);

        (out, next_carry)
    }

    /// Slides the decimated stream into per-hop excitation histories.
    fn window_excitation(
        &self,
        decimated: Tensor<B, 2>,
        carry: Tensor<B, 2>,
        steps: usize,
        batch: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let stride = self.exc_stride;
        let len = self.exc_len;
        let covered = (steps - 1) * stride + len;

        let extended = Tensor::cat(vec![carry, decimated], 1);

        let windows = extended
            .clone()
            .slice_dim(1, 0..covered as isize)
            .unfold::<3, _>(1, len, stride)
            .swap_dims(0, 1)
            .reshape([steps, batch, len]);

        let next_carry = extended.slice_dim(1, -((len - stride) as isize)..);

        (windows, next_carry)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::{
        super::super::{
            TenVadPitchEstimator,
            TenVadPitchScalarSource,
        },
        *,
    };
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;
    type D = <B as burn::tensor::backend::BackendTypes>::Device;

    const HOP: usize = 256;
    const N_BINS: usize = 513;

    fn pulse_hop(
        f0: f32,
        at: usize,
    ) -> Vec<f32> {
        let period = 16000.0 / f0;
        (0..HOP)
            .map(|i| {
                let pos = (at + i) as f32 % period;
                8000.0 * (-pos / (period * 0.08)).exp()
            })
            .collect()
    }

    fn spectrum(step: usize) -> Vec<f32> {
        (0..N_BINS)
            .map(|k| {
                let k = k as f32;
                1e7 * (-k / 70.0).exp() * (1.0 + 0.4 * (k * 0.05 + step as f32 * 0.3).sin())
            })
            .collect()
    }

    /// Drives the host and captures, per hop, the raw input, the filter it
    /// designed, and the excitation history it produced.
    fn host_reference(steps: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut est = TenVadPitchEstimator::new();
        let (mut raw, mut lpc, mut exc) = (Vec::new(), Vec::new(), Vec::new());

        for step in 0..steps {
            let hop = pulse_hop(150.0, step * HOP);
            est.frame_pitch(&hop, &spectrum(step));

            raw.extend_from_slice(&hop);
            lpc.extend_from_slice(est.lpc());
            exc.extend_from_slice(est.exc_buf());
        }
        (raw, lpc, exc)
    }

    fn config() -> PitchExcitationConfig {
        PitchExcitationConfig::new()
    }

    #[test]
    fn test_config_meta() {
        let cfg = config();
        assert_eq!(cfg.hop_size, 256);
        assert_eq!(cfg.fifo_len(), 512);
        assert_eq!(cfg.aligned_offset(), 432);
        assert_eq!(cfg.exc_len(), 129);
        assert_eq!(cfg.exc_stride(), 64);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        assert!(config().with_hop_size(64).validate().is_err());
        assert!(
            config()
                .with_anti_alias(PitchAntiAliasConfig::TruncatedFir { taps: 1 })
                .validate()
                .is_err()
        );
    }

    #[test]
    fn test_init_meta_matches_config() {
        let device = Default::default();
        let stage: PitchExcitation<B> = config().init(&device);
        assert_eq!(stage.hop_size(), 256);
        assert_eq!(stage.exc_len(), 129);
    }

    /// Runs the stage over a whole run of hops, against host-designed filters.
    fn run_stage(
        cfg: &PitchExcitationConfig,
        steps: usize,
        raw: &[f32],
        lpc: &[f32],
        device: &D,
    ) -> Vec<f32> {
        let stage: PitchExcitation<B> = cfg.init(device);
        let state = stage.init_state(1, device);

        let raw_t = Tensor::<B, 1>::from_floats(raw, device).reshape([steps, 1, HOP]);
        let lpc_t = Tensor::<B, 1>::from_floats(lpc, device).reshape([steps, 1, LPC_ORDER]);

        let (out, _) = stage.forward(raw_t, lpc_t, state);
        out.to_data_as::<f32>().to_vec_as::<f32>().unwrap()
    }

    #[test]
    fn test_forward_matches_host_stage() {
        let device = Default::default();
        let steps = 8;
        let (raw, lpc, want) = host_reference(steps);

        for anti_alias in [
            PitchAntiAliasConfig::Recurrence,
            PitchAntiAliasConfig::default(),
        ] {
            let cfg = config().with_anti_alias(anti_alias);
            let got = run_stage(&cfg, steps, &raw, &lpc, &device);

            assert_eq!(got.len(), want.len(), "{anti_alias:?}");
            let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() / peak < 5e-6,
                    "{anti_alias:?} sample {i}: {g} vs {w} (peak {peak})",
                );
            }
        }
    }

    #[test]
    fn test_forward_sequence_matches_stepwise() {
        let device = Default::default();
        let steps = 6;
        let (raw, lpc, _) = host_reference(steps);
        let cfg = config();

        let whole = run_stage(&cfg, steps, &raw, &lpc, &device);

        let stage: PitchExcitation<B> = cfg.init(&device);
        let mut state = stage.init_state(1, &device);
        let mut stepwise = Vec::new();
        for step in 0..steps {
            let r = Tensor::<B, 1>::from_floats(&raw[step * HOP..(step + 1) * HOP], &device)
                .reshape([1, 1, HOP]);
            let l = Tensor::<B, 1>::from_floats(
                &lpc[step * LPC_ORDER..(step + 1) * LPC_ORDER],
                &device,
            )
            .reshape([1, 1, LPC_ORDER]);
            let (out, next) = stage.forward(r, l, state);
            state = next;
            stepwise.extend(out.to_data_as::<f32>().to_vec_as::<f32>().unwrap());
        }

        TensorData::from(whole.as_slice()).assert_approx_eq::<f32>(
            &TensorData::from(stepwise.as_slice()),
            Tolerance::relative(1e-5),
        );
    }

    #[test]
    fn test_batch_rows_are_independent() {
        let device = Default::default();
        let steps = 4;
        let (raw, lpc, _) = host_reference(steps);
        let cfg = config();
        let stage: PitchExcitation<B> = cfg.init(&device);

        // Row 1 is a scaled copy, which the whitening FIR maps linearly.
        let raw_b: Vec<f32> = raw.iter().map(|v| -0.5 * v).collect();

        let mut interleaved = Vec::new();
        let mut lpc_pair = Vec::new();
        for step in 0..steps {
            interleaved.extend_from_slice(&raw[step * HOP..(step + 1) * HOP]);
            interleaved.extend_from_slice(&raw_b[step * HOP..(step + 1) * HOP]);
            for _ in 0..2 {
                lpc_pair.extend_from_slice(&lpc[step * LPC_ORDER..(step + 1) * LPC_ORDER]);
            }
        }

        let raw_t =
            Tensor::<B, 1>::from_floats(interleaved.as_slice(), &device).reshape([steps, 2, HOP]);
        let lpc_t = Tensor::<B, 1>::from_floats(lpc_pair.as_slice(), &device)
            .reshape([steps, 2, LPC_ORDER]);

        let (out, _) = stage.forward(raw_t, lpc_t, stage.init_state(2, &device));
        let got: Vec<f32> = out.to_data_as::<f32>().to_vec_as::<f32>().unwrap();

        let solo = run_stage(&cfg, steps, &raw, &lpc, &device);
        let len = stage.exc_len();
        let peak = solo.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        for step in 0..steps {
            for i in 0..len {
                let row0 = got[(step * 2) * len + i];
                let row1 = got[(step * 2 + 1) * len + i];
                let want0 = solo[step * len + i];
                assert!(
                    (row0 - want0).abs() / peak < 1e-6,
                    "step {step} sample {i}: row0 {row0} vs solo {want0}",
                );
                assert!(
                    (row1 - -0.5 * row0).abs() / peak < 1e-6,
                    "step {step} sample {i}: row1 {row1} is not -0.5x row0 {row0}",
                );
            }
        }
    }

    #[test]
    fn test_reset_rewinds_the_stream() {
        let device = Default::default();
        let steps = 3;
        let (raw, lpc, _) = host_reference(steps);
        let cfg = config();
        let stage: PitchExcitation<B> = cfg.init(&device);

        let raw_t = Tensor::<B, 1>::from_floats(raw.as_slice(), &device).reshape([steps, 1, HOP]);
        let lpc_t =
            Tensor::<B, 1>::from_floats(lpc.as_slice(), &device).reshape([steps, 1, LPC_ORDER]);

        let (first, state) =
            stage.forward(raw_t.clone(), lpc_t.clone(), stage.init_state(1, &device));
        // Continuing carries state, so the same input gives a different answer.
        let (carried, _) = stage.forward(raw_t.clone(), lpc_t.clone(), state);
        assert_ne!(
            first
                .clone()
                .to_data_as::<f32>()
                .to_vec_as::<f32>()
                .unwrap(),
            carried.to_data_as::<f32>().to_vec_as::<f32>().unwrap(),
        );

        let (again, _) = stage.forward(raw_t, lpc_t, stage.init_state(1, &device));
        first
            .to_data()
            .assert_approx_eq::<f32>(&again.to_data(), Tolerance::permissive());
    }
}
