//! # Stage 4: period tracking and the pitch estimate, on device.
//!
//! ```text
//! xcorr [steps, batch, 2, max_period] + energy [steps, batch, 2]
//!     ->  pitch [steps, batch, 1]   (Hz, 0 = unvoiced)
//! ```
//!
//! A Viterbi pass over the last `2·n_feat` half-hop slots picks a period path,
//! penalizing `PITCH_MAX_PATH_W · step²` for jumping between candidates. The
//! backtrace yields both a voicing score and a weighted least-squares fit over
//! the recovered periods, extrapolated half a frame forward.
//!
//! ## The one genuinely sequential stage
//!
//! The accumulator carries across hops and is renormalized by subtracting its
//! running maximum, so it cannot be cleared per hop and cannot be reassociated.
//! Two dependent steps per hop, each `[batch, dif_period]`.
//!
//! That is affordable in context: the driver already runs one sequential device
//! iteration per hop for the LSTM, and each of those dispatches far more work
//! than a step here does. This adds roughly twice the iteration count on
//! tensors three orders of magnitude smaller.
//!
//! **The backtrace, by contrast, is not sequential across hops.** At hop `t`,
//! backpointer row `sub` is absolute slot `2t + sub − 4`, so keeping the
//! histories in a flat, non-circular layout turns the whole backtrace into six
//! strided gathers rather than `6 × steps` scalar ones. That also deletes the
//! reference's circular-buffer bookkeeping outright.
//!
//! ## The candidate window is asymmetric, and wider than it looks
//!
//! The reference computes `SIDXT = min(0, 4 − idx)`, which is unbounded below:
//!
//! ```text
//! cand ∈ [ min(idx, 4), min(idx + 4, dif_period − 1) ]
//! ```
//!
//! so the window is 5 wide at the short-period end and **52 wide** at the long
//! end, where `jdx` reaches −51 and the penalty reaches 52. That is very likely
//! a bug in the reference, but it is load-bearing for output parity, so it is
//! reproduced exactly. It is also why the transition is a dense
//! `[dif_period, dif_period]` penalty matrix rather than a narrow band: invalid
//! transitions are simply given a penalty large enough to lose every `max`.

use burn::{
    config::Config,
    prelude::*,
};

use super::{
    super::coeff::{
        FEAT_MAX_NFRM,
        FEAT_TIME_WINDOW_MS,
        MAX_PERIOD_16KHZ,
        MIN_PERIOD_16KHZ,
        PITCH_MAX_PATH_W,
        PROC_FS,
        PROC_RESAMPLE_RATE,
        VOICED_THRESHOLD,
    },
    correlate::SUBS_PER_HOP,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    kits::speech::ten_vad::context::coeff::{
        HOP_SIZE,
        SAMPLE_RATE,
    },
};

/// The penalty assigned to a transition the reference would never consider.
///
/// Large enough to lose every `max` against the running floor, which is
/// `path_best_all - 1e10`, so validity falls out of the reduction instead of
/// needing a mask.
const INVALID_PENALTY: f32 = 1e30;

/// Config for [`PitchTrack`].
#[derive(Config, Debug, Copy)]
pub struct PitchTrackConfig {
    /// The hop size, in samples at 16 kHz.
    #[config(default = "HOP_SIZE")]
    pub hop_size: usize,
}

impl PitchTrackConfig {
    /// The longest candidate period, at the correlation rate.
    pub fn max_period(&self) -> usize {
        MAX_PERIOD_16KHZ / PROC_RESAMPLE_RATE
    }

    /// The width of the tracker's state space.
    pub fn dif_period(&self) -> usize {
        self.max_period() - MIN_PERIOD_16KHZ / PROC_RESAMPLE_RATE
    }

    /// How many whole hops of correlation history the tracker spans.
    pub fn n_feat(&self) -> usize {
        FEAT_MAX_NFRM.min(
            ((FEAT_TIME_WINDOW_MS * SAMPLE_RATE) as f32 / (self.hop_size * 1000) as f32).ceil()
                as usize,
        )
    }

    /// How many half-hop slots the tracker spans.
    pub fn slots(&self) -> usize {
        self.n_feat() * SUBS_PER_HOP
    }

    /// How many slots are carried between calls.
    pub fn carry_slots(&self) -> usize {
        self.slots() - SUBS_PER_HOP
    }

    /// Validates the geometry.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the hop yields fewer than two history slots
    /// or an empty period range.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.n_feat() < 2 {
            return Err(BunsenError::Invalid(format!(
                "PitchTrack hop_size ({}) yields {} history frames; at least 2 are needed",
                self.hop_size,
                self.n_feat(),
            )));
        }
        if self.dif_period() == 0 {
            return Err(BunsenError::Invalid(
                "PitchTrack period range is empty".to_string(),
            ));
        }
        Ok(())
    }

    /// The `[dif_period, dif_period]` transition penalty matrix, row-major.
    ///
    /// Entry `[idx][cand]` is `PITCH_MAX_PATH_W · jdx²` where reachable, and
    /// [`INVALID_PENALTY`] where not. The arithmetic order matches the
    /// reference's `(W · |j|) · |j|`.
    pub fn to_vec_penalty(&self) -> Vec<f32> {
        let dif = self.dif_period();
        let mut out = vec![INVALID_PENALTY; dif * dif];

        for idx in 0..dif {
            let first = 0.min(4 - idx as i32);
            for jdx in first..=4 {
                let cand = idx as i32 + jdx;
                if cand < 0 || cand as usize >= dif {
                    continue;
                }
                let magnitude = jdx.abs() as f32;
                out[idx * dif + cand as usize] = PITCH_MAX_PATH_W * magnitude * magnitude;
            }
        }
        out
    }

    /// Builds the tracker.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<PitchTrack<B>> {
        self.validate()?;
        let dif = self.dif_period();

        Ok(PitchTrack {
            max_period: self.max_period(),
            dif_period: dif,
            n_feat: self.n_feat(),
            penalty: Tensor::from_data(TensorData::new(self.to_vec_penalty(), [dif, dif]), device),
        })
    }

    /// Builds the tracker, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PitchTrack<B> {
        self.try_init(device).ok_or_panic()
    }
}

/// The period tracker.
///
/// Built by [`PitchTrackConfig::try_init`]; state lives in
/// [`PitchTrackState`].
#[derive(Debug, Clone)]
pub struct PitchTrack<B: Backend> {
    max_period: usize,
    dif_period: usize,
    n_feat: usize,

    /// `[dif_period, dif_period]` transition penalties.
    pub penalty: Tensor<B, 2>,
}

/// The state [`PitchTrack`] carries between calls.
#[derive(Debug, Clone)]
pub struct PitchTrackState<B: Backend> {
    /// `[batch, dif_period]` Viterbi accumulator, renormalized to peak zero.
    pub path_score: Tensor<B, 2>,

    /// `[batch, 1]` best score reached so far.
    pub path_best: Tensor<B, 2>,

    /// `[batch, 1]` best period index reached so far.
    pub best_period: Tensor<B, 2, Int>,

    /// `[batch, carry_slots, max_period]` correlations not yet aged out.
    pub slot_xcorr: Tensor<B, 3>,

    /// `[batch, carry_slots]` slot energies.
    pub slot_energy: Tensor<B, 2>,

    /// `[batch, carry_slots, dif_period]` backpointers.
    pub slot_prev: Tensor<B, 3, Int>,
}

impl<B: Backend> PitchTrack<B> {
    /// The width of the tracker's state space.
    pub fn dif_period(&self) -> usize {
        self.dif_period
    }

    /// How many half-hop slots the tracker spans.
    pub fn slots(&self) -> usize {
        self.n_feat * SUBS_PER_HOP
    }

    /// A zeroed start-of-stream state.
    pub fn init_state(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> PitchTrackState<B> {
        let carry = self.slots() - SUBS_PER_HOP;
        PitchTrackState {
            path_score: Tensor::zeros([batch_size, self.dif_period], device),
            path_best: Tensor::zeros([batch_size, 1], device),
            best_period: Tensor::zeros([batch_size, 1], device),
            slot_xcorr: Tensor::zeros([batch_size, carry, self.max_period], device),
            slot_energy: Tensor::zeros([batch_size, carry], device),
            slot_prev: Tensor::zeros([batch_size, carry, self.dif_period], device),
        }
    }

    /// Tracks a run of hops and reports their pitch.
    ///
    /// # Arguments
    /// * `xcorr`: `[steps, batch, 2, max_period]` normalized correlations.
    /// * `energy`: `[steps, batch, 2]` per-slot weights.
    /// * `state`: carried state.
    ///
    /// # Returns
    /// `[steps, batch, 1]` pitch in Hz, `0.0` where unvoiced, and the state to
    /// carry forward.
    pub fn forward(
        &self,
        xcorr: Tensor<B, 4>,
        energy: Tensor<B, 3>,
        state: PitchTrackState<B>,
    ) -> (Tensor<B, 3>, PitchTrackState<B>) {
        let [steps, batch, subs, max_period] = xcorr.dims();
        assert_eq!(subs, SUBS_PER_HOP, "PitchTrack expects two slots per hop");
        assert_eq!(max_period, self.max_period, "PitchTrack lag width mismatch");

        let slots = self.slots();
        let carry = slots - SUBS_PER_HOP;
        let dif = self.dif_period;
        let total_slots = carry + steps * SUBS_PER_HOP;

        // Flat, non-circular slot histories: [batch, carry + 2*steps, ..].
        let xcorr_hist = Tensor::cat(
            vec![
                state.slot_xcorr,
                xcorr
                    .swap_dims(0, 1)
                    .reshape([batch, steps * SUBS_PER_HOP, max_period]),
            ],
            1,
        );
        let energy_hist = Tensor::cat(
            vec![
                state.slot_energy,
                energy
                    .swap_dims(0, 1)
                    .reshape([batch, steps * SUBS_PER_HOP]),
            ],
            1,
        );

        // Each hop normalizes its own `slots`-wide window, so the weights are
        // per hop, not per slot: [batch, steps, slots].
        let weights = Self::normalize_weights(energy_hist.clone(), slots, steps);

        // --- forward pass: two dependent steps per hop ---
        let mut path_score = state.path_score;
        let mut path_best = state.path_best;
        let mut best_period = state.best_period;
        let mut new_prev = Vec::with_capacity(steps * SUBS_PER_HOP);
        let mut hop_best = Vec::with_capacity(steps);

        for step in 0..steps {
            for sub in 0..SUBS_PER_HOP {
                let slot = carry + step * SUBS_PER_HOP + sub;
                let xc = xcorr_hist
                    .clone()
                    .slice_dim(1, slot as isize..(slot + 1) as isize)
                    .squeeze_dim::<2>(1)
                    .slice_dim(1, 0..dif as isize);
                let w = weights
                    .clone()
                    .slice(s![
                        ..,
                        step as isize..(step + 1) as isize,
                        (carry + sub) as isize..(carry + sub + 1) as isize
                    ])
                    .reshape([batch, 1]);

                let (score, best, arg, prev) =
                    self.viterbi_step(path_score, path_best, best_period, xc, w);
                path_score = score;
                path_best = best;
                best_period = arg;
                new_prev.push(prev.unsqueeze_dim::<3>(1));
            }
            hop_best.push(best_period.clone());
        }

        let prev_hist = Tensor::cat(vec![state.slot_prev, Tensor::cat(new_prev, 1)], 1);

        // --- backtrace and fit, batched across hops ---
        // Stacked, not `cat` + `swap_dims`: the backtrace feeds this to
        // `gather` as an index, and `gather` ignores strides on a
        // non-contiguous index tensor -- it reads element 0 of every row
        // instead of the indexed one. `stack` builds `[steps, batch, 1]`
        // contiguously; a transposed view silently corrupts every hop after
        // the first. See `tests::test_gather_needs_a_contiguous_index`.
        let cursor: Tensor<B, 2, Int> = Tensor::stack::<3>(hop_best.clone(), 0).squeeze_dim::<2>(2);
        let pitch = self.backtrace_and_fit(
            &xcorr_hist,
            &prev_hist,
            &weights,
            cursor,
            steps,
            batch,
            total_slots,
        );

        let keep = total_slots - carry;
        (
            pitch.reshape([steps, batch, 1]),
            PitchTrackState {
                path_score,
                path_best,
                best_period,
                slot_xcorr: xcorr_hist.slice_dim(1, keep as isize..),
                slot_energy: energy_hist.slice_dim(1, keep as isize..),
                slot_prev: prev_hist.slice_dim(1, keep as isize..),
            },
        )
    }

    /// Scales each hop's window of slot energies to average one.
    ///
    /// The `1e-15` seed matches the reference and keeps an all-silent window
    /// from dividing by zero.
    ///
    /// The slot history is exactly `(steps - 1)·2 + slots` long, so this
    /// `unfold` has no leftover tail and needs no trim — unlike the ones in
    /// [`super::antialias`] and [`super::excitation`], where a tail would
    /// misplace every row after the first.
    fn normalize_weights(
        energy_hist: Tensor<B, 2>,
        slots: usize,
        steps: usize,
    ) -> Tensor<B, 3> {
        debug_assert_eq!(
            energy_hist.dims()[1],
            (steps - 1) * SUBS_PER_HOP + slots,
            "slot history must exactly cover its windows",
        );

        // [batch, steps, slots]
        let windows = energy_hist.unfold::<3, _>(1, slots, SUBS_PER_HOP);
        let total = windows.clone().sum_dim(2).add_scalar(1e-15f32);
        windows * (total.recip().mul_scalar(slots as f32))
    }

    /// One Viterbi step: the dense transition max, then the renormalization.
    fn viterbi_step(
        &self,
        path_score: Tensor<B, 2>,
        path_best: Tensor<B, 2>,
        best_period: Tensor<B, 2, Int>,
        xcorr: Tensor<B, 2>,
        weight: Tensor<B, 2>,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2, Int>,
        Tensor<B, 2, Int>,
    ) {
        // [batch, dif, dif]: score of arriving at `idx` from `cand`.
        let transitions =
            path_score.unsqueeze_dim::<3>(1) - self.penalty.clone().unsqueeze_dim::<3>(0);
        let (best_in, arg_in) = transitions.max_dim_with_indices(2);

        // The reference seeds its search at `path_best - 1e10` and keeps the
        // previous best period when nothing beats it; invalid transitions sit
        // far below that floor, so they never win.
        let floor = path_best.sub_scalar(1e10f32).unsqueeze_dim::<3>(2);
        let stalled = best_in.clone().lower_equal(floor.clone());
        let prev = arg_in.mask_where(stalled, best_period.clone().unsqueeze_dim::<3>(2));

        let scored = best_in.max_pair(floor).squeeze_dim::<2>(2) + xcorr * weight;
        let (top, arg_top) = scored.clone().max_dim_with_indices(1);

        (scored - top.clone(), top, arg_top, prev.squeeze_dim::<2>(2))
    }

    /// Walks each hop's path back six slots and fits a period contour.
    #[allow(clippy::too_many_arguments)]
    fn backtrace_and_fit(
        &self,
        xcorr_hist: &Tensor<B, 3>,
        prev_hist: &Tensor<B, 3, Int>,
        weights: &Tensor<B, 3>,
        cursor: Tensor<B, 2, Int>,
        steps: usize,
        batch: usize,
        total_slots: usize,
    ) -> Tensor<B, 2> {
        let slots = self.slots();
        let dif = self.dif_period;
        let device = cursor.device();

        let mut cursor = cursor;
        let mut frame_corr: Tensor<B, 2> = Tensor::zeros([steps, batch], &device);
        let mut sums = [
            Tensor::<B, 2>::zeros([steps, batch], &device), // sw
            Tensor::<B, 2>::zeros([steps, batch], &device), // sx
            Tensor::<B, 2>::zeros([steps, batch], &device), // sxx
            Tensor::<B, 2>::zeros([steps, batch], &device), // sxy
            Tensor::<B, 2>::zeros([steps, batch], &device), // sy
        ];

        // `k` counts back from the newest slot, so `sub = slots - 1 - k`, and
        // hop `t` reads absolute slot `2t + sub`.
        for k in 0..slots {
            let sub = slots - 1 - k;
            let lo = sub as isize;
            let hi = lo + (SUBS_PER_HOP * steps) as isize - 1;
            debug_assert!(hi <= total_slots as isize);

            // [batch, steps, ..] -> [steps, batch, ..]
            let xc = xcorr_hist
                .clone()
                .slice(s![.., lo..hi;SUBS_PER_HOP as isize, 0..dif as isize])
                .swap_dims(0, 1);
            let pp = prev_hist
                .clone()
                .slice(s![.., lo..hi;SUBS_PER_HOP as isize, ..])
                .swap_dims(0, 1);
            let w = weights
                .clone()
                .slice(s![.., .., lo..lo + 1])
                .squeeze_dim::<2>(2)
                .swap_dims(0, 1);

            let index = cursor.clone().unsqueeze_dim::<3>(2);
            frame_corr = frame_corr + w.clone() * xc.gather(2, index.clone()).squeeze_dim::<2>(2);

            // period = max_period - cursor, as a float.
            let period = cursor
                .clone()
                .float()
                .neg()
                .add_scalar(self.max_period as f32);
            let x = sub as f32;

            sums[0] = sums[0].clone() + w.clone();
            sums[1] = sums[1].clone() + w.clone().mul_scalar(x);
            sums[2] = sums[2].clone() + w.clone().mul_scalar(x).mul_scalar(x);
            sums[3] = sums[3].clone() + w.clone().mul_scalar(x) * period.clone();
            sums[4] = sums[4].clone() + w * period;

            cursor = pp.gather(2, index).squeeze_dim::<2>(2);
        }

        let [sw, sx, sxx, sxy, sy] = sums;

        let frame_corr = frame_corr.div_scalar(slots as f32).clamp_min(0.0f32);
        let voiced = frame_corr.greater_equal_elem(VOICED_THRESHOLD);

        let numerator = sw.clone() * sxy - sx.clone() * sy.clone();
        let denominator = sw.clone() * sxx - sx.clone() * sx.clone();
        let degenerate = denominator.clone().equal_elem(0.0f32);
        let slope = numerator / denominator.mask_fill(degenerate, 1e-15f32);

        // Cap the contour slope so one bad slot cannot swing the estimate.
        let limit =
            (sy.clone() / sw.clone()).div_scalar(4.0 * SUBS_PER_HOP as f32 * self.n_feat as f32);
        let clamped = slope.max_pair(limit.clone().neg()).min_pair(limit);
        let slope = clamped.mask_fill(voiced.clone().bool_not(), 0.0f32);

        let intercept = (sy - slope.clone() * sx) / sw;
        let period = intercept + slope.mul_scalar(5.5f32);

        let hz = period.clamp_min(1.0f32).recip().mul_scalar(PROC_FS as f32);
        hz.mask_fill(voiced.bool_not(), 0.0f32)
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

    /// Drives the host and captures, per hop, the two correlation slots it
    /// produced, their energies, and the pitch it reported.
    fn host_reference(steps: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut est = TenVadPitchEstimator::new();
        let slots = est.slots();
        let (mut xc, mut fw, mut hz) = (Vec::new(), Vec::new(), Vec::new());

        for step in 0..steps {
            let pitch = est.frame_pitch(&pulse_hop(150.0, step * HOP), &spectrum(step));
            hz.push(pitch);
            for sub in 0..SUBS_PER_HOP {
                xc.extend_from_slice(est.xcorr_slot(slots - SUBS_PER_HOP + sub));
                fw.push(est.frm_weight()[slots - SUBS_PER_HOP + sub]);
            }
        }
        (xc, fw, hz)
    }

    fn config() -> PitchTrackConfig {
        PitchTrackConfig::new()
    }

    #[test]
    fn test_gather_needs_a_contiguous_index() {
        // `gather` ignores strides on a non-contiguous index tensor: given a
        // transposed view it reads element 0 of each row rather than the
        // indexed element. The backtrace's cursor is exactly such an index, so
        // it is built with `stack` rather than `cat` + `swap_dims`.
        //
        // If this starts failing, burn has fixed the stride handling and the
        // comment at the cursor construction is stale -- the `stack` is then
        // merely tidy rather than load-bearing.
        let device = Default::default();
        let (steps, wide) = (3usize, 56usize);

        let mut v = vec![0i32; steps * wide];
        for t in 0..steps {
            for j in 0..wide {
                v[t * wide + j] = (t as i32) * 1000 + j as i32;
            }
        }
        let data = Tensor::<B, 3, Int>::from_data(TensorData::new(v, [steps, 1, wide]), &device);

        let contiguous = Tensor::<B, 3, Int>::full([steps, 1, 1], 37, &device);
        let rows: Vec<Tensor<B, 2, Int>> = (0..steps)
            .map(|_| Tensor::full([1, 1], 37, &device))
            .collect();
        let transposed = Tensor::cat(rows, 1).swap_dims(0, 1).unsqueeze_dim::<3>(2);

        let good: Vec<i32> = data
            .clone()
            .gather(2, contiguous)
            .to_data_as::<i32>()
            .to_vec_as::<i32>()
            .unwrap();
        let bad: Vec<i32> = data
            .gather(2, transposed)
            .to_data_as::<i32>()
            .to_vec_as::<i32>()
            .unwrap();

        assert_eq!(
            good,
            vec![37, 1037, 2037],
            "contiguous index should be exact"
        );
        assert_ne!(
            good, bad,
            "gather now honours strides on the index; the `stack` in \
             `backtrace_and_fit` is no longer load-bearing",
        );
    }

    #[test]
    fn test_config_meta() {
        let cfg = config();
        assert_eq!(cfg.max_period(), 64);
        assert_eq!(cfg.dif_period(), 56);
        assert_eq!(cfg.n_feat(), 3);
        assert_eq!(cfg.slots(), 6);
        assert_eq!(cfg.carry_slots(), 4);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        // A very long hop leaves fewer than two history frames.
        assert!(config().with_hop_size(1 << 16).validate().is_err());
    }

    #[test]
    fn test_penalty_matrix_reproduces_the_asymmetric_window() {
        let cfg = config();
        let dif = cfg.dif_period();
        let pen = cfg.to_vec_penalty();

        for idx in 0..dif {
            let lo = idx.min(4);
            let hi = (idx + 4).min(dif - 1);
            let mut widest = 0usize;
            for cand in 0..dif {
                let valid = cand >= lo && cand <= hi;
                let entry = pen[idx * dif + cand];
                if valid {
                    widest += 1;
                    let j = (cand as i32 - idx as i32).abs() as f32;
                    assert_eq!(entry, PITCH_MAX_PATH_W * j * j, "idx {idx} cand {cand}");
                } else {
                    assert_eq!(entry, INVALID_PENALTY, "idx {idx} cand {cand}");
                }
            }
            assert_eq!(widest, hi - lo + 1);
        }

        // The window really is 5 wide at one end and 52 at the other.
        let width = |idx: usize| (idx + 4).min(dif - 1) - idx.min(4) + 1;
        assert_eq!(width(0), 5);
        assert_eq!(width(dif - 1), 52);
    }

    #[test]
    fn test_forward_matches_host_stage() {
        let device = Default::default();
        let steps = 12;
        let (xc, fw, want) = host_reference(steps);
        let cfg = config();
        let track: PitchTrack<B> = cfg.init(&device);

        let xc_t = Tensor::<B, 1>::from_floats(xc.as_slice(), &device).reshape([
            steps,
            1,
            SUBS_PER_HOP,
            cfg.max_period(),
        ]);
        let fw_t =
            Tensor::<B, 1>::from_floats(fw.as_slice(), &device).reshape([steps, 1, SUBS_PER_HOP]);

        let (got, _) = track.forward(xc_t, fw_t, track.init_state(1, &device));
        let got: Vec<f32> = got.to_data_as::<f32>().to_vec_as::<f32>().unwrap();

        for (t, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(*g > 0.0, *w > 0.0, "hop {t}: voicing disagrees, {g} vs {w}");
            if *w > 0.0 {
                let rel = (g - w).abs() / w;
                assert!(rel < 1e-3, "hop {t}: {g} Hz vs {w} Hz (rel {rel})");
            }
        }
        assert!(
            want.iter().filter(|v| **v > 0.0).count() * 2 > steps,
            "fixture should be mostly voiced",
        );
    }

    #[test]
    fn test_forward_sequence_matches_stepwise() {
        let device = Default::default();
        let steps = 8;
        let (xc, fw, _) = host_reference(steps);
        let cfg = config();
        let track: PitchTrack<B> = cfg.init(&device);

        let xc_t = Tensor::<B, 1>::from_floats(xc.as_slice(), &device).reshape([
            steps,
            1,
            SUBS_PER_HOP,
            cfg.max_period(),
        ]);
        let fw_t =
            Tensor::<B, 1>::from_floats(fw.as_slice(), &device).reshape([steps, 1, SUBS_PER_HOP]);

        let (whole, _) = track.forward(xc_t.clone(), fw_t.clone(), track.init_state(1, &device));

        let mut state = track.init_state(1, &device);
        let mut stepwise = Vec::new();
        for step in 0..steps {
            let x = xc_t
                .clone()
                .slice_dim(0, step as isize..(step + 1) as isize);
            let f = fw_t
                .clone()
                .slice_dim(0, step as isize..(step + 1) as isize);
            let (out, next) = track.forward(x, f, state);
            state = next;
            stepwise.extend(out.to_data_as::<f32>().to_vec_as::<f32>().unwrap());
        }

        whole.to_data().assert_approx_eq::<f32>(
            &TensorData::new(stepwise, [steps, 1, 1]),
            Tolerance::relative(1e-4),
        );
    }

    #[test]
    fn test_silence_is_unvoiced() {
        let device = Default::default();
        let cfg = config();
        let track: PitchTrack<B> = cfg.init(&device);
        let steps = 4;

        let (out, _) = track.forward(
            Tensor::zeros([steps, 1, SUBS_PER_HOP, cfg.max_period()], &device),
            Tensor::zeros([steps, 1, SUBS_PER_HOP], &device),
            track.init_state(1, &device),
        );

        let got: Vec<f32> = out.to_data_as::<f32>().to_vec_as::<f32>().unwrap();
        for (t, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "hop {t} went non-finite on silence");
            assert_eq!(*v, 0.0, "hop {t} reported {v} Hz on silence");
        }
    }

    #[test]
    fn test_batch_rows_are_independent() {
        let device = Default::default();
        let steps = 6;
        let (xc, fw, _) = host_reference(steps);
        let cfg = config();
        let track: PitchTrack<B> = cfg.init(&device);
        let lags = cfg.max_period();

        // Row 0 is the real fixture; row 1 is silence, which must stay
        // unvoiced regardless of what row 0 is doing.
        let mut xc_pair = Vec::new();
        let mut fw_pair = Vec::new();
        for step in 0..steps {
            let base = step * SUBS_PER_HOP * lags;
            xc_pair.extend_from_slice(&xc[base..base + SUBS_PER_HOP * lags]);
            xc_pair.extend(std::iter::repeat_n(0.0f32, SUBS_PER_HOP * lags));
            fw_pair.extend_from_slice(&fw[step * SUBS_PER_HOP..(step + 1) * SUBS_PER_HOP]);
            fw_pair.extend(std::iter::repeat_n(0.0f32, SUBS_PER_HOP));
        }

        let (out, _) = track.forward(
            Tensor::<B, 1>::from_floats(xc_pair.as_slice(), &device).reshape([
                steps,
                2,
                SUBS_PER_HOP,
                lags,
            ]),
            Tensor::<B, 1>::from_floats(fw_pair.as_slice(), &device).reshape([
                steps,
                2,
                SUBS_PER_HOP,
            ]),
            track.init_state(2, &device),
        );
        let got: Vec<f32> = out.to_data_as::<f32>().to_vec_as::<f32>().unwrap();

        let xc_t = Tensor::<B, 1>::from_floats(xc.as_slice(), &device).reshape([
            steps,
            1,
            SUBS_PER_HOP,
            lags,
        ]);
        let fw_t =
            Tensor::<B, 1>::from_floats(fw.as_slice(), &device).reshape([steps, 1, SUBS_PER_HOP]);
        let (solo, _) = track.forward(xc_t, fw_t, track.init_state(1, &device));
        let solo: Vec<f32> = solo.to_data_as::<f32>().to_vec_as::<f32>().unwrap();

        for step in 0..steps {
            assert!(
                (got[step * 2] - solo[step]).abs() < 1e-3,
                "hop {step}: row 0 {} vs solo {}",
                got[step * 2],
                solo[step],
            );
            assert_eq!(got[step * 2 + 1], 0.0, "hop {step}: silent row 1 is voiced");
        }
    }
}
