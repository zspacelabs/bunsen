//! # Normalized cross-correlation over a lag range.
//!
//! Scores how well a reference window matches the signal at each of a range of
//! lags behind it. The basis of time-domain pitch detection, and equally of
//! delay estimation, echo alignment, and onset matching.
//!
//! ```text
//! buf:  [-------- max_lag --------][--- window ---]
//!       ^lag 0                      ^the reference
//!            ^lag 1
//! ```
//!
//! For each lag, the window starting there is scored against the reference:
//!
//! ```text
//! score[lag] = 2 * dot(ref, w[lag]) / (energy(w[lag]) + energy(ref) + floor)
//! ```
//!
//! ## Why this normalization
//!
//! `2·⟨x,y⟩ / (‖x‖² + ‖y‖²)` is **bounded above by 1**, and reaches it exactly
//! when `x == y`. That follows from AM-GM: `2⟨x,y⟩ ≤ 2‖x‖‖y‖ ≤ ‖x‖² + ‖y‖²`.
//! So a score is directly readable as "how close to a repeat is this", with no
//! per-signal calibration, and a threshold means the same thing at any
//! amplitude.
//!
//! It differs from the Pearson-style `⟨x,y⟩ / (‖x‖‖y‖)` in what it does with
//! unequal energies: the geometric mean is indifferent to them, this form
//! penalizes them. For periodicity that is usually what you want — a quiet
//! window that merely correlates in shape is not a repeat of a loud one.
//!
//! ## The energy floor is absolute
//!
//! [`LagSearchConfig::energy_floor`] is added to the denominator to keep
//! silence from scoring as a perfect match against silence. It is an *absolute*
//! quantity, so it is only meaningful relative to the amplitude scale of the
//! input — a floor tuned for int16-scale signals means nothing applied to
//! `[-1, 1]` audio. Set it in the same units as the signal, or leave it zero
//! and threshold on the reference energy instead.
//!
//! ## Lag energies are computed directly, not slid
//!
//! The obvious optimization is a sliding sum: subtract the sample leaving the
//! window, add the one entering. It is `O(max_lag + window)` instead of
//! `O(max_lag · window)`, and it is the wrong choice here.
//!
//! It serializes. Each lag depends on the previous, so the whole range becomes
//! a dependency chain of `max_lag` steps where the direct form is one parallel
//! reduction — and on a device the arithmetic is free while the serialization
//! is not.
//!
//! It is also *less accurate*. A running sum of squares accumulates rounding
//! monotonically across the whole range and never recovers, and in near-silence
//! the accumulated error can drive it negative — which is why implementations
//! that slide invariably end up clamping it at zero. The direct reduction has
//! no chain to accumulate along and needs no clamp.

use burn::{
    config::Config,
    prelude::*,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    WithOkOrPanic,
};

/// Config for [`LagSearch`].
#[derive(Config, Debug, Copy)]
pub struct LagSearchConfig {
    /// The length of the correlation window.
    pub window: usize,

    /// How many lags to score, counting back from the reference.
    pub max_lag: usize,

    /// Added to the denominator; see the module docs on scale.
    #[config(default = "0.0")]
    pub energy_floor: f32,
}

impl LagSearchConfig {
    /// The buffer length [`LagSearch::forward`] expects.
    ///
    /// Enough to hold every lagged window plus the reference behind them.
    pub fn buf_len(&self) -> usize {
        self.max_lag + self.window
    }

    /// Validates the geometry.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the window or lag count is zero, or if the
    /// floor is negative.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.window == 0 {
            return Err(BunsenError::Invalid(
                "LagSearch window must be non-zero".to_string(),
            ));
        }
        if self.max_lag == 0 {
            return Err(BunsenError::Invalid(
                "LagSearch max_lag must be non-zero".to_string(),
            ));
        }
        if self.energy_floor < 0.0 {
            return Err(BunsenError::Invalid(format!(
                "LagSearch energy_floor ({}) must not be negative",
                self.energy_floor,
            )));
        }
        Ok(())
    }

    /// Builds the search.
    ///
    /// # Errors
    /// See [`validate`](Self::validate).
    pub fn try_init(&self) -> BunsenResult<LagSearch> {
        self.validate()?;
        Ok(LagSearch { cfg: *self })
    }

    /// Builds the search, panicking on error.
    pub fn init(&self) -> LagSearch {
        self.try_init().ok_or_panic()
    }
}

/// A normalized lag search.
///
/// Carries no tensors — the geometry is all it needs — so one instance serves
/// any device and any batch. Built by [`LagSearchConfig::try_init`].
#[derive(Debug, Clone, Copy)]
pub struct LagSearch {
    cfg: LagSearchConfig,
}

impl LagSearch {
    /// The geometry this search was built for.
    pub fn config(&self) -> &LagSearchConfig {
        &self.cfg
    }

    /// The buffer length [`forward`](Self::forward) expects.
    pub fn buf_len(&self) -> usize {
        self.cfg.buf_len()
    }

    /// Scores every lag against the reference window.
    ///
    /// # Arguments
    /// * `buf`: `[rows, buf_len]`. The trailing
    ///   [`window`](LagSearchConfig::window) samples are the reference; lag `l`
    ///   scores the window starting at `l`. Lags therefore run *backwards* in
    ///   time — lag `0` is furthest from the reference — which a caller mapping
    ///   lags to periods must account for.
    ///
    /// # Returns
    /// `(`[rows, `max_lag`]` scores, `[rows, 1]` reference energy)`. The energy
    /// is returned because callers almost always need it to decide whether a
    /// score is meaningful at all, and recomputing it would be wasteful.
    ///
    /// # Panics
    /// If `buf`'s trailing axis is not [`buf_len`](Self::buf_len).
    pub fn forward<B: Backend>(
        &self,
        buf: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (window, max_lag) = (self.cfg.window, self.cfg.max_lag);

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["rows", "buf_len"],
            &buf,
            &[("buf_len", self.buf_len())],
        );

        // [rows, window]: the reference sits at the end of the buffer.
        let reference = buf.clone().slice_dim(1, max_lag as isize..);
        let ref_energy = reference.clone().powi_scalar(2).sum_dim(1);

        // The lagged windows cover `max_lag - 1 + window` samples, one short
        // of the buffer. On CubeCL a vectorized `unfold` truncates its outer
        // stride to a multiple of the line width, so that leftover sample
        // would displace every row after the first; trimming to the covered
        // span first avoids it. Pinned by `burner::tensor::burn_behavior`.
        let covered = max_lag - 1 + window;

        // [rows, max_lag, window]
        let windows = buf
            .slice_dim(1, 0..covered as isize)
            .unfold::<3, _>(1, window, 1);

        // Direct reduction rather than a sliding sum; see the module docs.
        // [rows, max_lag]
        let lag_energy = windows.clone().powi_scalar(2).sum_dim(2).squeeze_dim(2);

        // Broadcast-and-reduce rather than a matvec. `matmul` against a
        // `[rows, window, 1]` operand is a batched matvec, and on wgpu every
        // autotune candidate for `n == 1` fails outright -- pinned by
        // `burner::tensor::burn_behavior`. This form is the same arithmetic
        // and reduces over the same view the energy does, so the two fuse.
        // [rows, max_lag]
        let dot = (windows * reference.unsqueeze_dim::<3>(1))
            .sum_dim(2)
            .squeeze_dim(2);

        let denominator = lag_energy + ref_energy.clone() + self.cfg.energy_floor;
        let score = dot.mul_scalar(2.0f32) / denominator;

        (score, ref_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    const WINDOW: usize = 16;
    const MAX_LAG: usize = 24;

    fn cfg() -> LagSearchConfig {
        LagSearchConfig::new(WINDOW, MAX_LAG)
    }

    fn to_vec(t: Tensor<B, 2>) -> Vec<f32> {
        t.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic()
    }

    fn upload(
        rows: &[Vec<f32>],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 2> {
        let len = rows[0].len();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::from_data(TensorData::new(flat, [rows.len(), len]), device)
    }

    #[test]
    fn test_config_meta() {
        let c = cfg();
        assert_eq!(c.buf_len(), MAX_LAG + WINDOW);
        c.validate().unwrap();
        assert_eq!(c.init().buf_len(), MAX_LAG + WINDOW);
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        for bad in [
            LagSearchConfig::new(0, MAX_LAG),
            LagSearchConfig::new(WINDOW, 0),
            cfg().with_energy_floor(-1.0),
        ] {
            assert!(
                matches!(bad.validate(), Err(BunsenError::Invalid(_))),
                "expected Invalid: {bad:?}",
            );
        }
    }

    #[test]
    fn test_an_exact_repeat_scores_exactly_one() {
        // The closed form the normalization is chosen for: when the lagged
        // window equals the reference, `2ab/(a^2+b^2)` is exactly 1.
        let device = Default::default();
        let period = 8usize;
        let search = cfg().init();

        let buf: Vec<f32> = (0..search.buf_len())
            .map(|n| ((n % period) as f32 * 0.9).sin() + 0.3)
            .collect();
        let (score, _) = search.forward(upload(&[buf], &device));
        let got = to_vec(score);

        // The reference starts at `max_lag`, so any lag congruent to it mod
        // the period is an exact repeat.
        for (lag, &v) in got.iter().enumerate() {
            if lag % period == MAX_LAG % period {
                assert!(
                    (v - 1.0).abs() < 1e-4,
                    "lag {lag} is an exact repeat, scored {v}"
                );
            }
        }
    }

    #[test]
    fn test_scores_never_exceed_one() {
        // AM-GM, on data with no structure at all.
        let device = Default::default();
        let search = cfg().init();
        let buf = Tensor::<B, 2>::random(
            [6, search.buf_len()],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (score, _) = search.forward(buf);
        for (i, v) in to_vec(score).iter().enumerate() {
            assert!(*v <= 1.0 + 1e-5, "score {i} is {v}, above the bound");
        }
    }

    #[test]
    fn test_the_peak_lands_on_the_period() {
        let device = Default::default();
        let period = 7usize;
        let search = cfg().init();

        let buf: Vec<f32> = (0..search.buf_len())
            .map(|n| ((n % period) as f32 * 1.3).sin())
            .collect();
        let (score, _) = search.forward(upload(&[buf], &device));
        let got = to_vec(score);

        let peak = got
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(
            peak % period,
            MAX_LAG % period,
            "peak at lag {peak} is not a whole number of periods from the reference",
        );
    }

    #[test]
    fn test_matches_a_direct_host_computation() {
        let device = Default::default();
        let search = cfg().with_energy_floor(0.5).init();

        let buf: Vec<f32> = (0..search.buf_len())
            .map(|n| ((n as f32) * 0.37).sin() + 0.2 * ((n as f32) * 1.9).cos())
            .collect();
        let (score, energy) = search.forward(upload(std::slice::from_ref(&buf), &device));

        let reference = &buf[MAX_LAG..];
        let e_ref: f32 = reference.iter().map(|v| v * v).sum();
        assert!(
            (to_vec(energy)[0] - e_ref).abs() < 1e-4 * e_ref,
            "reference energy mismatch",
        );

        let got = to_vec(score);
        for (lag, &v) in got.iter().enumerate() {
            let w = &buf[lag..lag + WINDOW];
            let dot: f32 = w.iter().zip(reference.iter()).map(|(a, b)| a * b).sum();
            let e_lag: f32 = w.iter().map(|v| v * v).sum();
            let want = 2.0 * dot / (e_lag + e_ref + 0.5);
            assert!((v - want).abs() < 1e-4, "lag {lag}: {v} vs {want}");
        }
    }

    #[test]
    fn test_rows_are_independent() {
        let device = Default::default();
        let search = cfg().init();

        let a: Vec<f32> = (0..search.buf_len())
            .map(|n| ((n % 5) as f32).sin())
            .collect();
        let b: Vec<f32> = (0..search.buf_len())
            .map(|n| ((n % 9) as f32).cos())
            .collect();

        let (joint, _) = search.forward(upload(&[a.clone(), b], &device));
        let (solo, _) = search.forward(upload(std::slice::from_ref(&a), &device));

        let j = to_vec(joint);
        let s = to_vec(solo);
        for (lag, (&a, &b)) in j.iter().zip(s.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "lag {lag}: batched {a} vs alone {b}");
        }
    }

    #[test]
    fn test_the_energy_floor_damps_silence() {
        // Silence correlates perfectly with silence, which is exactly the
        // false positive the floor exists to suppress.
        let device = Default::default();
        let quiet: Vec<f32> = (0..cfg().buf_len())
            .map(|n| 1e-4 * (n as f32).sin())
            .collect();

        let (bare, _) = cfg()
            .init()
            .forward(upload(std::slice::from_ref(&quiet), &device));
        let (floored, _) = cfg()
            .with_energy_floor(1.0)
            .init()
            .forward(upload(&[quiet], &device));

        let b = to_vec(bare).iter().fold(0.0f32, |m, v| m.max(*v));
        let f = to_vec(floored).iter().fold(0.0f32, |m, v| m.max(*v));
        assert!(b > 0.5, "unfloored silence should score high, got {b}");
        assert!(f < 1e-3, "floored silence should score near zero, got {f}");
    }

    #[test]
    fn test_direct_energy_beats_a_sliding_sum() {
        // The justification for not sliding, made concrete. A sliding sum of
        // squares accumulates rounding monotonically across the lag range and
        // never recovers; the direct reduction has no chain to accumulate
        // along. Both are compared against an f64 ground truth.
        let long = LagSearchConfig::new(64, 512);
        let n = long.buf_len();

        // A loud prefix followed by a quiet tail: the regime where a running
        // sum's absolute error swamps the values it is still tracking.
        let buf: Vec<f32> = (0..n)
            .map(|i| {
                let amp = if i < n / 2 { 3.0e3 } else { 1.0e-2 };
                amp * ((i as f32) * 0.7).sin()
            })
            .collect();

        let truth = |lag: usize| -> f64 {
            buf[lag..lag + long.window]
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum()
        };

        // The sliding form, in f32, with the clamp such implementations need.
        let mut running: f32 = buf[..long.window].iter().map(|v| v * v).sum();
        let mut slide_err = 0.0f64;
        let mut direct_err = 0.0f64;
        for lag in 0..long.max_lag {
            if lag > 0 {
                running = (running - buf[lag - 1] * buf[lag - 1]).max(0.0)
                    + buf[lag + long.window - 1] * buf[lag + long.window - 1];
            }
            let direct: f32 = buf[lag..lag + long.window].iter().map(|v| v * v).sum();

            let t = truth(lag);
            let scale = t.max(1e-12);
            slide_err = slide_err.max(((running as f64) - t).abs() / scale);
            direct_err = direct_err.max(((direct as f64) - t).abs() / scale);
        }

        assert!(
            direct_err < slide_err,
            "direct {direct_err:.3e} should beat sliding {slide_err:.3e}",
        );
    }

    #[test]
    #[should_panic(expected = "buf_len")]
    fn test_wrong_buffer_length_is_rejected() {
        let device = Default::default();
        let search = cfg().init();
        search.forward(Tensor::<B, 2>::zeros([1, search.buf_len() + 3], &device));
    }
}
