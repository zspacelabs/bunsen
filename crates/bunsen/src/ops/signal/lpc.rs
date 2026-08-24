//! # Linear prediction: autocorrelation and the Levinson-Durbin recursion.
//!
//! The two host-side pieces of an all-pole spectral fit:
//!
//! * [`Autocorrelator`] recovers autocorrelation lags from a power spectrum,
//!   without a round trip through an inverse FFT.
//! * [`levinson_durbin`] solves the resulting Toeplitz system for the
//!   prediction coefficients.
//!
//! Both are scalar `f32`, and deliberately so: an LPC fit is a short recursion
//! over a handful of lags, so the work is dominated by call overhead rather
//! than arithmetic. What benefits from a device is fitting *many* spectra at
//! once, which is a different shape — run the recursion with a masked freeze
//! instead of a branch, since a batched implementation cannot bail out per row.
//!
//! ## Why not an inverse FFT
//!
//! Autocorrelation is the inverse transform of the power spectrum, so an
//! inverse FFT is the obvious route. Evaluating the cosine sum directly wins
//! here for two reasons. It needs only the first few lags, where an FFT
//! computes all of them; and it can accumulate in `f64` at no meaningful cost,
//! which matters because a flat `f32` sum over several hundred bins has error
//! growing with the term count, while an FFT's grows with its logarithm. The
//! direct sum in `f32` would be materially *worse* than an FFT; in `f64` it is
//! at least as good.
//!
//! ## Fitting many spectra at once
//!
//! [`levinson_durbin_batched`] is the device form, and it is not a
//! transcription of the host one. A batched implementation cannot branch per
//! row, so the early exit becomes a masked freeze; see that function for why
//! the freeze is applied *after* the update and why it needs no sticky
//! bookkeeping.

use burn::prelude::*;

/// Autocorrelation lags from a real, even power spectrum.
///
/// Holds the cosine table the sum is evaluated against, so building one is the
/// expensive part and evaluating it is cheap.
///
/// # Scaling and the Nyquist bin
///
/// For a spectrum `S` over an `N`-point FFT, lag `l` is
///
/// ```text
/// r[l] = 0.5 * S[0] + sum(S[k] * cos(2*pi*k*l/N) for k in 1..N/2)
/// ```
///
/// Two conventions are baked in and worth stating plainly, because neither is
/// the only reasonable choice:
///
/// * **The result is unnormalized**, larger than the true autocorrelation by
///   `N/2`. Nothing downstream in an LPC fit cares — [`levinson_durbin`] is
///   scale-invariant — but a caller using these lags for anything else should
///   divide.
/// * **The Nyquist bin is excluded**, since the sum stops at `N/2 - 1`. For a
///   spectrum that has been lowpassed well below Nyquist this is immaterial;
///   for one with real energy at Nyquist it is not.
#[derive(Debug, Clone, PartialEq)]
pub struct Autocorrelator {
    fft_size: usize,

    /// `cos(2*pi*t/fft_size)` for `t` in `0..fft_size`.
    cos_table: Vec<f64>,
}

impl Autocorrelator {
    /// Builds an autocorrelator for a given FFT size.
    ///
    /// # Panics
    /// If `fft_size` is not even.
    pub fn new(fft_size: usize) -> Self {
        assert_eq!(fft_size % 2, 0, "fft_size must be even");
        let cos_table = (0..fft_size)
            .map(|t| (core::f64::consts::TAU * t as f64 / fft_size as f64).cos())
            .collect();
        Self {
            fft_size,
            cos_table,
        }
    }

    /// The FFT size this autocorrelator was built for.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// The number of spectrum bins [`autocorrelate`](Self::autocorrelate)
    /// expects.
    pub fn n_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// The factor by which results exceed the true autocorrelation.
    ///
    /// Divide by this to get normalized lags; see the type docs.
    pub fn scale(&self) -> f32 {
        (self.fft_size / 2) as f32
    }

    /// Fills `out` with lags `0..out.len()` of `spectrum`.
    ///
    /// Accumulated in `f64`; see the module docs.
    ///
    /// # Arguments
    /// * `spectrum`: [`n_bins`](Self::n_bins) non-negative bin values.
    /// * `out`: the lags to fill, overwritten in full.
    ///
    /// # Panics
    /// If `spectrum` is not [`n_bins`](Self::n_bins) long.
    pub fn autocorrelate(
        &self,
        spectrum: &[f32],
        out: &mut [f32],
    ) {
        assert_eq!(
            spectrum.len(),
            self.n_bins(),
            "autocorrelate expects {} bins for a {}-point FFT",
            self.n_bins(),
            self.fft_size,
        );

        for (lag, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, &s) in spectrum.iter().enumerate().take(self.fft_size / 2).skip(1) {
                acc += s as f64 * self.cos_table[(lag * k) % self.fft_size];
            }
            *slot = (0.5 * spectrum[0] as f64 + acc) as f32;
        }
    }
}

/// Solves for linear-prediction coefficients by Levinson-Durbin recursion.
///
/// Returns the coefficients of the *whitening* filter, in the convention
///
/// ```text
/// residual[n] = x[n] + sum(out[j] * x[n - 1 - j] for j in 0..out.len())
/// ```
///
/// so a process `x[n] = a*x[n-1] + e[n]` yields `out[0] == -a`. Note the sign:
/// these are the negated prediction coefficients.
///
/// Scale-invariant: multiplying `ac` through by a constant leaves the solution
/// unchanged, including where an early bail-out lands, since every comparison
/// is against `ac[0]`. That is exact arithmetic; in `f32` the two runs agree to
/// rounding rather than bit for bit, because scaling the input rounds it and
/// the recursion then rounds differently.
///
/// # The early bail-out
///
/// `bail_ratio` stops the recursion once the residual error falls below
/// `bail_ratio * ac[0]`, leaving the remaining coefficients at **zero** rather
/// than at a partial value: index `k` is only ever written by `out[i] = r` at
/// step `k`, and the symmetric update at step `i` touches only `0..i-1`.
///
/// `None` runs to full order, which is the textbook algorithm and what most
/// callers want. The option exists because some reference implementations bail
/// (CELT's `_celt_lpc` uses `Some(0.001)`, i.e. 30 dB), and reproducing one of
/// those requires reproducing where it stopped.
///
/// The check happens **after** an iteration commits, so iteration `i` always
/// completes and only `i+1..` are skipped. A batched port must freeze after the
/// update for the same reason.
///
/// # Arguments
/// * `ac`: autocorrelation lags `0..=order`; at least 2 long.
/// * `out`: `ac.len() - 1` coefficients, overwritten in full.
/// * `bail_ratio`: early-exit threshold as a fraction of `ac[0]`.
///
/// # Panics
/// If `out.len() + 1 != ac.len()`, or if `ac` is shorter than 2.
pub fn levinson_durbin(
    ac: &[f32],
    out: &mut [f32],
    bail_ratio: Option<f32>,
) {
    assert!(ac.len() >= 2, "levinson_durbin needs at least lags 0 and 1");
    assert_eq!(
        out.len() + 1,
        ac.len(),
        "levinson_durbin writes {} coefficients for {} lags",
        ac.len() - 1,
        ac.len(),
    );

    out.fill(0.0);
    if ac[0] == 0.0 {
        return;
    }

    let order = out.len();
    let mut error = ac[0];

    for i in 0..order {
        // Sum the products first and fold in `ac[i + 1]` last. The order is
        // observable in f32, and this is the order CELT uses.
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += out[j] * ac[i - j];
        }
        rr += ac[i + 1];

        let r = -rr / error;
        out[i] = r;

        // The symmetric update, walking inward from both ends. For odd `i` the
        // final pair is the middle element against itself, written once as
        // `l[m] + r * l[m]`.
        for j in 0..((i + 1) >> 1) {
            let head = out[j];
            let tail = out[i - 1 - j];
            out[j] = head + r * tail;
            out[i - 1 - j] = tail + r * head;
        }

        error -= r * r * error;
        if bail_ratio.is_some_and(|ratio| error < ratio * ac[0]) {
            break;
        }
    }
}

/// [`levinson_durbin`] over many autocorrelation sequences at once.
///
/// Same recursion, same convention, same `bail_ratio` semantics -- but a
/// batched implementation cannot take a different number of iterations per
/// row, so it always runs the full order and *freezes* rows that would have
/// stopped.
///
/// # How the freeze works, and why it is exact
///
/// Three details carry the equivalence, and all three are easy to get wrong:
///
/// * **Commit, then freeze.** The scalar form checks its condition *after* an
///   iteration completes, so iteration `i` always commits and only `i+1..` are
///   skipped. The mask is therefore applied after the update, and recomputed
///   after that.
/// * **No sticky bit is needed.** Once a row's error is frozen below the
///   threshold it stays below it, so re-deriving the mask from the error each
///   step is already monotone.
/// * **The frozen tail is the zero initializer.** Coefficient `k` is only ever
///   written at step `k`, so a row that stopped early leaves exact zeros behind
///   rather than partial values -- which is what makes masking sufficient.
///
/// Frozen rows divide by `1` rather than by a stale error, so no row ever
/// produces `inf` or `NaN` to be masked away afterwards. Rows whose `ac[0]` is
/// not positive are frozen from the start and come back as zeros, matching the
/// scalar form's guard.
///
/// The inner product is accumulated one term at a time in increasing `j`
/// rather than by a tree reduce. That is deliberate: summation order is
/// observable in `f32`, and this order is the scalar function's. The cost is
/// `order` products and `order * (order - 1) / 2` adds on `[rows, 1]` tensors,
/// which is constant in the row count.
///
/// # Arguments
/// * `ac`: `[rows, order + 1]` autocorrelation lags.
/// * `bail_ratio`: as [`levinson_durbin`]; `None` runs to full order.
///
/// # Returns
/// `[rows, order]` coefficients.
///
/// # Panics
/// If `ac` has fewer than two lags.
pub fn levinson_durbin_batched<B: Backend>(
    ac: Tensor<B, 2>,
    bail_ratio: Option<f32>,
) -> Tensor<B, 2> {
    let [rows, lags] = ac.dims();
    assert!(
        lags >= 2,
        "levinson_durbin_batched needs at least lags 0 and 1",
    );
    let order = lags - 1;
    let device = ac.device();

    let ac0 = ac.clone().slice_dim(1, 0..1);

    // Rows that cannot be solved at all: the scalar form returns zeros for
    // these, and freezing them from the start does the same.
    let dead = ac0.clone().lower_equal_elem(0.0f32);

    let threshold = bail_ratio.map(|ratio| ac0.clone().mul_scalar(ratio));

    let mut error = ac0;
    let mut lpc: Tensor<B, 2> = Tensor::zeros([rows, order], &device);
    let mut done = dead.clone();

    for i in 0..order {
        // rr = sum(lpc[j] * ac[i - j] for j < i), then += ac[i + 1].
        let mut rr: Tensor<B, 2> = Tensor::zeros([rows, 1], &device);
        if i > 0 {
            // `ac[i], ac[i-1], ..., ac[1]`, so element `j` pairs with `lpc[j]`.
            let reversed = ac.clone().slice_dim(1, 1..(i + 1) as isize).flip([1]);
            let products = lpc.clone().slice_dim(1, 0..i as isize) * reversed;
            for j in 0..i {
                rr = rr + products.clone().slice_dim(1, j as isize..(j + 1) as isize);
            }
        }
        rr = rr + ac.clone().slice_dim(1, (i + 1) as isize..(i + 2) as isize);

        let safe = error.clone().mask_fill(done.clone(), 1.0f32);
        let r = rr.neg() / safe.clone();

        // The symmetric update `lpc[j] += r * lpc[i-1-j]` over the first half
        // is exactly `head + flip(head) * r` over the whole prefix, including
        // the middle element when `i` is odd -- which the scalar loop writes
        // twice with identical operands.
        let mut parts = Vec::with_capacity(3);
        if i > 0 {
            let head = lpc.clone().slice_dim(1, 0..i as isize);
            parts.push(head.clone() + head.flip([1]) * r.clone());
        }
        parts.push(r.clone());
        if i + 1 < order {
            parts.push(Tensor::zeros([rows, order - i - 1], &device));
        }
        let next_lpc = Tensor::cat(parts, 1);

        let next_error = safe.clone() - (r.clone() * r) * safe;

        // Commit, then freeze.
        let wide = done.clone().expand([rows, order]);
        lpc = next_lpc.mask_where(wide, lpc);
        error = next_error.mask_where(done, error);

        done = match &threshold {
            Some(t) => dead.clone().bool_or(error.clone().lower(t.clone())),
            None => dead.clone(),
        };
    }

    lpc
}

/// Applies the LPC analysis filter, with a different tap set per row.
///
/// The filter [`levinson_durbin`] solves for, in the same convention:
///
/// ```text
/// y[n] = x[m] + sum(taps[j] * x[m - 1 - j] for j in 0..order),  m = order + n
/// ```
///
/// so passing coefficients straight from [`levinson_durbin_batched`] whitens
/// the signal they were fitted to. Zero taps leave the signal unchanged.
///
/// Rows are independent and carry **their own taps**, which is the point:
/// speech coefficients are refitted every frame, so a run of frames is a run of
/// different filters. That is what rules out `conv1d`, whose kernel is shared
/// across the batch.
///
/// # Why shift-and-accumulate
///
/// Written as `[out_len, order + 1]` windows contracted against a per-row tap
/// vector, this would materialize a `rows * out_len * order` intermediate. As
/// `order` broadcast multiply-adds over `[rows, out_len]` slices it is far
/// smaller, and it accumulates in a fixed, documented order: the base sample
/// first, then lag `0` upward. Summation order is observable in `f32`, and
/// pinning it is what lets a host implementation and this one agree exactly
/// rather than approximately.
///
/// # Analysis only
///
/// This is the feed-forward direction. Synthesis --
/// `x[n] = y[n] - sum(taps[j] * x[n - 1 - j])` -- is a recurrence in its own
/// output and cannot be written this way; it needs a sequential scan.
///
/// # Arguments
/// * `windowed`: `[rows, order + out_len]`. The leading `order` samples are the
///   history the filter reaches back into, and produce no output of their own.
/// * `taps`: `[rows, order]` coefficients, one set per row.
///
/// # Returns
/// `[rows, out_len]` residual.
///
/// # Panics
/// If the row counts disagree, or if `windowed` is not longer than `order`.
pub fn lpc_residual_batched<B: Backend>(
    windowed: Tensor<B, 2>,
    taps: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [rows, span] = windowed.dims();
    let [tap_rows, order] = taps.dims();

    assert_eq!(
        rows, tap_rows,
        "lpc_residual_batched: {rows} signal rows against {tap_rows} tap rows",
    );
    assert!(
        span > order,
        "lpc_residual_batched needs more than {order} samples of window, got {span}",
    );

    let out_len = span - order;

    // The base sample, then each lag in increasing `j`.
    let mut acc = windowed
        .clone()
        .slice_dim(1, order as isize..(order + out_len) as isize);

    for j in 0..order {
        let tap = taps.clone().slice_dim(1, j as isize..(j + 1) as isize);
        let lo = (order - 1 - j) as isize;
        let lagged = windowed.clone().slice_dim(1, lo..lo + out_len as isize);
        acc = acc + lagged * tap;
    }

    acc
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::{
        errors::WithOkOrPanic,
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type Dev = PerformanceBackend;

    const FFT: usize = 64;

    /// A spectrum that is zero everywhere but bin `k`.
    fn spectral_delta(
        k: usize,
        amplitude: f32,
    ) -> Vec<f32> {
        let mut s = vec![0.0f32; FFT / 2 + 1];
        s[k] = amplitude;
        s
    }

    #[test]
    fn test_meta() {
        let ac = Autocorrelator::new(FFT);
        assert_eq!(ac.fft_size(), FFT);
        assert_eq!(ac.n_bins(), FFT / 2 + 1);
        assert_eq!(ac.scale(), (FFT / 2) as f32);
    }

    #[test]
    #[should_panic(expected = "fft_size must be even")]
    fn test_odd_fft_size_is_rejected() {
        let _ = Autocorrelator::new(63);
    }

    #[test]
    fn test_spectral_delta_autocorrelates_to_a_cosine() {
        // The closed form: a single bin `k` carries a single sinusoid, whose
        // autocorrelation is a cosine at that same frequency. This checks the
        // transform itself rather than agreement with another implementation.
        let auto = Autocorrelator::new(FFT);

        for k in [1usize, 3, 7, 17, FFT / 2 - 1] {
            let spectrum = spectral_delta(k, 2.0);
            let mut lags = vec![0.0f32; 24];
            auto.autocorrelate(&spectrum, &mut lags);

            for (l, &got) in lags.iter().enumerate() {
                let want =
                    2.0 * (core::f64::consts::TAU * (k * l) as f64 / FFT as f64).cos() as f32;
                assert!(
                    (got - want).abs() < 1e-5,
                    "bin {k}, lag {l}: got {got}, want {want}",
                );
            }
        }
    }

    #[test]
    fn test_dc_bin_autocorrelates_to_a_constant() {
        // Bin 0 is a constant offset, so every lag sees the same value -- and
        // it carries the documented `0.5` weight.
        let auto = Autocorrelator::new(FFT);
        let mut lags = vec![0.0f32; 8];
        auto.autocorrelate(&spectral_delta(0, 4.0), &mut lags);

        for (l, &got) in lags.iter().enumerate() {
            assert!((got - 2.0).abs() < 1e-6, "lag {l}: got {got}, want 2.0");
        }
    }

    #[test]
    fn test_nyquist_bin_is_excluded() {
        // Documented behavior, easy to regress: the sum stops below Nyquist.
        let auto = Autocorrelator::new(FFT);
        let mut lags = vec![1.0f32; 8];
        auto.autocorrelate(&spectral_delta(FFT / 2, 9.0), &mut lags);

        assert!(
            lags.iter().all(|v| *v == 0.0),
            "Nyquist energy leaked into the lags: {lags:?}",
        );
    }

    #[test]
    fn test_autocorrelation_is_linear_in_the_spectrum() {
        let auto = Autocorrelator::new(FFT);

        let a = spectral_delta(3, 1.0);
        let b = spectral_delta(11, 1.0);
        let mixed: Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| 2.0 * x + 5.0 * y)
            .collect();

        let mut la = vec![0.0f32; 12];
        let mut lb = vec![0.0f32; 12];
        let mut lm = vec![0.0f32; 12];
        auto.autocorrelate(&a, &mut la);
        auto.autocorrelate(&b, &mut lb);
        auto.autocorrelate(&mixed, &mut lm);

        for l in 0..12 {
            let want = 2.0 * la[l] + 5.0 * lb[l];
            assert!((lm[l] - want).abs() < 1e-5, "lag {l}: {} vs {want}", lm[l]);
        }
    }

    #[test]
    #[should_panic(expected = "expects 33 bins")]
    fn test_wrong_bin_count_is_rejected() {
        let auto = Autocorrelator::new(FFT);
        let mut lags = [0.0f32; 4];
        auto.autocorrelate(&[0.0; 8], &mut lags);
    }

    /// Autocorrelation of an AR(1) process, normalized to `r[0] == 1`.
    fn ar1_lags(
        a: f32,
        order: usize,
    ) -> Vec<f32> {
        (0..=order).map(|k| a.powi(k as i32)).collect()
    }

    #[test]
    fn test_recovers_an_ar1_process() {
        // `x[n] = a*x[n-1] + e[n]` has `r[k] = a^k`, and the whitening filter
        // is `x[n] - a*x[n-1]`, so the coefficients are `[-a, 0, 0, ...]`.
        for a in [0.3f32, -0.5, 0.9] {
            let ac = ar1_lags(a, 8);
            let mut lpc = [0.0f32; 8];
            levinson_durbin(&ac, &mut lpc, None);

            assert!(
                (lpc[0] + a).abs() < 1e-5,
                "a = {a}: expected lpc[0] = {}, got {}",
                -a,
                lpc[0],
            );
            for (j, &c) in lpc.iter().enumerate().skip(1) {
                assert!(c.abs() < 1e-5, "a = {a}: lpc[{j}] should vanish, got {c}");
            }
        }
    }

    #[test]
    fn test_recovers_an_ar2_process() {
        // Yule-Walker in reverse: build the lags an AR(2) process would have,
        // then check the recursion inverts back to its coefficients.
        let (a1, a2) = (0.6f64, -0.25f64);
        let order = 6;

        let mut r = vec![0.0f64; order + 1];
        r[0] = 1.0;
        r[1] = a1 / (1.0 - a2);
        for k in 2..=order {
            r[k] = a1 * r[k - 1] + a2 * r[k - 2];
        }
        let ac: Vec<f32> = r.iter().map(|v| *v as f32).collect();

        let mut lpc = [0.0f32; 6];
        levinson_durbin(&ac, &mut lpc, None);

        assert!((lpc[0] + a1 as f32).abs() < 1e-4, "lpc[0] = {}", lpc[0]);
        assert!((lpc[1] + a2 as f32).abs() < 1e-4, "lpc[1] = {}", lpc[1]);
        for (j, &c) in lpc.iter().enumerate().skip(2) {
            assert!(c.abs() < 1e-4, "lpc[{j}] should vanish, got {c}");
        }
    }

    #[test]
    fn test_is_scale_invariant() {
        // Mathematically exact, but not bit-exact in f32: scaling the input
        // rounds it, and the recursion then rounds differently. The tolerance
        // is against the solution's peak, since the trailing coefficients are
        // legitimately near zero and have no meaningful relative error.
        let ac = ar1_lags(0.7, 8);
        let mut base = [0.0f32; 8];
        levinson_durbin(&ac, &mut base, Some(0.001));

        let peak = base.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for scale in [1e-4f32, 3.0, 5e5] {
            let scaled: Vec<f32> = ac.iter().map(|v| v * scale).collect();
            let mut got = [0.0f32; 8];
            levinson_durbin(&scaled, &mut got, Some(0.001));

            for (j, (&b, &g)) in base.iter().zip(got.iter()).enumerate() {
                assert!(
                    (b - g).abs() < 1e-5 * peak,
                    "scale {scale}, lpc[{j}]: {b} vs {g}",
                );
            }
        }
    }

    #[test]
    fn test_zero_lag_zero_yields_zero_coefficients() {
        let mut lpc = [1.0f32; 4];
        levinson_durbin(&[0.0, 0.0, 0.0, 0.0, 0.0], &mut lpc, None);
        assert_eq!(lpc, [0.0; 4]);
    }

    #[test]
    fn test_bail_out_leaves_zeros_not_partial_values() {
        // The property a batched port depends on: a frozen tail is the zero
        // initializer, so masking is enough and no separate fill is needed.
        //
        // An AR(1) at a = 0.9 whitens completely at step 1, so a generous
        // threshold trips immediately.
        let ac = ar1_lags(0.9, 8);
        let mut lpc = [0.0f32; 8];
        levinson_durbin(&ac, &mut lpc, Some(0.5));

        assert!((lpc[0] + 0.9).abs() < 1e-5, "step 0 must still commit");
        for (j, &c) in lpc.iter().enumerate().skip(1) {
            assert_eq!(c, 0.0, "lpc[{j}] should be exactly zero after the bail");
        }
    }

    #[test]
    fn test_bail_out_only_truncates() {
        // Whatever the threshold, the coefficients that *were* computed match
        // the un-bailed run. The bail changes where it stops, nothing else.
        let ac = ar1_lags(0.45, 10);

        let mut full = [0.0f32; 10];
        levinson_durbin(&ac, &mut full, None);

        let mut bailed = [0.0f32; 10];
        levinson_durbin(&ac, &mut bailed, Some(0.001));

        let live = bailed.iter().rposition(|v| *v != 0.0).map_or(0, |i| i + 1);
        assert!(live >= 1, "expected at least one coefficient");
        for j in 0..live {
            assert!(
                (full[j] - bailed[j]).abs() < 1e-5,
                "lpc[{j}]: full {} vs bailed {}",
                full[j],
                bailed[j],
            );
        }
    }

    #[test]
    #[should_panic(expected = "writes 4 coefficients")]
    fn test_mismatched_output_length_is_rejected() {
        let mut lpc = [0.0f32; 3];
        levinson_durbin(&[1.0, 0.5, 0.2, 0.1, 0.05], &mut lpc, None);
    }

    /// A batch of genuinely positive-definite lag sequences.
    ///
    /// Built by autocorrelating random non-negative spectra, which guarantees
    /// the sequences are ones a real fit could produce -- rather than
    /// arbitrary numbers that might exercise paths no caller reaches.
    fn lag_batch(
        rows: usize,
        order: usize,
        device: &<Dev as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<f32> {
        let auto = Autocorrelator::new(FFT);
        let spectra: Vec<f32> = Tensor::<Dev, 2>::random(
            [rows, auto.n_bins()],
            Distribution::Uniform(0.0, 4.0),
            device,
        )
        .to_data_as::<f32>()
        .to_vec_as::<f32>()
        .ok_or_panic();

        let mut out = Vec::with_capacity(rows * (order + 1));
        for r in 0..rows {
            let mut lags = vec![0.0f32; order + 1];
            auto.autocorrelate(
                &spectra[r * auto.n_bins()..(r + 1) * auto.n_bins()],
                &mut lags,
            );
            out.extend_from_slice(&lags);
        }
        out
    }

    fn host_rows(
        flat: &[f32],
        rows: usize,
        order: usize,
        bail: Option<f32>,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * order];
        for r in 0..rows {
            levinson_durbin(
                &flat[r * (order + 1)..(r + 1) * (order + 1)],
                &mut out[r * order..(r + 1) * order],
                bail,
            );
        }
        out
    }

    #[test]
    fn test_batched_matches_the_scalar_solver() {
        // The differential anchor. The scalar side is itself pinned by the
        // closed-form AR tests above, so agreement here reaches all the way
        // back to the recursion rather than to another implementation.
        let device = Default::default();
        let (rows, order) = (12usize, 16usize);

        for bail in [None, Some(0.001f32), Some(0.2f32)] {
            let flat = lag_batch(rows, order, &device);
            let want = host_rows(&flat, rows, order, bail);

            let ac = Tensor::<Dev, 2>::from_data(TensorData::new(flat, [rows, order + 1]), &device);
            let got: Vec<f32> = levinson_durbin_batched(ac, bail)
                .to_data_as::<f32>()
                .to_vec_as::<f32>()
                .ok_or_panic();

            let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < 1e-4 * peak.max(1e-3),
                    "bail {bail:?}, row {}, coeff {}: {g} vs {w}",
                    i / order,
                    i % order,
                );
            }
        }
    }

    #[test]
    fn test_batched_recovers_an_ar1_process() {
        // Closed form, on the device side too: every row is a different pole.
        let device = Default::default();
        let order = 8;
        let poles = [0.2f32, -0.55, 0.8];

        let mut flat = Vec::new();
        for a in poles {
            flat.extend(ar1_lags(a, order));
        }
        let ac =
            Tensor::<Dev, 2>::from_data(TensorData::new(flat, [poles.len(), order + 1]), &device);

        let got: Vec<f32> = levinson_durbin_batched(ac, None)
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();

        for (r, a) in poles.iter().enumerate() {
            assert!(
                (got[r * order] + a).abs() < 1e-4,
                "row {r}: expected {}, got {}",
                -a,
                got[r * order],
            );
            for j in 1..order {
                assert!(
                    got[r * order + j].abs() < 1e-4,
                    "row {r}, coeff {j} should vanish, got {}",
                    got[r * order + j],
                );
            }
        }
    }

    #[test]
    fn test_batched_freezes_rows_independently() {
        // The whole point of the mask: rows that bail at different steps must
        // not disturb each other. A near-white row runs to full order; a
        // strongly correlated one stops almost immediately.
        let device = Default::default();
        let order = 10;

        let mut flat = ar1_lags(0.95, order);
        flat.extend(ar1_lags(0.05, order));
        let want = host_rows(&flat, 2, order, Some(0.001));

        let ac = Tensor::<Dev, 2>::from_data(TensorData::new(flat, [2, order + 1]), &device);
        let got: Vec<f32> = levinson_durbin_batched(ac, Some(0.001))
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();

        // They really do stop at different points, or this proves nothing.
        let live = |row: usize| {
            want[row * order..(row + 1) * order]
                .iter()
                .rposition(|v| *v != 0.0)
                .map_or(0, |i| i + 1)
        };
        assert_ne!(live(0), live(1), "rows should bail at different steps");

        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "row {}, coeff {}: {g} vs {w}",
                i / order,
                i % order,
            );
        }
    }

    #[test]
    fn test_batched_zero_lag_zero_row_is_zero() {
        // A degenerate row must come back zeroed rather than as NaN, and must
        // not poison its neighbours.
        let device = Default::default();
        let order = 6;

        let mut flat = vec![0.0f32; order + 1];
        flat.extend(ar1_lags(0.6, order));
        let ac = Tensor::<Dev, 2>::from_data(TensorData::new(flat, [2, order + 1]), &device);

        let got: Vec<f32> = levinson_durbin_batched(ac, Some(0.001))
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();

        for (j, &c) in got.iter().take(order).enumerate() {
            assert_eq!(c, 0.0, "dead row coeff {j} should be exactly zero");
        }
        assert!(
            (got[order] + 0.6).abs() < 1e-4,
            "the live row should be unaffected, got {}",
            got[order],
        );
    }

    #[test]
    fn test_residual_of_zero_taps_is_the_signal() {
        let device = Default::default();
        let (rows, order, out) = (2usize, 4usize, 6usize);

        let flat: Vec<f32> = (0..rows * (order + out)).map(|i| i as f32 * 0.5).collect();
        let windowed = Tensor::<Dev, 2>::from_data(
            TensorData::new(flat.clone(), [rows, order + out]),
            &device,
        );
        let taps = Tensor::<Dev, 2>::zeros([rows, order], &device);

        let got: Vec<f32> = lpc_residual_batched(windowed, taps)
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();

        for r in 0..rows {
            for n in 0..out {
                let want = flat[r * (order + out) + order + n];
                assert!(
                    (got[r * out + n] - want).abs() < 1e-6,
                    "row {r}, sample {n}: {} vs {want}",
                    got[r * out + n],
                );
            }
        }
    }

    #[test]
    fn test_residual_recovers_the_excitation_of_an_ar_process() {
        // Closed form, and the whole reason the filter exists: drive an AR(1)
        // process with a known excitation, whiten it with the matching tap,
        // and the excitation comes back out.
        let device = Default::default();
        let (a, order, out) = (0.8f32, 1usize, 12usize);

        let excitation: Vec<f32> = (0..out + order).map(|n| ((n as f32) * 1.7).sin()).collect();
        let mut x = vec![0.0f32; out + order];
        for n in 0..x.len() {
            let prev = if n == 0 { 0.0 } else { x[n - 1] };
            x[n] = a * prev + excitation[n];
        }

        let windowed = Tensor::<Dev, 2>::from_data(TensorData::new(x, [1, out + order]), &device);
        // `levinson_durbin`'s convention: the AR(1) whitener is `-a`.
        let taps = Tensor::<Dev, 2>::from_data(TensorData::new(vec![-a], [1, 1]), &device);

        let got: Vec<f32> = lpc_residual_batched(windowed, taps)
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();

        for n in 0..out {
            let want = excitation[order + n];
            assert!(
                (got[n] - want).abs() < 1e-5,
                "sample {n}: {} vs {want}",
                got[n],
            );
        }
    }

    #[test]
    fn test_residual_matches_a_scalar_filter() {
        // Against the difference equation written out, with a different tap
        // set per row -- which is the case `conv1d` cannot express at all.
        let device = Default::default();
        let (rows, order, out) = (3usize, 5usize, 9usize);

        let sig: Vec<f32> = (0..rows * (order + out))
            .map(|i| ((i as f32) * 0.41).sin() + 0.3 * ((i as f32) * 1.13).cos())
            .collect();
        let tap: Vec<f32> = (0..rows * order)
            .map(|i| 0.4 * ((i as f32) * 0.77).cos())
            .collect();

        let got: Vec<f32> = lpc_residual_batched(
            Tensor::<Dev, 2>::from_data(TensorData::new(sig.clone(), [rows, order + out]), &device),
            Tensor::<Dev, 2>::from_data(TensorData::new(tap.clone(), [rows, order]), &device),
        )
        .to_data_as::<f32>()
        .to_vec_as::<f32>()
        .ok_or_panic();

        for r in 0..rows {
            let row = &sig[r * (order + out)..(r + 1) * (order + out)];
            for n in 0..out {
                let m = order + n;
                let mut want = row[m];
                for j in 0..order {
                    want += tap[r * order + j] * row[m - 1 - j];
                }
                assert!(
                    (got[r * out + n] - want).abs() < 1e-5,
                    "row {r}, sample {n}: {} vs {want}",
                    got[r * out + n],
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "tap rows")]
    fn test_residual_rejects_mismatched_rows() {
        let device = Default::default();
        lpc_residual_batched(
            Tensor::<Dev, 2>::zeros([2, 8], &device),
            Tensor::<Dev, 2>::zeros([3, 4], &device),
        );
    }

    #[test]
    #[should_panic(expected = "samples of window")]
    fn test_residual_rejects_a_short_window() {
        let device = Default::default();
        lpc_residual_batched(
            Tensor::<Dev, 2>::zeros([1, 4], &device),
            Tensor::<Dev, 2>::zeros([1, 4], &device),
        );
    }

    #[test]
    #[should_panic(expected = "at least lags 0 and 1")]
    fn test_degenerate_input_is_rejected() {
        let mut lpc: [f32; 0] = [];
        levinson_durbin(&[1.0], &mut lpc, None);
    }
}
