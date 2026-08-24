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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    #[should_panic(expected = "at least lags 0 and 1")]
    fn test_degenerate_input_is_rejected() {
        let mut lpc: [f32; 0] = [];
        levinson_durbin(&[1.0], &mut lpc, None);
    }
}
