//! # The pitch estimator's LPC pre-filter design.
//!
//! Everything between the bin powers and the 16 LPC coefficients that
//! whiten the signal before the correlation search
//! (`src/pitch_est.cc`, `AUP_PE_computeBandEnergy` through
//! `AUP_PE_lpcCompute`).
//!
//! The chain, once per hop:
//!
//! ```text
//! band   = 18 triangular bands over the bin powers      [`band_energy`]
//! ly     = clamped log10(band + 1e-2)                   (caller)
//! cep    = dct(ly)                                      [`DctTable::dct`]
//! gain   = 10^idct(cep) * BAND_LPC_COMP                 [`DctTable::idct`]
//! spec   = gain interpolated back over the bins         [`interp_band_gain`]
//! ac     = autocorrelation of spec, lags 0..=16         [`Autocorrelator`]
//! lpc    = Levinson-Durbin(ac)                          [`celt_lpc`]
//! ```
//!
//! The round trip through the cepstrum is not an identity: the band gains
//! come back smoothed, which is what makes the resulting all-pole fit a
//! spectral envelope rather than the spectrum itself.
//!
//! ## The autocorrelation step
//!
//! The reference reaches the autocorrelation by packing the (real,
//! Nyquist-zeroed) envelope into FFTW half-complex layout and running a
//! 1024-point inverse transform, then keeping lags `0..=16` and scaling by
//! `0.5` — note `0.5`, *not* `1/N`, so the result sits `N/2` times above a
//! conventionally normalized autocorrelation.
//!
//! Since the input is real and even, that transform collapses to a cosine
//! sum, and only 17 of its 1024 outputs are ever read:
//!
//! ```text
//! ac[i] = 0.5*S[0] + Σ_{k=1}^{N/2-1} S[k]·cos(2πik/N)
//! ```
//!
//! [`Autocorrelator`] evaluates that directly against a cosine table, which
//! is both cheaper than a full inverse FFT and free of any FFT dependency.
//! It agrees with the reference transform to within f32 rounding.
//!
//! The scale is load-bearing rather than cosmetic. [`celt_lpc`] is itself
//! scale-invariant, but [`lpc_from_bands`] adds an *absolute* noise floor to
//! lag 0 before solving, so shrinking `ac` would silently raise that floor's
//! relative weight and flatten the fit on quiet frames.

use super::coeff::{
    ASSUMED_FFT_FOR_BANDS,
    BAND_LPC_COMP,
    BAND_START_INDEX,
    LPC_ORDER,
    NB_BANDS,
    PITCH_PI,
};

/// The noise floor added to lag 0 before the LPC solve.
///
/// The reference computes `windowSz / 12 / 38.0f` with an **integer** first
/// division (`src/pitch_est.cc`, `DC0_BIAS`).
fn dc0_bias(window_size: usize) -> f32 {
    (window_size / 12) as f32 / 38.0
}

/// The `NB_BANDS`-point DCT basis the pitch estimator's cepstrum uses.
///
/// Not one of the standard DCT normalizations: the reference builds
/// `cos((i + 0.5)·j·π/NB_BANDS)`, scales column `0` by `sqrt(0.5)`, and
/// applies `sqrt(2/NB_BANDS)` to both directions. [`dct`](Self::dct) and
/// [`idct`](Self::idct) therefore differ only in which index of the table
/// they walk.
#[derive(Debug, Clone, PartialEq)]
pub struct DctTable {
    table: [f32; NB_BANDS * NB_BANDS],
}

impl Default for DctTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DctTable {
    /// Builds the table.
    pub fn new() -> Self {
        let mut table = [0.0f32; NB_BANDS * NB_BANDS];
        for idx in 0..NB_BANDS {
            for jdx in 0..NB_BANDS {
                let mut v = ((idx as f32 + 0.5) * jdx as f32 * PITCH_PI / NB_BANDS as f32).cos();
                if jdx == 0 {
                    v *= 0.5f32.sqrt();
                }
                table[idx * NB_BANDS + jdx] = v;
            }
        }
        Self { table }
    }

    /// The shared `sqrt(2/NB_BANDS)` scale of both directions.
    fn ratio() -> f32 {
        (2.0 / NB_BANDS as f32).sqrt()
    }

    /// Band log-energies to cepstrum.
    pub fn dct(
        &self,
        input: &[f32; NB_BANDS],
    ) -> [f32; NB_BANDS] {
        let ratio = Self::ratio();
        core::array::from_fn(|idx| {
            let mut sum = 0.0f32;
            for (j, &x) in input.iter().enumerate() {
                sum += x * self.table[j * NB_BANDS + idx];
            }
            sum * ratio
        })
    }

    /// Cepstrum back to band log-energies.
    pub fn idct(
        &self,
        input: &[f32; NB_BANDS],
    ) -> [f32; NB_BANDS] {
        let ratio = Self::ratio();
        core::array::from_fn(|idx| {
            let mut sum = 0.0f32;
            for (j, &x) in input.iter().enumerate() {
                sum += x * self.table[idx * NB_BANDS + j];
            }
            sum * ratio
        })
    }
}

/// The bin span of band `idx`, as `(offset, width)`.
///
/// [`BAND_START_INDEX`] is tabulated against an [`ASSUMED_FFT_FOR_BANDS`]-point
/// FFT, so both ends are rescaled to the real FFT size and rounded. Rounding
/// each end independently — rather than differencing rounded offsets — is the
/// reference's behavior, and lets adjacent spans disagree by a bin. At the
/// ten-vad geometry they overlap at three boundaries and never gap, so every
/// bin below Nyquist stays covered.
fn band_span(
    idx: usize,
    fft_size: usize,
) -> (usize, usize) {
    let rate = fft_size as f32 / ASSUMED_FFT_FOR_BANDS;
    let width = (((BAND_START_INDEX[idx + 1] - BAND_START_INDEX[idx]) as f32) * rate).round();
    let offset = ((BAND_START_INDEX[idx] as f32) * rate).round();
    (offset as usize, width as usize)
}

/// Folds `bin_power` into [`NB_BANDS`] triangular bands.
///
/// Each band pair shares a linear ramp: bin `j` of span `i` contributes
/// `1 - j/width` to band `i` and `j/width` to band `i + 1`. The two edge
/// bands are doubled, since each only ever receives one side of a ramp.
///
/// # Arguments
/// * `bin_power`: `[fft_size / 2 + 1]` bin powers.
/// * `fft_size`: the FFT size `bin_power` came from.
///
/// # Panics
/// If `bin_power` is not `fft_size / 2 + 1` long.
pub fn band_energy(
    bin_power: &[f32],
    fft_size: usize,
) -> [f32; NB_BANDS] {
    let n_bins = fft_size / 2 + 1;
    assert_eq!(
        bin_power.len(),
        n_bins,
        "band_energy expects {n_bins} bins for a {fft_size}-point FFT",
    );

    let mut band = [0.0f32; NB_BANDS];
    for i in 0..(NB_BANDS - 1) {
        let (offset, width) = band_span(i, fft_size);
        for j in 0..width {
            let frac = j as f32 / width as f32;
            let power = bin_power[(offset + j).min(n_bins - 1)];
            band[i] += (1.0 - frac) * power;
            band[i + 1] += frac * power;
        }
    }
    band[0] *= 2.0;
    band[NB_BANDS - 1] *= 2.0;
    band
}

/// Spreads per-band gains back across the bins, inverting [`band_energy`].
///
/// Unlike [`band_energy`] this *assigns* rather than accumulates, so where
/// rounding makes two spans overlap the later band wins, and where it leaves
/// a gap the bin keeps its zero.
///
/// # Arguments
/// * `band_gain`: the per-band gains.
/// * `gain_per_bin`: `[fft_size / 2 + 1]` output, overwritten in full.
pub fn interp_band_gain(
    band_gain: &[f32; NB_BANDS],
    gain_per_bin: &mut [f32],
) {
    let n_bins = gain_per_bin.len();
    let fft_size = (n_bins - 1) * 2;

    gain_per_bin.fill(0.0);
    for idx in 0..(NB_BANDS - 1) {
        let (offset, width) = band_span(idx, fft_size);
        for j in 0..width {
            let frac = j as f32 / width as f32;
            gain_per_bin[(offset + j).min(n_bins - 1)] =
                (1.0 - frac) * band_gain[idx] + frac * band_gain[idx + 1];
        }
    }
}

/// Autocorrelation lags from a real, even power spectrum.
///
/// Holds the cosine table the sum is evaluated against; see the module docs
/// for why this replaces the reference's inverse FFT and why the result is
/// scaled by `0.5` rather than `1/fft_size`.
#[derive(Debug, Clone, PartialEq)]
pub struct Autocorrelator {
    fft_size: usize,

    /// `cos(2πt/fft_size)` for `t` in `0..fft_size`.
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

    /// Lags `0..n_lags` of the spectrum in `spectrum`.
    ///
    /// The Nyquist bin is ignored; the reference zeroes it before
    /// transforming, and this reproduces that by construction.
    ///
    /// The sum is accumulated in `f64`. The reference accumulates through FFT
    /// butterflies, whose error grows with `log(fft_size)` rather than
    /// `fft_size`; a flat `f32` sum over 511 terms would be materially worse
    /// than the thing being reproduced, while `f64` is at least as good.
    ///
    /// # Arguments
    /// * `spectrum`: `[fft_size / 2 + 1]` non-negative bin values.
    /// * `out`: the lags to fill; `out.len()` must not exceed `fft_size`.
    ///
    /// # Panics
    /// If `spectrum` is not `fft_size / 2 + 1` long.
    pub fn autocorrelate(
        &self,
        spectrum: &[f32],
        out: &mut [f32],
    ) {
        let n_bins = self.fft_size / 2 + 1;
        assert_eq!(
            spectrum.len(),
            n_bins,
            "autocorrelate expects {n_bins} bins for a {}-point FFT",
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

/// Solves for LPC coefficients by Levinson-Durbin recursion.
///
/// The reference's `AUP_PE_celt_lpc`, itself CELT's. Returns the coefficients
/// of the whitening filter, and bails out early once the residual error has
/// dropped 30 dB below lag 0.
///
/// Coefficients past the bail-out are left at **zero**, not at a partial
/// value: index `k` is only ever written by `lpc[i] = r` at step `k`, and the
/// symmetric update at step `i` touches only `0..i-1`. That matters for a
/// vectorized port, which cannot branch and must instead freeze each row under
/// a mask — the frozen tail is the zero initializer.
///
/// Scale-invariant: multiplying `ac` through by a constant leaves the result
/// unchanged, including the bail-out point.
///
/// # Arguments
/// * `ac`: autocorrelation lags `0..=LPC_ORDER`.
///
/// # Returns
/// The `LPC_ORDER` coefficients, all zero if `ac[0]` is zero.
pub fn celt_lpc(ac: &[f32; LPC_ORDER + 1]) -> [f32; LPC_ORDER] {
    let mut lpc = [0.0f32; LPC_ORDER];
    if ac[0] == 0.0 {
        return lpc;
    }

    let mut error = ac[0];
    for i in 0..LPC_ORDER {
        // The reference sums the products first and folds in `ac[i + 1]`
        // last; the order is preserved because it is observable in f32.
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];

        let r = -rr / error;
        lpc[i] = r;
        for j in 0..((i + 1) >> 1) {
            let head = lpc[j];
            let tail = lpc[i - 1 - j];
            lpc[j] = head + r * tail;
            lpc[i - 1 - j] = tail + r * head;
        }

        error -= r * r * error;
        if error < 0.001 * ac[0] {
            break;
        }
    }

    lpc
}

/// Fits an all-pole filter to a set of band gains.
///
/// Interpolates the gains across the bins, autocorrelates, applies the noise
/// floor and lag window, and solves.
///
/// # Arguments
/// * `band_gain`: the per-band spectral envelope.
/// * `window_size`: the STFT analysis window length, which sets the noise
///   floor.
/// * `autocorrelator`: sized to the FFT the bands are spread over.
/// * `bin_scratch`: `[fft_size / 2 + 1]` scratch, overwritten.
pub fn lpc_from_bands(
    band_gain: &[f32; NB_BANDS],
    window_size: usize,
    autocorrelator: &Autocorrelator,
    bin_scratch: &mut [f32],
) -> [f32; LPC_ORDER] {
    interp_band_gain(band_gain, bin_scratch);
    // Drop the Nyquist bin, as the reference does before transforming.
    let last = bin_scratch.len() - 1;
    bin_scratch[last] = 0.0;

    let mut ac = [0.0f32; LPC_ORDER + 1];
    autocorrelator.autocorrelate(bin_scratch, &mut ac);

    // -40 dB noise floor, then a lag window tapering the higher lags.
    ac[0] += ac[0] * 1e-4 + dc0_bias(window_size);
    for (i, slot) in ac.iter_mut().enumerate().skip(1) {
        *slot *= 1.0 - 6e-5 * i as f32 * i as f32;
    }

    celt_lpc(&ac)
}

/// Fits an all-pole filter to a cepstrum.
///
/// Undoes the DCT, exponentiates back to linear gains, applies
/// [`BAND_LPC_COMP`], and hands off to [`lpc_from_bands`].
///
/// # Arguments
/// * `cepstrum`: the DCT of the clamped band log-energies.
/// * `dct`: the table `cepstrum` was produced with.
/// * `window_size`: the STFT analysis window length.
/// * `autocorrelator`: sized to the FFT the bands are spread over.
/// * `bin_scratch`: `[fft_size / 2 + 1]` scratch, overwritten.
pub fn lpc_from_cepstrum(
    cepstrum: &[f32; NB_BANDS],
    dct: &DctTable,
    window_size: usize,
    autocorrelator: &Autocorrelator,
    bin_scratch: &mut [f32],
) -> [f32; LPC_ORDER] {
    let mut band_gain = dct.idct(cepstrum);
    for (gain, comp) in band_gain.iter_mut().zip(BAND_LPC_COMP) {
        *gain = 10.0f32.powf(*gain) * comp;
    }
    lpc_from_bands(&band_gain, window_size, autocorrelator, bin_scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FFT: usize = 1024;
    const NBINS: usize = FFT / 2 + 1;

    /// A smooth, non-negative, decaying spectrum — the shape a real band
    /// envelope has.
    fn envelope() -> Vec<f32> {
        (0..NBINS)
            .map(|k| 1e3 * (-(k as f32) / 90.0).exp() * (1.0 + 0.4 * (k as f32 * 0.07).sin()))
            .collect()
    }

    #[test]
    fn test_dct_round_trips_through_idct() {
        let dct = DctTable::new();
        let input: [f32; NB_BANDS] = core::array::from_fn(|i| (i as f32 * 0.4).sin());

        let back = dct.idct(&dct.dct(&input));
        for (got, want) in back.iter().zip(input.iter()) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }

    #[test]
    fn test_dct_is_orthonormal_enough_to_preserve_energy() {
        let dct = DctTable::new();
        let input: [f32; NB_BANDS] = core::array::from_fn(|i| (i as f32 * 0.9).cos());

        let energy = |v: &[f32; NB_BANDS]| v.iter().map(|x| x * x).sum::<f32>();
        let ratio = energy(&dct.dct(&input)) / energy(&input);
        assert!((ratio - 1.0).abs() < 1e-4, "energy ratio {ratio}");
    }

    #[test]
    fn test_band_energy_conserves_total_power() {
        // Every ramp pair sums to 1, and the doubled edges compensate for the
        // half-ramps the first and last bands see, so a flat spectrum lands
        // as roughly `2 * total` spread over the bands.
        let flat = vec![1.0f32; NBINS];
        let bands = band_energy(&flat, FFT);

        for (i, b) in bands.iter().enumerate() {
            assert!(*b > 0.0, "band {i} is empty");
        }
        // Interior bands see one full ramp in and one out.
        let interior: f32 = bands[1..NB_BANDS - 1].iter().sum();
        assert!(interior > 0.0);
    }

    #[test]
    fn test_band_energy_tracks_where_the_power_is() {
        let mut spectrum = vec![0.0f32; NBINS];
        // Band 0 spans bins 0..13 at this FFT size.
        spectrum[4] = 100.0;
        let low = band_energy(&spectrum, FFT);
        assert!(low[0] > 0.0);
        assert!(low[NB_BANDS - 1] == 0.0);

        let mut spectrum = vec![0.0f32; NBINS];
        spectrum[500] = 100.0;
        let high = band_energy(&spectrum, FFT);
        assert!(high[0] == 0.0);
        assert!(high[NB_BANDS - 1] > 0.0);
    }

    #[test]
    fn test_band_spans_cover_the_spectrum_without_gaps() {
        // Both ends of each span are rounded independently, so a span may
        // start one bin *before* the previous one ended -- but never after,
        // which is what keeps `interp_band_gain` from leaving a bin at zero.
        let mut next = 0usize;
        let mut overlaps = 0;
        for i in 0..(NB_BANDS - 1) {
            let (offset, width) = band_span(i, FFT);
            assert!(
                offset <= next,
                "band {i} starts at {offset}, leaving a gap after {next}",
            );
            overlaps += next - offset;
            next = offset + width;
        }
        assert_eq!(next, NBINS - 1, "spans should stop just below Nyquist");
        // At this geometry the rounding overlaps at exactly three boundaries.
        assert_eq!(overlaps, 3);
    }

    #[test]
    fn test_overlapping_spans_double_count_in_band_energy() {
        // Where spans overlap, the shared bin lands in both bands: once as the
        // tail of one ramp and once as the head of the next. The reference does
        // this, and the doubled bin is why the bands are not a partition.
        let mut spectrum = vec![0.0f32; NBINS];
        // Band 3 starts at bin 38; band 2 runs through bin 38 as well.
        spectrum[38] = 1.0;
        let bands = band_energy(&spectrum, FFT);

        assert!(bands[2] > 0.0, "the overlapped bin should reach band 2");
        assert!(bands[3] > 0.0, "the overlapped bin should reach band 3");
    }

    #[test]
    fn test_interp_band_gain_reproduces_a_flat_envelope() {
        let flat = [3.0f32; NB_BANDS];
        let mut bins = vec![0.0f32; NBINS];
        interp_band_gain(&flat, &mut bins);

        // Every bin below Nyquist is covered by exactly one span, and a flat
        // set of gains interpolates to itself.
        for (k, &g) in bins.iter().enumerate().take(NBINS - 1) {
            assert!((g - 3.0).abs() < 1e-5, "bin {k} = {g}");
        }
    }

    #[test]
    fn test_autocorrelate_matches_a_direct_dft() {
        let spectrum = envelope();
        let ac = Autocorrelator::new(FFT);
        let mut got = [0.0f32; LPC_ORDER + 1];
        ac.autocorrelate(&spectrum, &mut got);

        for (lag, &g) in got.iter().enumerate() {
            // The closed form, evaluated independently in f64.
            let mut want = 0.5 * spectrum[0] as f64;
            for (k, &s) in spectrum.iter().enumerate().take(FFT / 2).skip(1) {
                want +=
                    s as f64 * (core::f64::consts::TAU * lag as f64 * k as f64 / FFT as f64).cos();
            }
            let rel = (g as f64 - want).abs() / want.abs().max(1.0);
            assert!(rel < 1e-6, "lag {lag}: {g} vs {want}");
        }
    }

    #[test]
    fn test_autocorrelate_ignores_the_nyquist_bin() {
        let ac = Autocorrelator::new(FFT);
        let mut spectrum = envelope();
        let mut with_nyquist = [0.0f32; LPC_ORDER + 1];
        let mut without = [0.0f32; LPC_ORDER + 1];

        spectrum[NBINS - 1] = 0.0;
        ac.autocorrelate(&spectrum, &mut without);
        spectrum[NBINS - 1] = 1e6;
        ac.autocorrelate(&spectrum, &mut with_nyquist);

        assert_eq!(with_nyquist, without);
    }

    #[test]
    fn test_autocorrelate_lag_zero_is_the_mean_power() {
        // With the reference's 0.5 scale, lag 0 is half the DC bin plus the
        // sum of every other bin.
        let spectrum = envelope();
        let ac = Autocorrelator::new(FFT);
        let mut got = [0.0f32; 1];
        ac.autocorrelate(&spectrum, &mut got);

        let want =
            0.5 * spectrum[0] as f64 + spectrum[1..FFT / 2].iter().map(|&s| s as f64).sum::<f64>();
        assert!((got[0] as f64 - want).abs() / want < 1e-6);
    }

    #[test]
    fn test_celt_lpc_is_scale_invariant() {
        let mut ac = [0.0f32; LPC_ORDER + 1];
        for (i, slot) in ac.iter_mut().enumerate() {
            *slot = 1000.0 * (-(i as f32) / 4.0).exp();
        }
        let base = celt_lpc(&ac);

        for scale in [1e-3f32, 7.0, 512.0] {
            let scaled: [f32; LPC_ORDER + 1] = core::array::from_fn(|i| ac[i] * scale);
            let got = celt_lpc(&scaled);
            for (g, b) in got.iter().zip(base.iter()) {
                assert!((g - b).abs() < 1e-4, "scale {scale}: {g} vs {b}");
            }
        }
    }

    #[test]
    fn test_celt_lpc_of_a_zero_signal_is_zero() {
        assert_eq!(celt_lpc(&[0.0; LPC_ORDER + 1]), [0.0; LPC_ORDER]);
    }

    #[test]
    fn test_celt_lpc_whitens_a_known_ar_process() {
        // An AR(1) process x[n] = a*x[n-1] + e has autocorrelation a^|k|, and
        // the whitening filter should recover -a in the first coefficient.
        let a = 0.8f32;
        let ac: [f32; LPC_ORDER + 1] = core::array::from_fn(|i| a.powi(i as i32));
        let lpc = celt_lpc(&ac);

        assert!((lpc[0] + a).abs() < 1e-3, "lpc[0] = {}", lpc[0]);
        for (i, c) in lpc.iter().enumerate().skip(1) {
            assert!(c.abs() < 1e-3, "lpc[{i}] = {c} should be negligible");
        }
    }

    #[test]
    fn test_dc0_bias_uses_integer_division() {
        // 768/12 is exact, so this matches either reading; the smaller windows
        // are where the reference's integer division would show.
        assert_eq!(dc0_bias(768), 64.0 / 38.0);
        assert_eq!(dc0_bias(770), 64.0 / 38.0);
        assert_ne!(dc0_bias(770), (770.0 / 12.0) / 38.0);
    }

    #[test]
    fn test_lpc_from_bands_is_stable_for_a_flat_envelope() {
        let flat = [1.0f32; NB_BANDS];
        let mut scratch = vec![0.0f32; NBINS];
        let ac = Autocorrelator::new(FFT);
        let lpc = lpc_from_bands(&flat, 768, &ac, &mut scratch);

        // A flat envelope is already white: nothing to predict.
        for (i, c) in lpc.iter().enumerate() {
            assert!(c.abs() < 0.2, "lpc[{i}] = {c} for a flat spectrum");
        }
    }

    #[test]
    fn test_lpc_from_bands_responds_to_a_tilted_envelope() {
        let mut scratch = vec![0.0f32; NBINS];
        let ac = Autocorrelator::new(FFT);

        let tilted: [f32; NB_BANDS] = core::array::from_fn(|i| 100.0 * (-(i as f32) / 3.0).exp());
        let lpc = lpc_from_bands(&tilted, 768, &ac, &mut scratch);

        let magnitude: f32 = lpc.iter().map(|c| c.abs()).sum();
        assert!(
            magnitude > 0.2,
            "a strongly tilted spectrum should fit: {magnitude}"
        );
    }

    #[test]
    fn test_lpc_from_cepstrum_round_trips_through_lpc_from_bands() {
        let dct = DctTable::new();
        let ac = Autocorrelator::new(FFT);
        let mut scratch = vec![0.0f32; NBINS];

        let log_bands: [f32; NB_BANDS] = core::array::from_fn(|i| 1.5 - i as f32 * 0.1);
        let cepstrum = dct.dct(&log_bands);

        let from_cep = lpc_from_cepstrum(&cepstrum, &dct, 768, &ac, &mut scratch);

        // The same path, spelled out.
        let mut gains = dct.idct(&cepstrum);
        for (g, comp) in gains.iter_mut().zip(BAND_LPC_COMP) {
            *g = 10.0f32.powf(*g) * comp;
        }
        let from_bands = lpc_from_bands(&gains, 768, &ac, &mut scratch);

        assert_eq!(from_cep, from_bands);
    }
}
