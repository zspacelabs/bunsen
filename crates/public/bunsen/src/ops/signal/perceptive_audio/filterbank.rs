//! # Mel scale and triangular filterbank.
//!
//! Pure host-side construction in `Vec<f64>`; nothing here touches a
//! `Backend`. The device-side filterbank is lifted from this by
//! [`MelConverter`](super::PerceptiveAudioConverter), and this module doubles
//! as the reference the tensor path is checked against.
//!
//! The construction follows `librosa.filters.mel`: triangles are laid out on
//! evenly spaced mel points, evaluated against the `rfft` bin centres, and
//! optionally area-normalized.

use std::ops::Range;

use burn::config::Config;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::arange::vec_linspace,
};

/// Slaney: Hz per mel across the linear region.
const SLANEY_F_SP: f64 = 200.0 / 3.0;

/// Slaney: the linear/logarithmic breakpoint, in Hz.
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;

/// Slaney: the breakpoint in `perceptive_audio`, `SLANEY_MIN_LOG_HZ /
/// SLANEY_F_SP`.
///
/// Exactly 15; the Slaney curve is pinned so 1000 Hz lands on a whole mel.
const SLANEY_MIN_LOG_MEL: f64 = 15.0;

/// Slaney's logarithmic step, `6.4.ln() / 27`.
const SLANEY_LOGSTEP: f64 = 1.8562979903656263 /* 6.4.ln() */ / 27.0;

/// Selects the frequency-to-mel warping curve.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum MelScale {
    /// Slaney's Auditory Toolbox curve, as used by `librosa` (`htk=False`)
    /// and by Whisper.
    ///
    /// Linear below 1000 Hz at `200/3` Hz per mel, logarithmic above:
    /// `mel = 15 + ln(hz / 1000) / (ln(6.4) / 27)`.
    Slaney,

    /// The HTK curve: `mel = 2595 * log10(1 + hz / 700)`.
    Htk,
}

impl MelScale {
    /// Converts a frequency in Hz to `perceptive_audio`.
    ///
    /// Anchors: HTK 1000 Hz is `999.9855371396244`; Slaney 1000 Hz is exactly
    /// `15.0`, and Slaney 8000 Hz is `45.245640471924965`.
    pub fn hz_to_mel(
        &self,
        hz: f64,
    ) -> f64 {
        match self {
            Self::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
            Self::Slaney => {
                if hz < SLANEY_MIN_LOG_HZ {
                    hz / SLANEY_F_SP
                } else {
                    SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / SLANEY_LOGSTEP
                }
            }
        }
    }

    /// Converts `perceptive_audio` to a frequency in Hz.
    ///
    /// The inverse of [`hz_to_mel`](Self::hz_to_mel).
    pub fn mel_to_hz(
        &self,
        mel: f64,
    ) -> f64 {
        match self {
            Self::Htk => 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0),
            Self::Slaney => {
                if mel < SLANEY_MIN_LOG_MEL {
                    mel * SLANEY_F_SP
                } else {
                    SLANEY_MIN_LOG_HZ * (SLANEY_LOGSTEP * (mel - SLANEY_MIN_LOG_MEL)).exp()
                }
            }
        }
    }
}

/// Triangle area normalization.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum FilterNorm {
    /// Slaney normalization, as used by `librosa`'s `norm="slaney"`: each
    /// triangle is scaled by `2 / (f_hi - f_lo)` so it encloses unit area
    /// regardless of width.
    Slaney,

    /// No normalization; every triangle peaks at 1.
    None,
}

impl FilterNorm {
    /// The scale factor for a triangle spanning `[f_lo, f_hi]` Hz.
    ///
    /// Slaney scales by enclosed area, `2 / (f_hi - f_lo)`, so wide
    /// high-frequency triangles do not outweigh narrow low-frequency ones.
    pub fn gain(
        &self,
        f_lo: f64,
        f_hi: f64,
    ) -> f64 {
        match self {
            Self::Slaney => 2.0 / (f_hi - f_lo),
            Self::None => 1.0,
        }
    }
}

/// Builds the `n_points` mel-spaced frequencies spanning `[f_min, f_max]`.
///
/// The points are evenly spaced *in `perceptive_audio`* and returned in Hz, so
/// the first is `f_min` and the last is `f_max`. A filterbank of `n_mels`
/// triangles wants `n_mels + 2` of these: each triangle spans one point either
/// side of its centre.
///
/// # Arguments
/// * `n_points`: how many points to place; must be at least 2.
/// * `f_min`, `f_max`: the span, in Hz.
/// * `scale`: the warping curve.
pub fn mel_points(
    n_points: usize,
    f_min: f64,
    f_max: f64,
    scale: MelScale,
) -> Vec<f64> {
    assert!(
        n_points >= 2,
        "mel_points n_points ({n_points}) must be >= 2",
    );

    let mel_min = scale.hz_to_mel(f_min);
    let mel_max = scale.hz_to_mel(f_max);

    vec_linspace(mel_min, mel_max, n_points)
        .into_iter()
        .map(|mel| scale.mel_to_hz(mel))
        .collect()
}

/// Configures a row-major `[n_mels, n_bins]` triangular mel filterbank.
///
/// `n_bins` is `n_fft / 2 + 1`, matching the `rfft` bin count, and bin `j`
/// sits at `j * sample_rate / n_fft` Hz. Triangle `i` rises from
/// `mel_points[i]` to `mel_points[i + 1]` and falls to `mel_points[i + 2]`.
#[derive(Config, Debug)]
pub struct MelFilterbankConfig {
    /// The sample rate, in Hz.
    pub sample_rate: usize,

    /// The FFT length the spectrum was taken at.
    pub n_fft: usize,

    /// How many triangles to build.
    pub n_mels: usize,

    /// The span covered by the bank, in Hz.
    pub f_range: Range<f64>,

    /// The frequency-to-mel warping curve.
    #[config(default = "MelScale::Slaney")]
    pub scale: MelScale,

    /// Triangle area normalization.
    #[config(default = "FilterNorm::Slaney")]
    pub norm: FilterNorm,
}

impl MelFilterbankConfig {
    /// Builds a row-major `[n_mels, n_bins]` triangular mel filterbank.
    ///
    /// `n_bins` is `n_fft / 2 + 1`, matching the `rfft` bin count, and bin `j`
    /// sits at `j * sample_rate / n_fft` Hz. Triangle `i` rises from
    /// `mel_points[i]` to `mel_points[i + 1]` and falls to `mel_points[i + 2]`.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if `n_fft` or `n_mels` is zero, if
    /// `f_min >= f_max`, if the mel points are not strictly increasing (which
    /// would make a triangle degenerate), or if any triangle ends up **empty**
    /// — covering no `rfft` bin at all. An empty row silently zeroes a
    /// whole mel channel, so it is rejected rather than returned; it means
    /// `n_mels` is too large for this `n_fft` (`n_fft = 256` with `n_mels =
    /// 128` at 16 kHz leaves 13 rows empty).
    pub fn try_to_vec(&self) -> BunsenResult<Vec<f64>> {
        if self.n_fft == 0 {
            return Err(BunsenError::Invalid(
                "MelFilterbank n_fft must be non-zero".to_string(),
            ));
        }
        if self.n_mels == 0 {
            return Err(BunsenError::Invalid(
                "MelFilterbank n_mels must be non-zero".to_string(),
            ));
        }
        let f_min = self.f_range.start;
        let f_max = self.f_range.end;
        if f_min >= f_max {
            return Err(BunsenError::Invalid(format!(
                "MelFilterbank f_min ({f_min}) must be < f_max ({f_max})",
            )));
        }

        let n_bins = self.n_fft / 2 + 1;

        // The `rfft` bin centres: `n_bins` points from DC to Nyquist.
        let bin_hz = vec_linspace(0.0, self.sample_rate as f64 / 2.0, n_bins);

        // One point either side of each triangle's center.
        let points = mel_points(self.n_mels + 2, f_min, f_max, self.scale);

        for w in points.windows(2) {
            if w[0] >= w[1] {
                return Err(BunsenError::Invalid(format!(
                    "MelFilterbank mel points are not strictly increasing \
                 ({} then {}); n_mels ({}) is too large for the \
                 [{f_min}, {f_max}] Hz span",
                    w[0], w[1], self.n_mels
                )));
            }
        }

        let mut bank = vec![0.0_f64; self.n_mels * n_bins];

        for i in 0..self.n_mels {
            let (f_lo, f_ct, f_hi) = (points[i], points[i + 1], points[i + 2]);

            let gain = self.norm.gain(f_lo, f_hi);

            let row = &mut bank[i * n_bins..(i + 1) * n_bins];
            for (slot, &hz) in row.iter_mut().zip(&bin_hz) {
                let rising = (hz - f_lo) / (f_ct - f_lo);
                let falling = (f_hi - hz) / (f_hi - f_ct);

                *slot = rising.min(falling).max(0.0) * gain;
            }

            if row.iter().all(|&v| v == 0.0) {
                return Err(BunsenError::Invalid(format!(
                    "MelFilterbank triangle {i} spanning [{f_lo}, {f_hi}] Hz \
                 covers no rfft bin (bin spacing {} Hz); n_mels ({}) is \
                 too large for n_fft ({})",
                    self.sample_rate as f64 / self.n_fft as f64,
                    self.n_mels,
                    self.n_fft
                )));
            }
        }

        Ok(bank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts `actual` matches `expected` to a relative tolerance.
    fn assert_rel(
        actual: f64,
        expected: f64,
        rel: f64,
    ) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= rel * scale,
            "expected {expected}, got {actual} (rel tol {rel})",
        );
    }

    /// Position-weighted checksum, `Σ bank[i, j] * (i + 1) * (j + 1)`.
    ///
    /// Sensitive to every weight's value *and* its placement, so a bank that
    /// is transposed, shifted by a bin, or normalized per-column still fails
    /// even when the plain sum happens to match.
    fn weighted_checksum(
        bank: &[f64],
        n_mels: usize,
        n_bins: usize,
    ) -> f64 {
        assert_eq!(bank.len(), n_mels * n_bins);
        let mut acc = 0.0;
        for i in 0..n_mels {
            for j in 0..n_bins {
                acc += bank[i * n_bins + j] * (i + 1) as f64 * (j + 1) as f64;
            }
        }
        acc
    }

    /// Structural pins — the sum, the maximum, and the position-weighted
    /// checksum of the bank — computed from a transcription of
    /// `librosa.filters.mel`.
    ///
    /// `cross_test::test_filterbank_matches_librosa` compares the whole bank
    /// against real `librosa` output and is the ground truth. These stay
    /// because they need no fixture file and fail with a different signature:
    /// a bank that is transposed, shifted by a bin, or normalized along the
    /// wrong axis breaks them even when its element values are right.
    const SLANEY_16K_400_80_SUM: f64 = 1.999024102917857;
    const SLANEY_16K_400_80_MAX: f64 = 0.025880684545275;
    const SLANEY_16K_400_80_CHK: f64 = 7175.379677895588;

    #[test]
    fn test_mel_scale_anchors() {
        // HTK: `2595 * log10(1 + hz/700)`. Note this is *not* exactly 1000
        // mel at 1000 Hz — that anchor belongs to the 1127*ln convention.
        assert_rel(MelScale::Htk.hz_to_mel(0.0), 0.0, 1e-12);
        assert_rel(MelScale::Htk.hz_to_mel(1000.0), 999.9855371396244, 1e-12);
        assert_rel(MelScale::Htk.hz_to_mel(8000.0), 2840.023046708319, 1e-12);

        // Slaney is pinned so the breakpoint lands on a whole mel.
        assert_rel(MelScale::Slaney.hz_to_mel(0.0), 0.0, 1e-12);
        assert_eq!(MelScale::Slaney.hz_to_mel(1000.0), 15.0);
        assert_rel(
            MelScale::Slaney.hz_to_mel(8000.0),
            45.245640471924965,
            1e-12,
        );
    }

    #[test]
    fn test_mel_scale_roundtrips() {
        for scale in [MelScale::Slaney, MelScale::Htk] {
            // Sweep across the Slaney breakpoint, which is where a piecewise
            // inverse is most likely to be wrong.
            for step in 0..=160 {
                let hz = step as f64 * 50.0;
                let back = scale.mel_to_hz(scale.hz_to_mel(hz));
                assert_rel(back, hz, 1e-10);
            }
        }
    }

    #[test]
    fn test_slaney_is_continuous_at_the_breakpoint() {
        // The piecewise definition must not step at 1000 Hz.
        let below = MelScale::Slaney.hz_to_mel(SLANEY_MIN_LOG_HZ - 1e-6);
        let above = MelScale::Slaney.hz_to_mel(SLANEY_MIN_LOG_HZ + 1e-6);
        assert!(
            (above - below).abs() < 1e-7,
            "Slaney steps at the breakpoint: {below} then {above}",
        );
    }

    #[test]
    fn test_mel_points_spans_the_range() {
        for scale in [MelScale::Slaney, MelScale::Htk] {
            let points = mel_points(82, 0.0, 8000.0, scale);

            assert_eq!(points.len(), 82);
            assert_rel(points[0], 0.0, 1e-9);
            assert_rel(points[81], 8000.0, 1e-9);

            for w in points.windows(2) {
                assert!(w[0] < w[1], "not increasing: {} then {}", w[0], w[1]);
            }
        }
    }

    #[test]
    fn test_filterbank_matches_librosa_algorithm() {
        let sample_rate = 16_000;
        let n_fft = 400;
        let n_mels = 80;
        let n_bins = 201;
        let f_range = 0.0..8000.0;

        let bank = MelFilterbankConfig::new(sample_rate, n_fft, n_mels, f_range)
            .with_scale(MelScale::Slaney)
            .with_norm(FilterNorm::Slaney)
            .try_to_vec()
            .unwrap();

        assert_eq!(bank.len(), n_mels * n_bins);

        let sum: f64 = bank.iter().sum();
        let max = bank.iter().copied().fold(f64::MIN, f64::max);
        assert_rel(sum, SLANEY_16K_400_80_SUM, 1e-9);
        assert_rel(max, SLANEY_16K_400_80_MAX, 1e-9);
        assert_rel(
            weighted_checksum(&bank, n_mels, n_bins),
            SLANEY_16K_400_80_CHK,
            1e-9,
        );

        // Per-row anchors: the lowest triangles are narrow enough to cover
        // only one or two bins, which is exactly where an off-by-one in the
        // ramp arithmetic shows up.
        let row = |i: usize| &bank[i * n_bins..(i + 1) * n_bins];
        assert_rel(row(0).iter().sum(), 0.024862593984176, 1e-9);
        assert_rel(row(40).iter().sum(), 0.026664860534783, 1e-9);
        assert_rel(row(79).iter().sum(), 0.024925339738859, 1e-9);

        let argmax = |i: usize| {
            row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(j, _)| j)
                .unwrap()
        };
        assert_eq!(
            [argmax(0), argmax(1), argmax(40), argmax(79)],
            [1, 2, 43, 192]
        );

        // Row 0 touches exactly one bin.
        let nonzero: Vec<usize> = row(0)
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(j, _)| j)
            .collect();
        assert_eq!(nonzero, vec![1]);

        // Every row is reachable; `mel_filterbank` would have errored
        // otherwise, so this pins the guarantee rather than the arithmetic.
        for i in 0..n_mels {
            assert!(row(i).iter().any(|&v| v > 0.0), "row {i} is empty");
        }
    }

    #[test]
    fn test_filterbank_htk_without_norm() {
        let sample_rate = 16_000;
        let n_fft = 400;
        let n_mels = 80;
        let n_bins = 201;
        let f_range = 0.0..8000.0;

        let bank = MelFilterbankConfig::new(sample_rate, n_fft, n_mels, f_range)
            .with_scale(MelScale::Htk)
            .with_norm(FilterNorm::None)
            .try_to_vec()
            .unwrap();

        assert_rel(bank.iter().sum(), 196.151976953471774, 1e-9);
        assert_rel(
            weighted_checksum(&bank, n_mels, n_bins),
            1299649.153805326205,
            1e-9,
        );

        // Unnormalized triangles peak at 1, but only if a bin lands exactly
        // on a centre — none quite does here, so the peak sits just under.
        let max = bank.iter().copied().fold(f64::MIN, f64::max);
        assert!(max <= 1.0, "unnormalized weight exceeds 1: {max}");
        assert_rel(max, 0.998554715903963, 1e-9);
    }

    #[test]
    fn test_filterbank_rejects_empty_rows() {
        let sample_rate = 16_000;
        let n_fft = 256;
        let n_mels = 128;
        let f_range = 0.0..8000.0;

        // 128 triangles over 129 bins leaves 13 of them covering no bin.
        let err = MelFilterbankConfig::new(sample_rate, n_fft, n_mels, f_range)
            .with_scale(MelScale::Slaney)
            .with_norm(FilterNorm::Slaney)
            .try_to_vec();

        assert!(
            matches!(&err, Err(BunsenError::Invalid(m)) if m.contains("covers no rfft bin")),
            "expected an empty-triangle error, got {err:?}",
        );
    }

    #[test]
    fn test_filterbank_validation() {
        let ok = |n_fft, n_mels, f_range| {
            let sample_rate = 16_000;
            MelFilterbankConfig::new(sample_rate, n_fft, n_mels, f_range).try_to_vec()
        };

        for bad in [
            ok(0, 80, 0.0..8000.0),
            ok(400, 0, 0.0..8000.0),
            // f_min == f_max, and inverted.
            ok(400, 80, 8000.0..8000.0),
            ok(400, 80, 8000.0..0.0),
        ] {
            assert!(
                matches!(bad, Err(BunsenError::Invalid(_))),
                "expected Invalid, got {bad:?}",
            );
        }

        assert!(ok(400, 80, 0.0..8000.0).is_ok());
    }
}
