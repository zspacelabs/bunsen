//! # Pitch-estimator coefficients.
//!
//! The fixed constants the estimator is defined by: the band layout, the
//! period bounds, the anti-alias design and the tracker's weights.
//!
//! These are **reference data rather than tunables**. The band edges and
//! per-band compensation come from `LPCNet`'s feature front end, and a consumer
//! whose downstream model was fitted against these values cannot change them
//! without refitting that model. Geometry that a caller may legitimately vary
//! -- hop size, chunk size, filter length -- lives on the stage configs
//! instead.

/// The sample rate, in Hz, the estimator's constants are defined for.
///
/// The period bounds, band layout and anti-alias design below are all tied to
/// it; changing it alone does not rescale them.
pub const SAMPLE_RATE: usize = 16000;

/// The hop size, in samples, the estimator advances per call.
pub const HOP_SIZE: usize = 256;

/// The number of bands the pitch estimator's LPC front end works in.
///
/// Unrelated to the 40 mel bands of the feature path: these 18 bands exist
/// only to shape the LPC pre-filter.
pub const NB_BANDS: usize = 18;

/// The order of the LPC pre-filter.
pub const LPC_ORDER: usize = 16;

/// The shortest pitch period considered, in samples at 16 kHz (≈500 Hz).
pub const MIN_PERIOD_16KHZ: usize = 32;

/// The longest pitch period considered, in samples at 16 kHz (≈62 Hz).
pub const MAX_PERIOD_16KHZ: usize = 256;

const _: () = assert!(
    MAX_PERIOD_16KHZ > MIN_PERIOD_16KHZ,
    "the period search range must be non-empty",
);

/// How far the correlation branch lags the current hop, in samples at 16 kHz.
///
/// The reference reads its LPC-filter input from this many samples behind the
/// newest sample, so the excitation buffer is aligned with the correlation
/// window.
pub const XCORR_TRAINING_OFFSET: usize = 80;

/// The correlation history the tracker spans, in milliseconds.
pub const FEAT_TIME_WINDOW_MS: usize = 40;

/// The cap on the number of history frames, whatever the hop size implies.
pub const FEAT_MAX_NFRM: usize = 12;

/// The internal processing rate of the correlation branch, in Hz.
///
/// The reference decimates to this rate before correlating, which is what
/// makes the period search 64 lags wide instead of 256.
pub const PROC_FS: usize = 4000;

/// The decimation factor from 16 kHz to the correlation branch's rate.
pub const PROC_RESAMPLE_RATE: usize = SAMPLE_RATE / PROC_FS;

/// The frame-correlation above which a frame is called voiced.
///
/// The reference's `pitchEstVoicedThr` for a 4 kHz processing rate.
pub const VOICED_THRESHOLD: f32 = 0.4;

/// The per-step penalty weight of the Viterbi period tracker.
///
/// A candidate `jdx` steps from the current period is discounted by
/// `PITCH_MAX_PATH_W * jdx²`.
pub const PITCH_MAX_PATH_W: f32 = 0.02;

/// The FFT size the band layout in [`BAND_START_INDEX`] was tabulated for.
///
/// The reference rescales those indices by `fft_size / ASSUMED_FFT_FOR_BANDS`
/// rather than retabulating them, so this is a divisor, not a real FFT size.
pub const ASSUMED_FFT_FOR_BANDS: f32 = 80.0;

/// π, as the reference spells it.
///
/// The reference's `AUP_PE_PI` is the literal `3.1415926f`, which is one ULP
/// below [`core::f32::consts::PI`]. It is preserved here because it seeds the
/// DCT table; the difference is far below any threshold in the estimator, but
/// matching it costs nothing.
///
/// `clippy::approx_constant` is exactly right that this is a worse π. That is
/// the point: it is the reference's π, and using a better one would be the bug.
#[allow(clippy::approx_constant)]
pub const PITCH_PI: f32 = 3.1415926;

/// The band edges, as bin indices under an [`ASSUMED_FFT_FOR_BANDS`]-point FFT.
///
/// Approximately: 0, 200, 400, 600, 800, 1k, 1.2k, 1.4k, 1.6k, 2k, 2.4k,
/// 2.8k, 3.2k, 4k, 4.8k, 5.6k, 6.8k, 8k Hz.
pub const BAND_START_INDEX: [usize; NB_BANDS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];

/// Per-band compensation applied to the band gains before the LPC solve.
///
/// Roughly the reciprocal of each band's width in [`BAND_START_INDEX`] units,
/// undoing the widening of the upper bands.
pub const BAND_LPC_COMP: [f32; NB_BANDS] = [
    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2,
    0.166667, 0.173913,
];

/// The number of second-order sections in the anti-alias filter.
pub const ANTI_ALIAS_SECTIONS: usize = 5;

/// Numerator coefficients of the 4 kHz anti-alias cascade.
///
/// Applied before decimating 16 kHz to [`PROC_FS`].
pub const ANTI_ALIAS_B_4KHZ: [[f32; 3]; ANTI_ALIAS_SECTIONS] = [
    [1.0, 1.198825, 1.0],
    [1.0, -0.5674614, 1.0],
    [1.0, -1.099061, 1.0],
    [1.0, -1.265846, 1.0],
    [1.0, -1.318849, 1.0],
];

/// Denominator coefficients of the 4 kHz anti-alias cascade.
pub const ANTI_ALIAS_A_4KHZ: [[f32; 3]; ANTI_ALIAS_SECTIONS] = [
    [1.0, -1.445267, 0.5463974],
    [1.0, -1.42672, 0.6820138],
    [1.0, -1.408255, 0.8286664],
    [1.0, -1.400909, 0.924032],
    [1.0, -1.408242, 0.9789776],
];

/// Per-section gains of the 4 kHz anti-alias cascade.
pub const ANTI_ALIAS_G_4KHZ: [f32; ANTI_ALIAS_SECTIONS] =
    [0.2692541, 0.2692541, 0.2692541, 0.2692541, 0.2692541];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_band_start_index_is_monotonic() {
        for pair in BAND_START_INDEX.windows(2) {
            assert!(pair[1] > pair[0], "band edges must strictly increase");
        }
        assert_eq!(BAND_START_INDEX[0], 0);
        assert_eq!(
            BAND_START_INDEX[NB_BANDS - 1] as f32,
            ASSUMED_FFT_FOR_BANDS / 2.0,
            "the top edge is the Nyquist bin of the assumed FFT",
        );
    }

    #[test]
    fn test_pitch_pi_is_one_ulp_below_std_pi() {
        // The reference's truncated literal, kept deliberately.
        assert_ne!(PITCH_PI, core::f32::consts::PI);
        assert_eq!(
            f32::from_bits(PITCH_PI.to_bits() + 1),
            core::f32::consts::PI
        );
    }

    #[test]
    fn test_period_bounds_divide_the_resample_rate() {
        assert_eq!(MIN_PERIOD_16KHZ % PROC_RESAMPLE_RATE, 0);
        assert_eq!(MAX_PERIOD_16KHZ % PROC_RESAMPLE_RATE, 0);
    }

    #[test]
    fn test_anti_alias_sections_are_normalized() {
        // Every section is stated with a leading denominator coefficient of 1,
        // so the difference equation needs no division.
        for a in ANTI_ALIAS_A_4KHZ {
            assert_eq!(a[0], 1.0);
        }
        for b in ANTI_ALIAS_B_4KHZ {
            assert_eq!(b[0], 1.0);
        }
        // A stable all-pole section needs |a2| < 1.
        for a in ANTI_ALIAS_A_4KHZ {
            assert!(
                a[2].abs() < 1.0,
                "section pole radius must be inside the unit circle"
            );
        }
    }
}
