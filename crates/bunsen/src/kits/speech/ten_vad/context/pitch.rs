//! # ten-vad pitch feature source.
//!
//! Feature `40` of the ten-vad feature vector is a pitch estimate, in Hz,
//! with `0.0` meaning "unvoiced" (`ALGO_TRACE.md` §3.5).
//!
//! The reference estimator is a large, deeply serial RNNoise-derived tracker:
//! band energy -> DCT -> celt LPC -> LPC pre-filter -> a five-section IIR
//! cascade -> 4 kHz decimation -> normalized moving cross-correlation ->
//! Viterbi DP over candidate periods -> weighted linear regression. None of
//! it vectorizes, and all of it carries state across frames.
//!
//! Rather than block the rest of the front end on that port, the driver takes
//! its pitch through the [`TenVadPitchSource`] seam, and ships [`ZeroPitch`]
//! as the default. Everything upstream of feature `40` — pre-emphasis, the
//! sliding STFT, the mel filterbank, the log, the normalization, and the
//! frame context stack — is complete and exercised regardless.
//!
//! An implementation of the real estimator drops in behind this trait without
//! touching the driver.

use crate::kits::speech::ten_vad::context::coeff::{
    FEATURE_EPS,
    FEATURE_MEANS,
    FEATURE_STDS,
    N_MELS,
};

/// A source for the ten-vad pitch feature.
///
/// Implemented by [`ZeroPitch`].
///
/// The interface is host-side by necessity: the reference algorithm is a
/// serial recurrence over scalars, not a tensor op. Implementations are
/// per-stream — the driver runs one instance per batch row.
pub trait TenVadPitchSource {
    /// Estimates the pitch of one hop.
    ///
    /// # Arguments
    /// * `raw_hop` - the hop's samples at the reference's int16 scale. These
    ///   are the **raw** samples: pre-emphasis is applied only to the STFT
    ///   branch, never to the pitch branch (`ALGO_TRACE.md` §3.3).
    /// * `bin_power` - the `[n_bins]` bin powers, `re^2 + im^2`, **before** the
    ///   `1 / 32768^2` normalization the mel branch applies.
    ///
    /// # Returns
    /// The pitch in Hz, or `0.0` when nothing voiced was detected.
    fn frame_pitch(
        &mut self,
        raw_hop: &[f32],
        bin_power: &[f32],
    ) -> f32;

    /// Whether [`frame_pitch`](Self::frame_pitch) inspects its arguments.
    ///
    /// The driver reads `raw_hop` and `bin_power` back from the device to
    /// call [`frame_pitch`](Self::frame_pitch). Returning `false` lets it skip
    /// that synchronization entirely and keep the sequence path on-device.
    ///
    /// An implementation returning `false` must produce the same value for
    /// every input, and must not depend on being called once per frame.
    fn inspects_input(&self) -> bool {
        true
    }

    /// Resets any carried state to the start-of-stream condition.
    fn reset(&mut self);
}

/// A [`TenVadPitchSource`] that always reports unvoiced.
///
/// Feature `40` is then pinned to the constant
/// `(0.0 - FEATURE_MEANS[40]) / (FEATURE_STDS[40] + FEATURE_EPS)`, which
/// [`ZeroPitch::normalized_feature`] reports.
///
/// This is the driver's default until the reference estimator is ported. The
/// other 40 features are unaffected: nothing upstream of the pitch branch
/// reads its output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZeroPitch;

impl ZeroPitch {
    /// The normalized value feature `40` takes under [`ZeroPitch`].
    pub fn normalized_feature() -> f32 {
        (0.0 - FEATURE_MEANS[N_MELS]) / (FEATURE_STDS[N_MELS] + FEATURE_EPS)
    }
}

impl TenVadPitchSource for ZeroPitch {
    fn frame_pitch(
        &mut self,
        _raw_hop: &[f32],
        _bin_power: &[f32],
    ) -> f32 {
        0.0
    }

    fn inspects_input(&self) -> bool {
        false
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_pitch_is_always_unvoiced() {
        let mut pitch = ZeroPitch;

        assert_eq!(pitch.frame_pitch(&[], &[]), 0.0);
        assert_eq!(pitch.frame_pitch(&[1.0, -2.0, 3.0], &[9.0; 513]), 0.0);
        assert_eq!(pitch.frame_pitch(&[f32::MAX], &[f32::MIN]), 0.0);
    }

    #[test]
    fn test_zero_pitch_skips_readback() {
        // The driver keys its device-to-host sync off this.
        assert!(!ZeroPitch.inspects_input());
    }

    #[test]
    fn test_zero_pitch_reset_is_stateless() {
        let mut pitch = ZeroPitch;
        let before = pitch.frame_pitch(&[1.0], &[1.0]);
        pitch.reset();
        let after = pitch.frame_pitch(&[1.0], &[1.0]);
        assert_eq!(before, after);
        assert_eq!(pitch, ZeroPitch);
    }

    #[test]
    fn test_normalized_feature_matches_the_normalization_formula() {
        let expected = (0.0 - FEATURE_MEANS[N_MELS]) / (FEATURE_STDS[N_MELS] + FEATURE_EPS);
        assert_eq!(ZeroPitch::normalized_feature(), expected);

        // A pitch mean of ~92.36 Hz over a std of ~115.21 puts silence a bit
        // over half a standard deviation below the mean.
        assert!(ZeroPitch::normalized_feature() < 0.0);
        assert!((ZeroPitch::normalized_feature() - (-0.80161)).abs() < 1e-4);
    }

    #[test]
    fn test_usable_as_a_trait_object() {
        let mut pitch: Box<dyn TenVadPitchSource> = Box::new(ZeroPitch);
        assert_eq!(pitch.frame_pitch(&[0.5], &[0.5]), 0.0);
        assert!(!pitch.inspects_input());
        pitch.reset();
    }

    /// A minimal non-trivial implementation, to prove the seam is usable and
    /// that `inspects_input` defaults to `true`.
    #[derive(Default)]
    struct CountingPitch {
        calls: usize,
    }

    impl TenVadPitchSource for CountingPitch {
        fn frame_pitch(
            &mut self,
            raw_hop: &[f32],
            _bin_power: &[f32],
        ) -> f32 {
            self.calls += 1;
            raw_hop.len() as f32
        }

        fn reset(&mut self) {
            self.calls = 0;
        }
    }

    #[test]
    fn test_custom_source_defaults_to_inspecting_input() {
        let mut pitch = CountingPitch::default();
        assert!(pitch.inspects_input());

        assert_eq!(pitch.frame_pitch(&[0.0; 4], &[]), 4.0);
        assert_eq!(pitch.calls, 1);

        pitch.reset();
        assert_eq!(pitch.calls, 0);
    }
}
