//! # ten-vad front-end coefficients.
//!
//! The fixed scalar constants and normalization tables the ten-vad
//! pre-processing driver is built from.
//!
//! [`FEATURE_MEANS`] and [`FEATURE_STDS`] are transcribed verbatim from the
//! reference implementation's `src/coeff.h`; the remaining constants come
//! from the reference feature path (see `ALGO_TRACE.md` §3.1, §3.3, §3.6).
//!
//! These are reference data, not tunables: changing any of them decouples the
//! driver from the pretrained weights.

/// The number of mel filterbank bands.
///
/// Features `0..N_MELS` are the log-mel energies; feature `N_MELS` is the
/// pitch. See [`crate::kits::speech::ten_vad::TenVadMeta::n_freq`].
pub const N_MELS: usize = 40;

/// The ten-vad feature width: [`N_MELS`] log-mel bands plus one pitch bin.
pub const N_FREQ: usize = N_MELS + 1;

/// The frame context depth; the model consumes `[f_{t-2}, f_{t-1}, f_t]`.
///
/// See `ALGO_TRACE.md` §3.7.
pub const D_CTX: usize = 3;

/// The hop size, in samples, of one ten-vad frame (16 ms at 16 kHz).
///
/// The reference C API accepts other hop sizes but drains its FIFO in
/// 256-sample steps and reports only the last internal frame, so 256 is the
/// only size that yields one score per model call (`ALGO_TRACE.md` §7).
pub const HOP_SIZE: usize = 256;

/// The sample rate, in Hz, the ten-vad front end is defined for.
pub const SAMPLE_RATE: usize = 16000;

/// The reference driver's periodic LSTM reset period, in model calls.
///
/// 1875 hops is 30 s at 16 kHz. The C driver zeroes both LSTM states this
/// often, leaving the feature stack intact (`ALGO_TRACE.md` §5,
/// `src/aed.cc:476-481`). The reference marks the value `// TODO`
/// (`src/aed.cc:640`); it is reproduced here for parity, not because it is
/// obviously right.
///
/// See [`TenVadContextConfig::reset_frames`] to change or disable it.
///
/// [`TenVadContextConfig::reset_frames`]:
///     crate::kits::speech::ten_vad::TenVadContextConfig
pub const RESET_FRAMES: usize = 1875;

/// The epsilon used both as the log floor and as the normalization guard.
///
/// The reference applies it twice: `log(melPower + EPS)` and
/// `(v - MEAN) / (STD + EPS)` (`ALGO_TRACE.md` §3.6).
pub const FEATURE_EPS: f32 = 1e-20;

/// The pre-emphasis coefficient: `y[n] = x[n] - PRE_EMPHASIS_COEFF * x[n-1]`.
///
/// See `ALGO_TRACE.md` §3.3.
pub const PRE_EMPHASIS_COEFF: f32 = 0.97;

/// The bin-power normalizer, `32768^2`.
///
/// The reference pipeline runs at int16 scale, and divides the bin powers by
/// this before the mel filterbank (`ALGO_TRACE.md` §3.6).
pub const POWER_NORMAL: f32 = 32768.0 * 32768.0;

/// The scale from unit-range audio to the reference's int16 scale.
///
/// The reference casts `i16` samples to `f32` without rescaling. bunsen's
/// driver takes `[-1, 1]` audio (matching the rest of the crate) and
/// multiplies by this on entry, so both paths see the same values.
pub const INPUT_SCALE: f32 = 32768.0;

/// Per-feature means, from the reference `src/coeff.h`.
///
/// Index `40` is the pitch mean, in Hz.
///
/// Transcribed at the reference's full written precision rather than rounded
/// to what `f32` can hold, so the table stays diffable against `coeff.h`.
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
pub const FEATURE_MEANS: [f32; N_FREQ] = [
    -8.198236465454e+00, -6.265716552734e+00, -5.483818531036e+00,
    -4.758691310883e+00, -4.417088985443e+00, -4.142892837524e+00,
    -3.912850379944e+00, -3.845927953720e+00, -3.657090425491e+00,
    -3.723418712616e+00, -3.876134157181e+00, -3.843890905380e+00,
    -3.690405130386e+00, -3.756065845490e+00, -3.698696136475e+00,
    -3.650463104248e+00, -3.700468778610e+00, -3.567321300507e+00,
    -3.498900175095e+00, -3.477807044983e+00, -3.458816051483e+00,
    -3.444923877716e+00, -3.401328563690e+00, -3.306261301041e+00,
    -3.278556823730e+00, -3.233250856400e+00, -3.198616027832e+00,
    -3.204526424408e+00, -3.208798646927e+00, -3.257838010788e+00,
    -3.381376743317e+00, -3.534021377563e+00, -3.640867948532e+00,
    -3.726858854294e+00, -3.773730993271e+00, -3.804667234421e+00,
    -3.832901000977e+00, -3.871120452881e+00, -3.990592956543e+00,
    -4.480289459229e+00, 9.235690307617e+01,];

/// Per-feature standard deviations, from the reference `src/coeff.h`.
///
/// Index `40` is the pitch standard deviation, in Hz.
///
/// Transcribed at the reference's full written precision rather than rounded
/// to what `f32` can hold, so the table stays diffable against `coeff.h`.
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
pub const FEATURE_STDS: [f32; N_FREQ] = [
    5.166063785553e+00, 4.977209568024e+00, 4.698895931244e+00,
    4.630621433258e+00, 4.634347915649e+00, 4.641156196594e+00,
    4.640676498413e+00, 4.666367053986e+00, 4.650534629822e+00,
    4.640020847321e+00, 4.637400150299e+00, 4.620099067688e+00,
    4.596316337585e+00, 4.562654972076e+00, 4.554360389709e+00,
    4.566910743713e+00, 4.562489986420e+00, 4.562412738800e+00,
    4.585299491882e+00, 4.600179672241e+00, 4.592845916748e+00,
    4.585922718048e+00, 4.583496570587e+00, 4.626092910767e+00,
    4.626957893372e+00, 4.626289367676e+00, 4.637005805969e+00,
    4.683015823364e+00, 4.726813793182e+00, 4.734289646149e+00,
    4.753227233887e+00, 4.849722862244e+00, 4.869434833527e+00,
    4.884482860565e+00, 4.921327114105e+00, 4.959212303162e+00,
    4.996619224548e+00, 5.044823646545e+00, 5.072216987610e+00,
    5.096439361572e+00, 1.152136917114e+02,];

#[cfg(test)]
// The anchors below are quoted from the reference `coeff.h` at its full
// written precision, so they stay greppable against the source table.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_constants() {
        assert_eq!(N_MELS, 40);
        assert_eq!(N_FREQ, 41);
        assert_eq!(D_CTX, 3);
        assert_eq!(HOP_SIZE, 256);
        assert_eq!(SAMPLE_RATE, 16000);

        // The tables are indexed by feature, so they must be N_FREQ wide.
        assert_eq!(FEATURE_MEANS.len(), N_FREQ);
        assert_eq!(FEATURE_STDS.len(), N_FREQ);
    }

    #[test]
    fn test_scalar_constants() {
        assert_eq!(FEATURE_EPS, 1e-20);
        assert_eq!(PRE_EMPHASIS_COEFF, 0.97);
        assert_eq!(INPUT_SCALE, 32768.0);
        assert_eq!(POWER_NORMAL, 32768.0 * 32768.0);
        assert_eq!(POWER_NORMAL, 1073741824.0);
    }

    #[test]
    fn test_table_anchors() {
        // Anchors against the reference `src/coeff.h` table, at the two ends
        // and at the pitch entry.
        assert_eq!(FEATURE_MEANS[0], -8.198236465454e+00);
        assert_eq!(FEATURE_MEANS[N_MELS - 1], -4.480289459229e+00);
        assert_eq!(FEATURE_MEANS[N_MELS], 9.235690307617e+01);

        assert_eq!(FEATURE_STDS[0], 5.166063785553e+00);
        assert_eq!(FEATURE_STDS[N_MELS - 1], 5.096439361572e+00);
        assert_eq!(FEATURE_STDS[N_MELS], 1.152136917114e+02);
    }

    #[test]
    fn test_stds_are_usable_divisors() {
        // Every std is used as `1 / (std + EPS)`; a non-positive entry would
        // make the normalization explode or flip sign.
        for (i, &std) in FEATURE_STDS.iter().enumerate() {
            assert!(std > 0.0, "FEATURE_STDS[{i}] = {std} is not positive");
            assert!(std.is_finite(), "FEATURE_STDS[{i}] = {std} is not finite");
        }
        for (i, &mean) in FEATURE_MEANS.iter().enumerate() {
            assert!(
                mean.is_finite(),
                "FEATURE_MEANS[{i}] = {mean} is not finite"
            );
        }
    }

    #[test]
    fn test_log_mel_means_are_negative() {
        // The mel bands are `log(power / 32768^2 + eps)` of speech-scale
        // audio, so their means sit well below zero; the pitch entry (in Hz)
        // is the only positive one.
        for (i, &mean) in FEATURE_MEANS[..N_MELS].iter().enumerate() {
            assert!(mean < 0.0, "FEATURE_MEANS[{i}] = {mean} should be negative");
        }
        // The pitch mean is in Hz, so a compile-time check is available.
        const { assert!(FEATURE_MEANS[N_MELS] > 0.0) };
    }
}
