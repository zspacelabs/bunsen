//! # The Whisper audio front end.
//!
//! Whisper's encoder takes log-mels in a specific geometry and a specific
//! packaging. The geometry is [`MelConverterOptions`]' own default — 16 kHz,
//! 400-sample periodic Hann, hop 160, Slaney mels, `log10` over a `1e-10`
//! floor — so [`mel_options`] only fills in the two values a checkpoint
//! decides. The packaging is [`package_mels`].
//!
//! The split matters for streaming. [`RangeClamp::PerCall`] reduces over
//! whatever it is handed, so it must be applied **once, to the joined
//! spectrogram** — never per chunk. Driving the converter is therefore the
//! caller's business, and [`package_mels`] takes the result.

use burn::{
    Tensor,
    prelude::Backend,
};

use crate::ops::signal::mels::{
    AffineCompress,
    MelConverterOptions,
    RangeClamp,
};

/// The dynamic-range window Whisper floors its log-mels against, in dB.
const RANGE_CLAMP_DB: f64 = 8.0;

/// The mel geometry Whisper's encoder was trained on.
///
/// [`MelConverterOptions::default`] already is that geometry; this names the
/// pairing and fills in the two values that come from the checkpoint rather
/// than from the convention.
///
/// # Arguments
/// * `sample_rate` - the input sample rate, in Hz. Whisper's own is 16000.
/// * `n_mels` - the encoder's mel channel count, from the checkpoint.
pub fn mel_options(
    sample_rate: usize,
    n_mels: usize,
) -> MelConverterOptions {
    MelConverterOptions::default()
        .with_sample_rate(sample_rate)
        .with_n_mels(n_mels)
}

/// Packages a joined log-mel spectrogram into encoder input.
///
/// Drops the trailing frame (Whisper's `stft[..., :-1]`), floors the dynamic
/// range at 8 dB below the maximum, applies the `(log + 4) / 4` affine tail,
/// and transposes to the encoder's channels-first layout.
///
/// Apply this **once**, to the whole spectrogram. The clamp reduces over what
/// it is given, so packaging each streamed chunk separately would floor every
/// chunk against its own maximum and match nothing.
///
/// # Arguments
/// * `joined` - `[batch, frames, n_mels]`, the concatenated output of a
///   [`MelConversionContext`](crate::ops::signal::mels::MelConversionContext)
///   including its `finish` tail.
///
/// # Returns
/// `[batch, n_mels, frames - 1]`.
///
/// # Panics
/// If `frames` is less than 2. One frame leaves nothing after the trailing
/// frame is dropped, and the clamp has no maximum to reduce over.
pub fn package_mels<B: Backend>(joined: Tensor<B, 3>) -> Tensor<B, 3> {
    let frames = joined.dims()[1];
    assert!(
        frames >= 2,
        "package_mels needs at least 2 frames, got {frames}: one is dropped, \
         and the clamp reduces over what remains",
    );

    // Whisper slices `stft[..., :-1]`; the clamp reference is taken after.
    let cut = joined.slice_dim(1, 0..frames as isize - 1);

    let packaged =
        AffineCompress::default().apply(RangeClamp::PerCall { db: RANGE_CLAMP_DB }.apply(cut));

    // `[batch, frames, n_mels]` -> the `[batch, n_mels, seq]` the encoder wants.
    packaged.swap_dims(1, 2)
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        assert_tensor_close_to_vec,
    };

    type B = CpuBackend;

    /// Only the two checkpoint-derived values differ from the default.
    #[test]
    fn test_mel_options_keeps_the_default_geometry() {
        let opts = mel_options(8_000, 128);
        let base = MelConverterOptions::default();

        assert_eq!(opts.sample_rate, 8_000);
        assert_eq!(opts.n_mels, 128);

        assert_eq!(opts.n_fft, base.n_fft);
        assert_eq!(opts.hop, base.hop);
        assert_eq!(opts.window, base.window);
        assert_eq!(opts.mel_scale, base.mel_scale);
        assert_eq!(opts.filter_norm, base.filter_norm);
        assert_eq!(opts.log_base, base.log_base);
    }

    /// Drops the trailing frame and transposes to channels-first.
    #[test]
    fn test_package_mels_drops_a_frame_and_transposes() {
        let device = Default::default();
        let joined: Tensor<B, 3> = Tensor::zeros([2, 5, 3], &device);

        assert_eq!(package_mels(joined).dims(), [2, 3, 4]);
    }

    /// Too few frames panics with the reason, rather than deep inside a
    /// reduction over an empty tensor.
    #[test]
    #[should_panic(expected = "at least 2 frames")]
    fn test_package_mels_rejects_a_single_frame() {
        let device = Default::default();
        let joined: Tensor<B, 3> = Tensor::zeros([1, 1, 4], &device);

        let _ = package_mels(joined);
    }

    /// The trailing frame is dropped **before** the clamp takes its
    /// reference, so an outlier in it cannot set the floor for everything
    /// else. That ordering is the part worth pinning.
    #[test]
    fn test_package_mels_clamps_after_dropping_the_frame() {
        let device = Default::default();

        // One row, one mel, three frames. The last is an outlier that would
        // dominate the maximum if it survived to the clamp.
        let joined: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(vec![0.0_f64, -20.0, 99.0], [1, 3, 1]),
            &device,
        );

        // Kept `[0, -20]`; max 0, so the floor is -8 and -20 lifts to it.
        // Affine `(v + 4) / 4` then gives `[1.0, -1.0]`.
        assert_tensor_close_to_vec(
            &package_mels(joined),
            &[1.0, -1.0],
            Tolerance::absolute(1e-6),
        );
    }
}
