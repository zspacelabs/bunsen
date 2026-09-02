//! # The Whisper audio front end.
//!
//! Whisper's encoder takes log-mels in a specific geometry and a specific
//! packaging, both declared by the model's [`WhisperFrontEndConfig`]. The
//! geometry is a grid fixed in time &mdash; a `window_ms` periodic Hann
//! window every `hop_ms`, Slaney mels, `log10` over a `1e-10` floor &mdash;
//! that [`mel_options`](WhisperFrontEndConfig::mel_options) puts on samples
//! at the model's rate; at the defaults it is [`MelConverterOptions`]' own
//! default. The packaging is three things, and a stream needs them apart:
//!
//! * [`trim_stream_tail`] drops the trailing frame, Whisper's `stft[..., :-1]`.
//!   It applies once, at the end of a stream, after the converter's `finish`
//!   has flushed the end padding.
//! * [`package_window`](WhisperFrontEndConfig::package_window) floors a window
//!   `range_clamp_db` below a reference maximum, applies the `(log + 4) / 4`
//!   affine tail, and transposes to channels-first. It applies per window, and
//!   takes the reference rather than computing one.
//! * The reference is a [`ClampPolicy`]'s business, because upstream takes it
//!   over the whole clip and a stream cannot.
//!
//! [`package_mels`](WhisperFrontEndConfig::package_mels) is the whole-clip
//! composition of the three: trim, then package against each window's own
//! maximum. It is what every existing test pins, and it is unchanged by the
//! split.

use burn::{
    Tensor,
    prelude::Backend,
};

use crate::{
    errors::BunsenResult,
    kits::speech::whisper::{
        blocks::WhisperFrontEndConfig,
        driver::clamp::{
            ClampPolicy,
            PerWindow,
        },
    },
    ops::signal::mels::{
        AffineCompress,
        MelConverterOptions,
    },
};

impl WhisperFrontEndConfig {
    /// The mel geometry Whisper's encoder was trained on, at this front
    /// end's rate.
    ///
    /// The grid is fixed in time &mdash; a `window_ms` periodic Hann window
    /// every `hop_ms` &mdash; and the rate puts it on samples. The mel
    /// scale, its normalisation and the log compression are
    /// [`MelConverterOptions`]' defaults, so at the default front end the
    /// result is that struct's default entirely.
    ///
    /// # Arguments
    /// * `n_mels` - the encoder's mel channel count, from the checkpoint.
    ///
    /// # Errors
    /// As [`validate`](Self::validate): the rate must put the grid on whole
    /// samples.
    pub fn mel_options(
        &self,
        n_mels: usize,
    ) -> BunsenResult<MelConverterOptions> {
        self.validate()?;
        Ok(MelConverterOptions::default()
            .with_sample_rate(self.sample_rate)
            .with_n_fft(self.n_fft())
            .with_hop(self.hop())
            .with_n_mels(n_mels))
    }

    /// Packages one window of log-mels into encoder input, against a
    /// reference.
    ///
    /// Floors every value at `range_clamp_db` below its row's `reference`,
    /// applies the `(log + 4) / 4` affine tail, and transposes to the
    /// encoder's channels-first layout. Where the reference comes from is a
    /// [`ClampPolicy`]'s decision; this only applies it.
    ///
    /// # Arguments
    /// * `window` - `[batch, frames, n_mels]` log-mels.
    /// * `reference` - `[batch]`, the maximum each row is floored against, in
    ///   the post-log domain.
    ///
    /// # Returns
    /// `[batch, n_mels, frames]`.
    pub fn package_window<B: Backend>(
        &self,
        window: Tensor<B, 3>,
        reference: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let [batch, _, _] = window.dims();
        assert_eq!(reference.dims(), [batch], "one reference per batch row",);

        // Each row against its own reference: the batch is independent
        // streams, and one row must not see another's peak.
        let floor = reference
            .sub_scalar(self.range_clamp_db)
            .reshape([batch, 1, 1])
            .expand(window.dims());
        let floored = window.max_pair(floor);

        // `[batch, frames, n_mels]` -> the `[batch, n_mels, seq]` the encoder
        // wants.
        AffineCompress::default().apply(floored).swap_dims(1, 2)
    }

    /// Packages a joined log-mel spectrogram into encoder input, whole.
    ///
    /// [`trim_stream_tail`], then [`package_window`](Self::package_window)
    /// against each row's own maximum ([`PerWindow`]) &mdash; which, when the
    /// window is the whole clip, is upstream's
    /// `maximum(log_spec, log_spec.max() - 8.0)` exactly.
    ///
    /// Apply this **once**, to the whole spectrogram. The reference reduces
    /// over what it is given, so packaging each streamed chunk separately
    /// would floor every chunk against its own maximum and match nothing. A
    /// stream that cannot wait for the whole clip packages per window with a
    /// [`ClampPolicy`] instead.
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
    /// If `frames` is less than 2. One frame leaves nothing after the
    /// trailing frame is dropped, and the clamp has no maximum to reduce
    /// over.
    pub fn package_mels<B: Backend>(
        &self,
        joined: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let window = trim_stream_tail(joined);
        let reference = PerWindow.reference(&window);
        self.package_window(window, reference)
    }
}

/// Drops the trailing frame of a whole stream's log-mels.
///
/// Whisper's `log_mel_spectrogram` computes `stft[..., :-1]`: the converter
/// emits one frame more than upstream keeps, and this removes it, once,
/// after its `finish` tail. The dropped frame is the one the end padding
/// produced; upstream discards it so that 30 s is exactly 3000 frames.
///
/// # Arguments
/// * `joined` - `[batch, frames, n_mels]` log-mels of a whole stream.
///
/// # Returns
/// `[batch, frames - 1, n_mels]`.
///
/// # Panics
/// If `frames` is less than 2: one frame leaves nothing to package.
pub fn trim_stream_tail<B: Backend>(joined: Tensor<B, 3>) -> Tensor<B, 3> {
    let frames = joined.dims()[1];
    assert!(
        frames >= 2,
        "package_mels needs at least 2 frames, got {frames}: one is dropped, \
         and the clamp reduces over what remains",
    );

    joined.slice_dim(1, 0..frames as isize - 1)
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::Tolerance,
    };

    use super::*;
    use crate::{
        kits::speech::whisper::blocks::{
            AUDIO_ENCODER_STRIDE,
            WhisperTokenLayoutConfig,
        },
        support::testing::{
            CpuBackend,
            assert_tensor_close_to_vec,
        },
    };

    type B = CpuBackend;

    /// The default front end: what every packaging test pins.
    fn front_end() -> WhisperFrontEndConfig {
        WhisperFrontEndConfig::new()
    }

    fn package_mels(joined: Tensor<B, 3>) -> Tensor<B, 3> {
        front_end().package_mels(joined)
    }

    fn package_window(
        window: Tensor<B, 3>,
        reference: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        front_end().package_window(window, reference)
    }

    /// At the default front end the derived geometry is
    /// `MelConverterOptions`' own default; at another rate the 10 ms / 25 ms
    /// grid scales and nothing else moves.
    #[test]
    fn test_mel_options_derives_the_grid_from_the_rate() {
        let base = MelConverterOptions::default();

        let opts = front_end().mel_options(80).unwrap();
        assert_eq!(opts.sample_rate, base.sample_rate);
        assert_eq!(opts.n_mels, base.n_mels);
        assert_eq!(opts.n_fft, base.n_fft);
        assert_eq!(opts.hop, base.hop);

        let opts = front_end()
            .with_sample_rate(8_000)
            .mel_options(128)
            .unwrap();
        assert_eq!(opts.sample_rate, 8_000);
        assert_eq!(opts.n_mels, 128);
        assert_eq!(opts.n_fft, 200);
        assert_eq!(opts.hop, 80);
        assert_eq!(opts.window, base.window);
        assert_eq!(opts.mel_scale, base.mel_scale);
        assert_eq!(opts.filter_norm, base.filter_norm);
        assert_eq!(opts.log_base, base.log_base);

        let opts = front_end()
            .with_sample_rate(48_000)
            .mel_options(80)
            .unwrap();
        assert_eq!((opts.n_fft, opts.hop), (1_200, 480));

        let opts = front_end()
            .with_hop_ms(20)
            .with_window_ms(50)
            .mel_options(80)
            .unwrap();
        assert_eq!((opts.n_fft, opts.hop), (800, 320));

        assert!(
            front_end()
                .with_sample_rate(44_100)
                .mel_options(80)
                .is_err(),
            "not a whole hop"
        );
        assert!(front_end().with_sample_rate(0).mel_options(80).is_err());
    }

    /// The default hop is one timestamp step per encoder position.
    #[test]
    fn test_hop_is_one_timestamp_step_per_encoder_position() {
        let hop_seconds = front_end().hop_ms as f64 / 1000.0;
        let step_seconds =
            WhisperTokenLayoutConfig::new().timestamp_step_seconds / AUDIO_ENCODER_STRIDE as f64;
        assert!((hop_seconds - step_seconds).abs() < 1e-12);
    }

    /// The clamp range is the front end's, not a constant.
    #[test]
    fn test_package_window_uses_the_configured_range() {
        let device = Default::default();
        let window: Tensor<B, 3> =
            Tensor::from_data(TensorData::new(vec![0.0_f64, -20.0], [1, 2, 1]), &device);
        let reference: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(vec![0.0_f64], [1]), &device);

        // A 4 dB range floors -20 at -4: affine `(v + 4) / 4` gives 0.
        assert_tensor_close_to_vec(
            &front_end()
                .with_range_clamp_db(4.0)
                .package_window(window, reference),
            &[1.0, 0.0],
            Tolerance::absolute(1e-6),
        );
    }

    /// Sweeps the shape parameters: for every accepted frame count, and
    /// across batch and mel widths, the output is `[batch, n_mels, frames -
    /// 1]`.
    #[test]
    fn test_package_mels_shape_over_a_range() {
        let device = Default::default();

        for batch in 1..4 {
            for n_mels in 1..5 {
                for frames in 2..12 {
                    let joined: Tensor<B, 3> = Tensor::zeros([batch, frames, n_mels], &device);

                    assert_eq!(
                        package_mels(joined).dims(),
                        [batch, n_mels, frames - 1],
                        "batch {batch}, n_mels {n_mels}, frames {frames}",
                    );
                }
            }
        }
    }

    /// Sweeps the frame count against the invariant that matters: the dropped
    /// frame never reaches the clamp, so an outlier parked in it cannot move
    /// the reference for anything that survives.
    ///
    /// Every kept value is 0, so the clamp is inert and the affine takes them
    /// all to `(0 + 4) / 4`. If the trailing frame leaked through, its 99
    /// would become the maximum and floor everything else at 91.
    #[test]
    fn test_package_mels_ignores_the_dropped_frame_over_a_range() {
        let device = Default::default();
        let n_mels = 2;

        for frames in 2..10 {
            let mut data = vec![0.0_f64; frames * n_mels];
            for slot in data.iter_mut().skip((frames - 1) * n_mels) {
                *slot = 99.0;
            }

            let joined: Tensor<B, 3> =
                Tensor::from_data(TensorData::new(data, [1, frames, n_mels]), &device);

            assert_tensor_close_to_vec(
                &package_mels(joined),
                &vec![1.0; n_mels * (frames - 1)],
                Tolerance::absolute(1e-6),
            );
        }
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

    /// The other degenerate count: zero frames would slice `0..-1`.
    #[test]
    #[should_panic(expected = "at least 2 frames")]
    fn test_package_mels_rejects_zero_frames() {
        let device = Default::default();
        let joined: Tensor<B, 3> = Tensor::zeros([1, 0, 4], &device);

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

    /// **I4.** The split is behaviour-preserving: trim, then package against
    /// each row's own maximum, is `package_mels` to the bit &mdash; on rows
    /// with different peaks, so that the per-row reference is exercised.
    #[test]
    fn test_split_packaging_equals_package_mels() {
        let device = Default::default();
        let (batch, frames, n_mels) = (3, 7, 4);

        let data: Vec<f64> = (0..batch * frames * n_mels)
            .map(|k| ((k * 37) % 23) as f64 - 15.0 + (k / (frames * n_mels)) as f64 * 3.0)
            .collect();
        let joined: Tensor<B, 3> =
            Tensor::from_data(TensorData::new(data, [batch, frames, n_mels]), &device);

        let whole = package_mels(joined.clone());

        let window = trim_stream_tail(joined);
        let split = package_window(window.clone(), PerWindow.reference(&window));

        assert_eq!(split.dims(), [batch, n_mels, frames - 1]);
        split.to_data().assert_eq(&whole.to_data(), true);
    }

    /// The reference is per row: raising one row's reference floors that row
    /// harder and leaves the others alone.
    #[test]
    fn test_package_window_floors_each_row_against_its_reference() {
        let device = Default::default();

        // Two rows, two frames, one mel: `[0, -20]` in each row.
        let window: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(vec![0.0_f64, -20.0, 0.0, -20.0], [2, 2, 1]),
            &device,
        );

        // Row 0 against its own maximum (0): floor -8. Row 1 against a
        // reference of 12, as if something louder had been heard: floor 4,
        // which lifts both of its values.
        let reference: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(vec![0.0_f64, 12.0], [2]), &device);

        assert_tensor_close_to_vec(
            &package_window(window, reference),
            &[1.0, -1.0, 2.0, 2.0],
            Tolerance::absolute(1e-6),
        );
    }

    /// A reference below the window's own maximum does not clip the peak:
    /// values above the floor pass through unchanged, whatever the reference.
    #[test]
    fn test_package_window_never_clips_above_the_floor() {
        let device = Default::default();
        let window: Tensor<B, 3> =
            Tensor::from_data(TensorData::new(vec![4.0_f64, 0.0], [1, 2, 1]), &device);
        let reference: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(vec![0.0_f64], [1]), &device);

        // Floor at -8; nothing is below it, so it is the affine alone.
        assert_tensor_close_to_vec(
            &package_window(window, reference),
            &[2.0, 1.0],
            Tolerance::absolute(1e-6),
        );
    }
}
