//! # The audio front end a checkpoint was trained with.
//!
//! Whisper's log-perceptive_audio are a grid fixed in time &mdash; a 25 ms
//! window every 10 ms &mdash; computed at 16 kHz and floored 8 dB under each
//! window's maximum. A checkpoint records none of that; it is the convention of
//! the pipeline that trained it. [`WhisperFrontEndConfig`] declares it on the
//! model, defaulting to upstream's, so every sample-domain number is
//! derived from it rather than written down, and a checkpoint trained
//! differently can say so. The mel options and the packaging it drives
//! live with the driver, as methods on it.

use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::driver::{
        PerWindow,
        StreamClampPolicy,
        support::drop_last_frame,
    },
    ops::signal::perceptive_audio::{
        AffineCompress,
        PerceptiveAudioConverter,
        PerceptiveAudioConverterOptions,
    },
};

/// The audio front end a checkpoint's log-perceptive_audio were computed with.
///
/// The grid is in time; [`hop`](Self::hop) and [`n_fft`](Self::n_fft) put
/// it on samples at [`sample_rate`](Self::sample_rate).
#[derive(Config, Debug, PartialEq)]
pub struct WhisperFrontEndConfig {
    /// The sample rate, in Hz.
    #[config(default = "16_000")]
    pub sample_rate: usize,

    /// The hop between mel frames, in milliseconds. One timestamp step over
    /// the encoder's stride
    /// ([`AUDIO_ENCODER_STRIDE`](super::AUDIO_ENCODER_STRIDE)), so that one
    /// encoder position is one timestamp token.
    #[config(default = "10")]
    pub hop_ms: usize,

    /// The analysis window, in milliseconds.
    #[config(default = "25")]
    pub window_ms: usize,

    /// The dynamic range kept under each window's maximum, in dB:
    /// log-perceptive_audio further below the maximum are floored to it
    /// before packaging.
    #[config(default = "8.0")]
    pub range_clamp_db: f64,
}

impl WhisperFrontEndConfig {
    /// The hop, in samples.
    pub fn hop(&self) -> usize {
        self.sample_rate * self.hop_ms / 1000
    }

    /// The analysis window, in samples: the FFT length.
    pub fn n_fft(&self) -> usize {
        self.sample_rate * self.window_ms / 1000
    }

    /// Checks that the grid falls on whole samples.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the rate, hop or window is zero, or the
    /// rate does not put the hop and the window on whole samples. At the
    /// default 10 ms and 25 ms that is any rate not a multiple of 200 Hz.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.sample_rate == 0 || self.hop_ms == 0 || self.window_ms == 0 {
            return Err(BunsenError::Invalid(format!(
                "a front end needs a rate, a hop and a window; got {} Hz, {} ms, {} ms",
                self.sample_rate, self.hop_ms, self.window_ms,
            )));
        }
        for (what, ms) in [("hop", self.hop_ms), ("window", self.window_ms)] {
            if !(self.sample_rate * ms).is_multiple_of(1000) {
                return Err(BunsenError::Invalid(format!(
                    "a {ms} ms {what} is not a whole number of samples at {} Hz",
                    self.sample_rate,
                )));
            }
        }
        Ok(())
    }

    /// Initialize a mel converter for this front end.
    ///
    /// # Arguments
    /// * `n_mels` - the encoder's mel channel count, from the checkpoint.
    /// * `device` - the device to initialize the mel converter on.
    ///
    /// # Errors
    /// See [`validate`](PerceptiveAudioConverterOptions::validate) and
    /// [`to_vec_filterbank`](PerceptiveAudioConverterOptions::try_to_filterbank_vec).
    pub fn try_init_mel_converter<B: Backend>(
        &self,
        n_mels: usize,
        device: &B::Device,
    ) -> BunsenResult<PerceptiveAudioConverter<B>> {
        self.mel_converter_options(n_mels)?.try_init(device)
    }

    /// Build the [`PerceptiveAudioConverterOptions`] for this frontend.
    ///
    /// The grid is fixed in time &mdash; a `window_ms` periodic Hann window
    /// every `hop_ms` &mdash; and the rate puts it on samples. The mel
    /// scale, its normalisation and the log compression are
    /// [`PerceptiveAudioConverterOptions`]' defaults, so at the default front
    /// end the result is that struct's default entirely.
    ///
    /// # Arguments
    /// * `n_mels` - the encoder's mel channel count, from the checkpoint.
    ///
    /// # Errors
    /// As [`validate`](Self::validate): the rate must put the grid on whole
    /// samples.
    pub fn mel_converter_options(
        &self,
        n_mels: usize,
    ) -> BunsenResult<PerceptiveAudioConverterOptions> {
        self.validate()?;
        Ok(PerceptiveAudioConverterOptions::default()
            .with_sample_rate(self.sample_rate)
            .with_n_fft(self.n_fft())
            .with_hop(self.hop())
            .with_n_mels(n_mels))
    }

    /// Packages one window of log-perceptive_audio into encoder input, against
    /// a reference.
    ///
    /// Floors every value at `range_clamp_db` below its row's `reference`,
    /// applies the `(log + 4) / 4` affine tail, and transposes to the
    /// encoder's channels-first layout. Where the reference comes from is a
    /// [`StreamClampPolicy`]'s decision; this only applies it.
    ///
    /// # Arguments
    /// * `window` - `[batch, frames, n_mels]` log-perceptive_audio.
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
        assert_eq!(reference.dims(), [batch], "one reference per batch row");

        // Each row against its own reference: the batch is independent
        // streams, and one row must not see another's peak.
        let floor = reference
            .sub_scalar(self.range_clamp_db)
            .reshape([batch, 1, 1])
            .expand(window.dims());

        // `[batch, frames, n_mels]`
        let floored = window.max_pair(floor);

        AffineCompress::default()
            .apply(floored)
            // `[batch, n_mels, frames]`
            .swap_dims(1, 2)
    }

    /// Packages a joined log-mel spectrogram into encoder input, whole.
    ///
    /// [`drop_last_frame`], then [`package_window`](Self::package_window)
    /// against each row's own maximum ([`PerWindow`]) &mdash; which, when the
    /// window is the whole clip, is upstream's
    /// `maximum(log_spec, log_spec.max() - 8.0)` exactly.
    ///
    /// Apply this **once**, to the whole spectrogram. The reference reduces
    /// over what it is given, so packaging each streamed chunk separately
    /// would floor every chunk against its own maximum and match nothing. A
    /// stream that cannot wait for the whole clip packages per window with a
    /// [`StreamClampPolicy`] instead.
    ///
    /// # Arguments
    /// * `joined` - `[batch, frames, n_mels]`, the concatenated output of a
    ///   [`MelConversionContext`](crate::ops::signal::perceptive_audio::PerceptiveAudioConversionContext)
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
        let window = drop_last_frame(joined);
        let reference = PerWindow.reference(window.clone());
        self.package_window(window, reference)
    }
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

    #[test]
    fn test_defaults_are_upstreams() {
        let front_end = WhisperFrontEndConfig::new();
        assert_eq!(front_end.sample_rate, 16_000);
        assert_eq!((front_end.hop(), front_end.n_fft()), (160, 400));
        assert_eq!(front_end.range_clamp_db, 8.0);
        assert!(front_end.validate().is_ok());
    }

    #[test]
    fn test_grid_scales_with_the_rate() {
        let at = |rate: usize| WhisperFrontEndConfig::new().with_sample_rate(rate);

        assert_eq!((at(8_000).hop(), at(8_000).n_fft()), (80, 200));
        assert!(at(8_000).validate().is_ok());
        assert_eq!((at(48_000).hop(), at(48_000).n_fft()), (480, 1_200));

        assert!(at(44_100).validate().is_err(), "not a whole hop");
        assert!(at(0).validate().is_err());
        assert!(at(16_000).with_hop_ms(0).validate().is_err());
        assert!(
            at(16_000).with_window_ms(3).validate().is_ok(),
            "3 ms is 48 whole samples"
        );
        assert!(
            at(16_000)
                .with_hop_ms(3)
                .with_window_ms(7)
                .validate()
                .is_ok(),
            "any whole-sample grid passes; the 10 ms / 25 ms pairing is convention, not checked"
        );
        assert!(
            at(22_050).with_hop_ms(10).validate().is_err(),
            "220.5 samples"
        );
    }

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
        let base = PerceptiveAudioConverterOptions::default();

        let opts = front_end().mel_converter_options(80).unwrap();
        assert_eq!(opts.sample_rate, base.sample_rate);
        assert_eq!(opts.n_mels, base.n_mels);
        assert_eq!(opts.n_fft, base.n_fft);
        assert_eq!(opts.hop, base.hop);

        let opts = front_end()
            .with_sample_rate(8_000)
            .mel_converter_options(128)
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
            .mel_converter_options(80)
            .unwrap();
        assert_eq!((opts.n_fft, opts.hop), (1_200, 480));

        let opts = front_end()
            .with_hop_ms(20)
            .with_window_ms(50)
            .mel_converter_options(80)
            .unwrap();
        assert_eq!((opts.n_fft, opts.hop), (800, 320));

        assert!(
            front_end()
                .with_sample_rate(44_100)
                .mel_converter_options(80)
                .is_err(),
            "not a whole hop"
        );
        assert!(
            front_end()
                .with_sample_rate(0)
                .mel_converter_options(80)
                .is_err()
        );
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

        let window = drop_last_frame(joined);
        let split = package_window(window.clone(), PerWindow.reference(window));

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
