//! # The log-mel front end: a mel grid fixed in time, floored under a level.

use burn::{
    Tensor,
    config::Config,
    tensor::{
        Device,
        kind::Basic,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
    kits::speech::support::{
        LevelStatistic,
        MaxLevel,
    },
    ops::signal::perceptive_audio::{
        AffineCompress,
        PerceptiveAudioConverter,
        PerceptiveAudioConverterConfig,
    },
};

/// The log-mel front end a checkpoint's features were computed with.
///
/// Whisper-family log-mels are a grid fixed in time (a 25 ms window every
/// 10 ms), computed at 16 kHz and floored 8 dB under a reference maximum.
/// A checkpoint records none of that; it is the convention of the pipeline
/// that trained it. A model declares it
/// ([`WhisperApiConfig::front_end`](crate::kits::speech::asr::whisper::blocks::WhisperApiConfig::front_end),
/// and the Qwen3-ASR and Voxtral front ends' `log_mel`), defaulting to
/// upstream Whisper's, so every sample-domain number is derived from
/// it rather than written down, and a checkpoint trained differently can
/// say so. It also builds what it drives: the mel converter
/// ([`try_init_audio_converter`](Self::try_init_audio_converter)) and the
/// packaging of a window into encoder input
/// ([`package_window`](Self::package_window)).
///
/// The grid is in time; [`hop`](Self::hop) and [`n_fft`](Self::n_fft) put
/// it on samples at [`sample_rate`](Self::sample_rate).
#[derive(Config, Debug, PartialEq)]
pub struct LogMelFrontEndConfig {
    /// The sample rate, in Hz.
    #[config(default = 16_000)]
    pub sample_rate: usize,

    /// The hop between mel frames, in milliseconds. For Whisper, one
    /// timestamp step over the encoder's stride
    /// ([`AUDIO_ENCODER_STRIDE`](crate::kits::speech::asr::whisper::blocks::AUDIO_ENCODER_STRIDE)),
    /// so that one encoder position is one timestamp token.
    #[config(default = 10)]
    pub hop_ms: usize,

    /// The analysis window, in milliseconds.
    #[config(default = 25)]
    pub window_ms: usize,

    /// The dynamic range kept under each window's maximum, in dB:
    /// log-mels further below the maximum are floored to it
    /// before packaging.
    #[config(default = 8.0)]
    pub range_clamp_db: f64,
}

impl LogMelFrontEndConfig {
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
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the rate, hop
    /// or window is zero (with a [`ConstraintError`] cause), or the rate does
    /// not put the hop and the window on whole samples. At the default 10 ms
    /// and 25 ms that is any rate not a multiple of 200 Hz.
    pub fn try_validate(&self) -> BunsenResult<()> {
        const OWNER: &str = "LogMelFrontEndConfig";
        for (field, value) in [
            ("sample_rate", self.sample_rate),
            ("hop_ms", self.hop_ms),
            ("window_ms", self.window_ms),
        ] {
            if value == 0 {
                return Err(ConstraintError::zero_or_empty(OWNER, field).into());
            }
        }
        for (field, what, ms) in [
            ("hop_ms", "hop", self.hop_ms),
            ("window_ms", "window", self.window_ms),
        ] {
            if !(self.sample_rate * ms).is_multiple_of(1000) {
                return Err(crate::errors::ConstraintError::custom(
                    OWNER,
                    field,
                    format!(
                        "a {ms} ms {what} is not a whole number of samples at {} Hz",
                        self.sample_rate,
                    ),
                )
                .into());
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
    /// See [`try_validate`](PerceptiveAudioConverterConfig::try_validate) and
    /// [`try_to_filterbank_vec`](PerceptiveAudioConverterConfig::try_to_filterbank_vec).
    pub fn try_init_audio_converter(
        &self,
        n_mels: usize,
        device: &Device,
    ) -> BunsenResult<PerceptiveAudioConverter> {
        self.mel_converter_options(n_mels)?.try_init(device)
    }

    /// Build the [`PerceptiveAudioConverterConfig`] for this frontend.
    ///
    /// The grid is fixed in time &mdash; a `window_ms` periodic Hann window
    /// every `hop_ms` &mdash; and the rate puts it on samples. The mel
    /// scale, its normalisation and the log compression are
    /// [`PerceptiveAudioConverterConfig`]' defaults, so at the default front
    /// end the result is that struct's default entirely.
    ///
    /// # Arguments
    /// * `n_mels` - the encoder's mel channel count, from the checkpoint.
    ///
    /// # Errors
    /// As [`try_validate`](Self::try_validate): the rate must put the grid on whole
    /// samples.
    pub fn mel_converter_options(
        &self,
        n_mels: usize,
    ) -> BunsenResult<PerceptiveAudioConverterConfig> {
        self.try_validate()?;
        Ok(PerceptiveAudioConverterConfig::default()
            .with_sample_rate(self.sample_rate)
            .with_n_fft(self.n_fft())
            .with_hop(self.hop())
            .with_n_mels(n_mels))
    }

    /// Packages one window of log-mels into encoder input, against
    /// a reference.
    ///
    /// Floors every value at `range_clamp_db` below its row's `reference`,
    /// applies the `(log + 4) / 4` affine tail, and transposes to the
    /// encoder's channels-first layout. Where the reference comes from is a
    /// stream's [`LevelPolicy`](crate::kits::speech::asr::driver::LevelPolicy)
    /// decision; this only applies it.
    ///
    /// # Arguments
    /// * `window` - `[batch, frames, n_mels]` log-mels.
    /// * `reference` - `[batch]`, the maximum each row is floored against, in
    ///   the post-log domain.
    ///
    /// # Returns
    /// `[batch, n_mels, frames]`.
    pub fn package_window(
        &self,
        window: Tensor<3>,
        reference: Tensor<1>,
    ) -> Tensor<3> {
        let [batch, frames, n_mels] = crate::contracts::unpack_shape_contract!(
            ["batch", "frames", "n_mels"],
            &window.dims(),
            &["batch", "frames", "n_mels"]
        );
        // One reference per batch row.
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch"],
            &reference.dims(),
            &[("batch", batch)]
        );

        // Each row against its own reference: the batch is independent
        // streams, and one row must not see another's peak.
        let floor = reference
            .sub_scalar(self.range_clamp_db)
            .reshape([batch, 1, 1])
            .expand(window.dims());

        // `[batch, frames, n_mels]`
        let floored = window.max_pair(floor);

        // `[batch, n_mels, frames]`
        let packaged = AffineCompress::default().apply(floored).swap_dims(1, 2);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "n_mels", "frames"],
            &packaged.dims(),
            &[("batch", batch), ("n_mels", n_mels), ("frames", frames)]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (frames, n_mels);
        packaged
    }

    /// Packages a joined log-mel spectrogram into encoder input, whole.
    ///
    /// [`drop_last_frame`], then [`package_window`](Self::package_window)
    /// against each row's own maximum ([`MaxLevel`]) &mdash; which, when the
    /// window is the whole clip, is upstream's
    /// `maximum(log_spec, log_spec.max() - 8.0)` exactly.
    ///
    /// Apply this **once**, to the whole spectrogram. The reference reduces
    /// over what it is given, so packaging each streamed chunk separately
    /// would floor every chunk against its own maximum and match nothing. A
    /// stream that cannot wait for the whole clip packages per window under a
    /// [`LevelPolicy`](crate::kits::speech::asr::driver::LevelPolicy) instead.
    ///
    /// # Arguments
    /// * `joined` - `[batch, frames, n_mels]`, the concatenated output of a
    ///   [`PerceptiveAudioConversionContext`](crate::ops::signal::perceptive_audio::PerceptiveAudioConversionContext)
    ///   including its `finish` tail.
    ///
    /// # Returns
    /// `[batch, n_mels, frames - 1]`.
    ///
    /// # Panics
    /// If `frames` is less than 2. One frame leaves nothing after the
    /// trailing frame is dropped, and the clamp has no maximum to reduce
    /// over.
    pub fn package_mels(
        &self,
        joined: Tensor<3>,
    ) -> Tensor<3> {
        let window = drop_last_frame(joined);
        let reference = MaxLevel::of(window.clone()).max;
        self.package_window(window, reference)
    }
}

/// Drops the trailing frame of a whole stream's log-mels.
///
/// # Arguments
/// * `stream` - `[batch, frames, n_mels]`.
///
/// # Returns
/// `[batch, frames - 1, n_mels]`.
///
/// # Panics
/// If `frames` is less than 2: one frame leaves nothing to package.
pub fn drop_last_frame<K>(stream: Tensor<3, K>) -> Tensor<3, K>
where
    K: Basic,
{
    let frames = stream.dims()[1];
    assert!(
        frames >= 2,
        "package_mels needs at least 2 frames, got {frames}: one is dropped, \
         and the clamp reduces over what remains",
    );
    stream.slice_dim(1, 0..frames as isize - 1)
}

/// The log-mels of whole waveforms: `wave` through a fresh context of
/// `converter`, the tail its finish flushes joined on.
///
/// # Arguments
/// * `wave` - `[batch, samples]`, in the dtype the caller computes in.
///
/// # Returns
/// `[batch, frames, n_mels]`, before any packaging.
///
/// # Errors
/// As [`PerceptiveAudioConversionContext::transform`](crate::ops::signal::perceptive_audio::PerceptiveAudioConversionContext::transform).
pub fn log_mels_of(
    converter: &PerceptiveAudioConverter,
    wave: Tensor<2>,
) -> BunsenResult<Tensor<3>> {
    let [batch] =
        crate::contracts::unpack_shape_contract!(["batch", "samples"], &wave.dims(), &["batch"]);
    let (mels, context) = converter.new_context(batch).transform(wave)?;
    let joined = match context.finish() {
        Some(tail) => Tensor::cat(vec![mels, tail], 1),
        None => mels,
    };
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract_periodically!(
        ["batch", "frames", "n_mels"],
        &joined.dims(),
        &[("batch", batch)]
    );
    Ok(joined)
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::{
            TensorData,
            s,
        },
        tensor::{
            Int,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::{
            assert_tensor_close_to_vec,
            cpu_device,
        },
    };

    #[test]
    fn test_defaults_are_upstreams() {
        let front_end = LogMelFrontEndConfig::new();
        assert_eq!(front_end.sample_rate, 16_000);
        assert_eq!((front_end.hop(), front_end.n_fft()), (160, 400));
        assert_eq!(front_end.range_clamp_db, 8.0);
        assert!(front_end.try_validate().is_ok());
    }

    #[test]
    fn test_grid_scales_with_the_rate() {
        let at = |rate: usize| LogMelFrontEndConfig::new().with_sample_rate(rate);

        assert_eq!((at(8_000).hop(), at(8_000).n_fft()), (80, 200));
        assert!(at(8_000).try_validate().is_ok());
        assert_eq!((at(48_000).hop(), at(48_000).n_fft()), (480, 1_200));

        assert!(at(44_100).try_validate().is_err(), "not a whole hop");
        ErrorMatcher::illegal().assert_err(&at(0).try_validate());
        ErrorMatcher::illegal().assert_err(&at(16_000).with_hop_ms(0).try_validate());
        assert!(
            at(16_000).with_window_ms(3).try_validate().is_ok(),
            "3 ms is 48 whole samples"
        );
        assert!(
            at(16_000)
                .with_hop_ms(3)
                .with_window_ms(7)
                .try_validate()
                .is_ok(),
            "any whole-sample grid passes; the 10 ms / 25 ms pairing is convention, not checked"
        );
        assert!(
            at(22_050).with_hop_ms(10).try_validate().is_err(),
            "220.5 samples"
        );
    }

    /// The default front end: what every packaging test pins.
    fn front_end() -> LogMelFrontEndConfig {
        LogMelFrontEndConfig::new()
    }

    fn package_mels(joined: Tensor<3>) -> Tensor<3> {
        front_end().package_mels(joined)
    }

    fn package_window(
        window: Tensor<3>,
        reference: Tensor<1>,
    ) -> Tensor<3> {
        front_end().package_window(window, reference)
    }

    /// At the default front end the derived geometry is
    /// `MelConverterOptions`' own default; at another rate the 10 ms / 25 ms
    /// grid scales and nothing else moves.
    #[test]
    fn test_mel_options_derives_the_grid_from_the_rate() {
        let base = PerceptiveAudioConverterConfig::default();

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
        ErrorMatcher::illegal()
            .assert_err(&front_end().with_sample_rate(0).mel_converter_options(80));
    }

    /// The clamp range is the front end's, not a constant.
    #[test]
    fn test_package_window_uses_the_configured_range() {
        let device = cpu_device();
        let window: Tensor<3> =
            Tensor::from_data(TensorData::new(vec![0.0_f64, -20.0], [1, 2, 1]), &device);
        let reference: Tensor<1> = Tensor::from_data(TensorData::new(vec![0.0_f64], [1]), &device);

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
        let device = cpu_device();

        for batch in 1..4 {
            for n_mels in 1..5 {
                for frames in 2..12 {
                    let joined: Tensor<3> = Tensor::zeros([batch, frames, n_mels], &device);

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
        let device = cpu_device();
        let n_mels = 2;

        for frames in 2..10 {
            let mut data = vec![0.0_f64; frames * n_mels];
            for slot in data.iter_mut().skip((frames - 1) * n_mels) {
                *slot = 99.0;
            }

            let joined: Tensor<3> =
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
        let device = cpu_device();
        let joined: Tensor<3> = Tensor::zeros([1, 1, 4], &device);

        let _ = package_mels(joined);
    }

    /// The other degenerate count: zero frames would slice `0..-1`.
    #[test]
    #[should_panic(expected = "at least 2 frames")]
    fn test_package_mels_rejects_zero_frames() {
        let device = cpu_device();
        let joined: Tensor<3> = Tensor::zeros([1, 0, 4], &device);

        let _ = package_mels(joined);
    }

    /// The trailing frame is dropped **before** the clamp takes its
    /// reference, so an outlier in it cannot set the floor for everything
    /// else. That ordering is the part worth pinning.
    #[test]
    fn test_package_mels_clamps_after_dropping_the_frame() {
        let device = cpu_device();

        // One row, one mel, three frames. The last is an outlier that would
        // dominate the maximum if it survived to the clamp.
        let joined: Tensor<3> = Tensor::from_data(
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
        let device = cpu_device();
        let (batch, frames, n_mels) = (3, 7, 4);

        let data: Vec<f64> = (0..batch * frames * n_mels)
            .map(|k| ((k * 37) % 23) as f64 - 15.0 + (k / (frames * n_mels)) as f64 * 3.0)
            .collect();
        let joined: Tensor<3> =
            Tensor::from_data(TensorData::new(data, [batch, frames, n_mels]), &device);

        let whole = package_mels(joined.clone());

        let window = drop_last_frame(joined);
        let split = package_window(window.clone(), MaxLevel::of(window).max);

        assert_eq!(split.dims(), [batch, n_mels, frames - 1]);
        split.to_data().assert_eq(&whole.to_data(), true);
    }

    /// The reference is per row: raising one row's reference floors that row
    /// harder and leaves the others alone.
    #[test]
    fn test_package_window_floors_each_row_against_its_reference() {
        let device = cpu_device();

        // Two rows, two frames, one mel: `[0, -20]` in each row.
        let window: Tensor<3> = Tensor::from_data(
            TensorData::new(vec![0.0_f64, -20.0, 0.0, -20.0], [2, 2, 1]),
            &device,
        );

        // Row 0 against its own maximum (0): floor -8. Row 1 against a
        // reference of 12, as if something louder had been heard: floor 4,
        // which lifts both of its values.
        let reference: Tensor<1> =
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
        let device = cpu_device();
        let window: Tensor<3> =
            Tensor::from_data(TensorData::new(vec![4.0_f64, 0.0], [1, 2, 1]), &device);
        let reference: Tensor<1> = Tensor::from_data(TensorData::new(vec![0.0_f64], [1]), &device);

        // Floor at -8; nothing is below it, so it is the affine alone.
        assert_tensor_close_to_vec(
            &package_window(window, reference),
            &[2.0, 1.0],
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn test_drop_last_frame() {
        let device = cpu_device();
        let stream: Tensor<3, Int> = Tensor::<1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        let actual: Tensor<3, Int> = drop_last_frame(stream.clone());

        let expected: Tensor<3, Int> = stream.slice(s![.., ..-1, ..]);

        actual
            .to_data_as::<i32>()
            .assert_eq(&expected.to_data_as::<i32>(), true);
    }
}
