//! # The ten-vad pitch feature seam.
//!
//! Feature `40` of the ten-vad feature vector is a pitch estimate, in Hz,
//! with `0.0` meaning "unvoiced" (`ALGO_TRACE.md` §3.5).
//!
//! The driver reaches it through [`TenVadPitchSource`], which is tensor-in,
//! tensor-out so the front end can stay device-resident. Implementations:
//!
//! * [`ZeroPitch`] — a constant stub that never inspects its input.
//! * [`HostPitch`](super::HostPitch) — adapts a host-side
//!   [`TenVadPitchScalarSource`] (notably
//!   [`TenVadPitchEstimator`](super::TenVadPitchEstimator), the reference port)
//!   at the cost of a device-to-host readback.
//!
//! ## Why there are three traits
//!
//! * [`TenVadPitchSource`] is the device seam the driver calls.
//! * [`TenVadPitchSourceInit`] builds one. A tensor-native source has to
//!   *allocate* its carried buffers for a `(batch_size, device)` pair, so it
//!   cannot be a prototype cloned per batch row the way a host source can.
//! * [`TenVadPitchScalarSource`] is the per-stream host contract the reference
//!   estimator implements, kept separate because the reference algorithm is a
//!   serial recurrence over scalars, not a tensor op.

use burn::prelude::*;

use crate::{
    errors::WithOkOrPanic,
    kits::speech::ten_vad::context::coeff::{
        FEATURE_EPS,
        FEATURE_MEANS,
        FEATURE_STDS,
        N_MELS,
    },
    prelude::BunsenResult,
};

/// A source for the ten-vad pitch feature.
///
/// The driver holds exactly one of these per context, covering every stream in
/// the batch. Built by [`TenVadPitchSourceInit`].
pub trait TenVadPitchSource<B: Backend> {
    /// Estimates the pitch of one hop.
    ///
    /// # Arguments
    /// * `raw`: `[batch, hop_size]` samples at the reference's int16 scale.
    ///   These are the **raw** samples: pre-emphasis is applied only to the
    ///   STFT branch, never to the pitch branch (`ALGO_TRACE.md` §3.3).
    /// * `bin_power`: `[batch, n_bins]` bin powers, `re^2 + im^2`, **before**
    ///   the `1 / 32768^2` normalization the mel branch applies.
    ///
    /// # Returns
    /// `[batch, 1]` pitch in Hz, `0.0` where nothing voiced was detected. The
    /// trailing axis is the feature column, so the driver's concatenation onto
    /// the log-mel block is free.
    fn forward(
        &mut self,
        raw: Tensor<B, 2>,
        bin_power: Tensor<B, 2>,
    ) -> Tensor<B, 2>;

    /// Estimates the pitch of `steps` consecutive hops.
    ///
    /// Equivalent to `steps` calls of [`forward`](Self::forward).
    ///
    /// # Arguments
    /// * `raw`: `[steps, batch, hop_size]` consecutive raw hops.
    /// * `bin_power`: `[steps, batch, n_bins]` consecutive bin powers.
    ///
    /// # Returns
    /// `[steps, batch, 1]` pitch in Hz.
    fn forward_sequence(
        &mut self,
        raw: Tensor<B, 3>,
        bin_power: Tensor<B, 3>,
    ) -> Tensor<B, 3>;

    /// Resets any carried state to the start-of-stream condition.
    fn reset(&mut self);
}

/// Builds a [`TenVadPitchSource`] bound to a batch size and a device.
///
/// This is the seam
/// [`TenVadFeatures::init_state`](crate::kits::speech::ten_vad::context::TenVadFeatures::init_state)
/// threads through. It exists because a tensor-native source allocates its
/// carried buffers at construction and so cannot be cloned per batch row.
pub trait TenVadPitchSourceInit<B: Backend> {
    /// The source this builds.
    type Source: TenVadPitchSource<B>;

    /// Builds a start-of-stream source over `batch_size` independent streams.
    ///
    /// # Arguments
    /// * `batch_size`: the number of independent streams; must be non-zero.
    /// * `device`: the device the carried buffers are allocated on.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`](crate::prelude::BunsenError::Invalid) if the
    /// source geometry is invalid.
    fn try_init_source(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> BunsenResult<Self::Source>;

    /// Builds a start-of-stream source, panicking on error.
    ///
    /// See [`try_init_source`](Self::try_init_source).
    fn init_source(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Self::Source {
        self.try_init_source(batch_size, device).ok_or_panic()
    }
}

/// A host-side, per-stream scalar pitch estimator.
///
/// The reference algorithm is a serial recurrence over scalars rather than a
/// tensor op, so it is expressed here and adapted to the device seam by
/// [`HostPitch`](super::HostPitch). Implemented by
/// [`TenVadPitchEstimator`](super::TenVadPitchEstimator).
pub trait TenVadPitchScalarSource {
    /// Estimates the pitch of one hop.
    ///
    /// # Arguments
    /// * `raw_hop` - the hop's samples at the reference's int16 scale, **raw**
    ///   rather than pre-emphasized.
    /// * `bin_power` - the `[n_bins]` bin powers, **before** the `1 / 32768^2`
    ///   normalization the mel branch applies.
    ///
    /// # Returns
    /// The pitch in Hz, or `0.0` when nothing voiced was detected.
    fn frame_pitch(
        &mut self,
        raw_hop: &[f32],
        bin_power: &[f32],
    ) -> f32;

    /// Resets any carried state to the start-of-stream condition.
    fn reset(&mut self);
}

/// A [`TenVadPitchSource`] that always reports unvoiced.
///
/// Feature `40` is then pinned to the constant
/// `(0.0 - FEATURE_MEANS[40]) / (FEATURE_STDS[40] + FEATURE_EPS)`, which
/// [`ZeroPitch::normalized_feature`] reports.
///
/// The other 40 features are unaffected: nothing upstream of the pitch branch
/// reads its output. This is a deliberate approximation, not a placeholder —
/// it never inspects its arguments, so the whole front end stays on-device,
/// which the faithful sources cannot offer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZeroPitch;

impl ZeroPitch {
    /// The normalized value feature `40` takes under [`ZeroPitch`].
    pub fn normalized_feature() -> f32 {
        (0.0 - FEATURE_MEANS[N_MELS]) / (FEATURE_STDS[N_MELS] + FEATURE_EPS)
    }
}

impl<B: Backend> TenVadPitchSource<B> for ZeroPitch {
    fn forward(
        &mut self,
        raw: Tensor<B, 2>,
        _bin_power: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        Tensor::zeros([raw.dims()[0], 1], &raw.device())
    }

    fn forward_sequence(
        &mut self,
        raw: Tensor<B, 3>,
        _bin_power: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [steps, batch, _] = raw.dims();
        Tensor::zeros([steps, batch, 1], &raw.device())
    }

    fn reset(&mut self) {}
}

impl<B: Backend> TenVadPitchSourceInit<B> for ZeroPitch {
    type Source = ZeroPitch;

    fn try_init_source(
        &self,
        _batch_size: usize,
        _device: &B::Device,
    ) -> BunsenResult<Self::Source> {
        Ok(ZeroPitch)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    use super::*;
    use crate::support::testing::PerformanceBackend;

    type B = PerformanceBackend;

    #[test]
    fn test_zero_pitch_is_always_unvoiced() {
        let device = Default::default();
        let mut pitch = ZeroPitch;

        let raw = Tensor::<B, 2>::from_floats([[1.0, -2.0, 3.0]], &device);
        let power = Tensor::<B, 2>::ones([1, 513], &device);

        let out = TenVadPitchSource::<B>::forward(&mut pitch, raw, power);
        assert_eq!(out.dims(), [1, 1]);
        assert_eq!(out.into_scalar().elem::<f32>(), 0.0);
    }

    #[test]
    fn test_zero_pitch_sequence_shape_and_value() {
        let device = Default::default();
        let mut pitch = ZeroPitch;

        let raw = Tensor::<B, 3>::zeros([5, 2, 256], &device);
        let power = Tensor::<B, 3>::ones([5, 2, 513], &device);

        let out = TenVadPitchSource::<B>::forward_sequence(&mut pitch, raw, power);
        assert_eq!(out.dims(), [5, 2, 1]);
        assert_eq!(out.sum().into_scalar().elem::<f32>(), 0.0);
    }

    #[test]
    fn test_zero_pitch_ignores_its_input() {
        // The whole point of ZeroPitch: it reads shape and device, never data,
        // so the driver's sequence path never has to synchronize.
        let device = Default::default();
        let mut pitch = ZeroPitch;

        let quiet = Tensor::<B, 2>::zeros([1, 256], &device);
        let loud = Tensor::<B, 2>::full([1, 256], 30000.0, &device);
        let power = Tensor::<B, 2>::ones([1, 513], &device);

        let a = TenVadPitchSource::<B>::forward(&mut pitch, quiet, power.clone());
        let b = TenVadPitchSource::<B>::forward(&mut pitch, loud, power);
        a.into_data().assert_eq(&b.into_data(), true);
    }

    #[test]
    fn test_zero_pitch_reset_is_stateless() {
        let device = Default::default();
        let mut pitch = ZeroPitch;
        let raw = Tensor::<B, 2>::ones([1, 256], &device);
        let power = Tensor::<B, 2>::ones([1, 513], &device);

        let before = TenVadPitchSource::<B>::forward(&mut pitch, raw.clone(), power.clone());
        TenVadPitchSource::<B>::reset(&mut pitch);
        let after = TenVadPitchSource::<B>::forward(&mut pitch, raw, power);

        before.into_data().assert_eq(&after.into_data(), true);
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
        let device = Default::default();
        let mut pitch: Box<dyn TenVadPitchSource<B>> = Box::new(ZeroPitch);

        let raw = Tensor::<B, 2>::zeros([1, 256], &device);
        let power = Tensor::<B, 2>::zeros([1, 513], &device);
        assert_eq!(pitch.forward(raw, power).dims(), [1, 1]);
        pitch.reset();
    }

    #[test]
    fn test_zero_pitch_init_ignores_batch_and_device() {
        let device = Default::default();
        let built = TenVadPitchSourceInit::<B>::init_source(&ZeroPitch, 4, &device);
        assert_eq!(built, ZeroPitch);
    }

    /// A minimal scalar implementation, to prove that seam is usable too.
    #[derive(Debug, Clone, Default)]
    struct CountingPitch {
        calls: usize,
    }

    impl TenVadPitchScalarSource for CountingPitch {
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
    fn test_scalar_seam_is_usable() {
        let mut pitch = CountingPitch::default();

        assert_eq!(pitch.frame_pitch(&[0.0; 4], &[]), 4.0);
        assert_eq!(pitch.calls, 1);

        pitch.reset();
        assert_eq!(pitch.calls, 0);
    }
}
