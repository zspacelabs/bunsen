//! # The ten-vad pitch feature seam.
//!
//! Feature `40` of the ten-vad feature vector is a pitch estimate, in Hz,
//! with `0.0` meaning "unvoiced" (`ALGO_TRACE.md` §3.5).
//!
//! The driver reaches it through [`PitchSource`], which is tensor-in,
//! tensor-out so the front end can stay device-resident. Implementations:
//!
//! * [`ZeroPitch`] — a constant stub that never inspects its input.
//! * [`HostPitch`](super::HostPitch) — adapts a host-side [`PitchScalarSource`]
//!   (notably [`HostPitchEstimator`](super::HostPitchEstimator), the reference
//!   port) at the cost of a device-to-host readback.
//!
//! ## Why there are three traits
//!
//! * [`PitchSource`] is the device seam the driver calls.
//! * [`PitchSourceInit`] builds one. A tensor-native source has to *allocate*
//!   its carried buffers for a `(batch_size, device)` pair, so it cannot be a
//!   prototype cloned per batch row the way a host source can.
//! * [`PitchScalarSource`] is the per-stream host contract the reference
//!   estimator implements, kept separate because the reference algorithm is a
//!   serial recurrence over scalars, not a tensor op.

use burn::prelude::*;

use super::{
    estimator::HostPitchEstimator,
    host::HostPitch,
    tensor::source::{
        TensorPitchConfig,
        TensorPitchContext,
    },
};
use crate::{
    errors::WithOkOrPanic,
    prelude::BunsenResult,
};

/// A source for the ten-vad pitch feature.
///
/// The driver holds exactly one of these per context, covering every stream in
/// the batch. Built by [`PitchSourceInit`].
pub trait PitchSource<B: Backend> {
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

/// Builds a [`PitchSource`] bound to a batch size and a device.
///
/// This is the seam
/// threads through. It exists because a tensor-native source allocates its
/// carried buffers at construction and so cannot be cloned per batch row.
pub trait PitchSourceInit<B: Backend> {
    /// The source this builds.
    type Source: PitchSource<B>;

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
/// [`HostPitchEstimator`](super::HostPitchEstimator).
pub trait PitchScalarSource {
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

/// A [`PitchSource`] that always reports unvoiced.
///
/// Always reports `0.0` Hz, and never inspects its arguments, so a pipeline
/// using it stays entirely on-device.
///
/// A deliberate approximation rather than a placeholder: a caller that does
/// not need pitch, or that wants to measure what the branch costs, can drop it
/// without changing anything upstream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZeroPitch;

impl<B: Backend> PitchSource<B> for ZeroPitch {
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

impl<B: Backend> PitchSourceInit<B> for ZeroPitch {
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
    use burn::tensor::{
        Tensor,
        TensorData,
        Tolerance,
    };

    use super::*;
    use crate::{
        errors::WithOkOrPanic,
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    #[test]
    fn test_zero_pitch_is_always_unvoiced() {
        let device = Default::default();
        let mut pitch = ZeroPitch;

        let raw = Tensor::<B, 2>::from_floats([[1.0, -2.0, 3.0]], &device);
        let power = Tensor::<B, 2>::ones([1, 513], &device);

        let out = PitchSource::<B>::forward(&mut pitch, raw, power);
        assert_eq!(out.dims(), [1, 1]);
        assert_eq!(out.into_scalar().elem::<f32>(), 0.0);
    }

    #[test]
    fn test_zero_pitch_sequence_shape_and_value() {
        let device = Default::default();
        let mut pitch = ZeroPitch;

        let raw = Tensor::<B, 3>::zeros([5, 2, 256], &device);
        let power = Tensor::<B, 3>::ones([5, 2, 513], &device);

        let out = PitchSource::<B>::forward_sequence(&mut pitch, raw, power);
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

        let a = PitchSource::<B>::forward(&mut pitch, quiet, power.clone());
        let b = PitchSource::<B>::forward(&mut pitch, loud, power);
        a.into_data().assert_eq(&b.into_data(), true);
    }

    #[test]
    fn test_zero_pitch_reset_is_stateless() {
        let device = Default::default();
        let mut pitch = ZeroPitch;
        let raw = Tensor::<B, 2>::ones([1, 256], &device);
        let power = Tensor::<B, 2>::ones([1, 513], &device);

        let before = PitchSource::<B>::forward(&mut pitch, raw.clone(), power.clone());
        PitchSource::<B>::reset(&mut pitch);
        let after = PitchSource::<B>::forward(&mut pitch, raw, power);

        before.into_data().assert_eq(&after.into_data(), true);
        assert_eq!(pitch, ZeroPitch);
    }

    #[test]
    fn test_usable_as_a_trait_object() {
        let device = Default::default();
        let mut pitch: Box<dyn PitchSource<B>> = Box::new(ZeroPitch);

        let raw = Tensor::<B, 2>::zeros([1, 256], &device);
        let power = Tensor::<B, 2>::zeros([1, 513], &device);
        assert_eq!(pitch.forward(raw, power).dims(), [1, 1]);
        pitch.reset();
    }

    #[test]
    fn test_zero_pitch_init_ignores_batch_and_device() {
        let device = Default::default();
        let built = PitchSourceInit::<B>::init_source(&ZeroPitch, 4, &device);
        assert_eq!(built, ZeroPitch);
    }

    /// A minimal scalar implementation, to prove that seam is usable too.
    #[derive(Debug, Clone, Default)]
    struct CountingPitch {
        calls: usize,
    }

    impl PitchScalarSource for CountingPitch {
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

    // ---- PitchSourceConfig / PitchSourceKind ----------------------------
    //
    // The module's public entry point: a caller picks a variant by config and
    // drives whatever it produces through the `PitchSource` seam. Every test
    // below runs the *same* assertion across all three variants, because the
    // whole point of the seam is that they are interchangeable.

    const HOP: usize = 256;
    const BINS: usize = 513;

    /// Every variant a caller can select.
    fn all_configs() -> Vec<(&'static str, PitchSourceConfig)> {
        vec![
            ("zero", PitchSourceConfig::Zero),
            ("host", PitchSourceConfig::Host),
            (
                "tensor",
                PitchSourceConfig::Tensor(TensorPitchConfig::new()),
            ),
            (
                "tensor/recurrence",
                PitchSourceConfig::Tensor(TensorPitchConfig::reference()),
            ),
        ]
    }

    /// A voiced-ish hop and a plausible spectrum for it.
    fn frame(
        batch: usize,
        seed: f32,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let raw: Vec<f32> = (0..batch * HOP)
            .map(|i| 4000.0 * (((i % HOP) as f32 + seed) * 0.08).sin())
            .collect();
        let power: Vec<f32> = (0..batch * BINS)
            .map(|i| 1e5 * (-((i % BINS) as f32) / 60.0).exp() + 10.0)
            .collect();
        (
            Tensor::from_data(TensorData::new(raw, [batch, HOP]), device),
            Tensor::from_data(TensorData::new(power, [batch, BINS]), device),
        )
    }

    #[test]
    fn test_default_config_is_the_device_estimator() {
        assert!(matches!(
            PitchSourceConfig::default(),
            PitchSourceConfig::Tensor(_),
        ));
    }

    #[test]
    fn test_each_config_builds_its_own_kind() {
        let device = Default::default();

        assert!(matches!(
            PitchSourceConfig::Zero.init_source(1, &device),
            PitchSourceKind::<B>::Zero(_),
        ));
        assert!(matches!(
            PitchSourceConfig::Host.init_source(1, &device),
            PitchSourceKind::<B>::Host(_),
        ));
        assert!(matches!(
            PitchSourceConfig::Tensor(TensorPitchConfig::new()).init_source(1, &device),
            PitchSourceKind::<B>::Tensor(_),
        ));
    }

    #[test]
    fn test_try_init_reports_an_invalid_tensor_config() {
        // The only variant with geometry to get wrong, and the only reason
        // `try_init_source` is fallible at all.
        let device = Default::default();
        let bad = PitchSourceConfig::Tensor(TensorPitchConfig::new().with_chunk_steps(Some(0)));

        assert!(matches!(
            PitchSourceInit::<B>::try_init_source(&bad, 1, &device),
            Err(BunsenError::Invalid(_)),
        ));
    }

    #[test]
    fn test_every_kind_reports_a_plausible_pitch() {
        // The seam's actual contract: `[batch, hop] + [batch, bins]` in,
        // `[batch, 1]` of non-negative hertz out, whichever variant is behind
        // it.
        let device = Default::default();
        let (raw, power) = frame(2, 0.0, &device);

        for (name, cfg) in all_configs() {
            let mut src: PitchSourceKind<B> = cfg.init_source(2, &device);
            let out = src.forward(raw.clone(), power.clone());

            assert_eq!(out.dims(), [2, 1], "{name}");
            for (i, hz) in out
                .to_data_as::<f32>()
                .to_vec_as::<f32>()
                .ok_or_panic()
                .iter()
                .enumerate()
            {
                assert!(hz.is_finite(), "{name} row {i}: {hz} is not finite");
                assert!(*hz >= 0.0, "{name} row {i}: {hz} is negative");
                assert!(*hz < 1000.0, "{name} row {i}: {hz} Hz is implausible");
            }
        }
    }

    #[test]
    fn test_every_kind_agrees_with_itself_across_the_two_entry_points() {
        // `forward` must be the one-step case of `forward_sequence`, or a
        // caller that batches gets a different answer than one that does not.
        let device = Default::default();
        let (raw, power) = frame(1, 3.0, &device);

        for (name, cfg) in all_configs() {
            let mut stepwise: PitchSourceKind<B> = cfg.init_source(1, &device);
            let single = stepwise.forward(raw.clone(), power.clone());

            let mut batched: PitchSourceKind<B> = cfg.init_source(1, &device);
            let seq = batched.forward_sequence(
                raw.clone().reshape([1, 1, HOP]),
                power.clone().reshape([1, 1, BINS]),
            );

            assert_eq!(seq.dims(), [1, 1, 1], "{name}");
            single.clone().to_data_as::<f32>().assert_approx_eq::<f32>(
                &seq.reshape([1, 1]).to_data_as::<f32>(),
                Tolerance::permissive(),
            );
        }
    }

    #[test]
    fn test_every_kind_carries_state_and_resets_it() {
        // Two hops differ from one hop repeated only if state carried; and
        // after `reset` the first hop must reproduce exactly.
        let device = Default::default();
        let (a, pa) = frame(1, 0.0, &device);
        let (b, pb) = frame(1, 11.0, &device);

        for (name, cfg) in all_configs() {
            let mut src: PitchSourceKind<B> = cfg.init_source(1, &device);

            let first = src.forward(a.clone(), pa.clone());
            src.forward(b.clone(), pb.clone());

            PitchSource::<B>::reset(&mut src);
            let again = src.forward(a.clone(), pa.clone());

            first
                .clone()
                .to_data_as::<f32>()
                .assert_approx_eq::<f32>(&again.to_data_as::<f32>(), Tolerance::permissive());
            let _ = name;
        }
    }

    #[test]
    fn test_stateful_kinds_actually_carry_state() {
        // Guard the guard for the reset test above: it proves nothing unless
        // the source has state to rewind. `Zero` legitimately has none.
        let device = Default::default();
        let (a, pa) = frame(1, 0.0, &device);
        let (b, pb) = frame(1, 11.0, &device);

        for (name, cfg) in [
            ("host", PitchSourceConfig::Host),
            (
                "tensor",
                PitchSourceConfig::Tensor(TensorPitchConfig::new()),
            ),
        ] {
            let mut fresh: PitchSourceKind<B> = cfg.init_source(1, &device);
            let cold = fresh.forward(b.clone(), pb.clone());

            let mut warmed: PitchSourceKind<B> = cfg.init_source(1, &device);
            warmed.forward(a.clone(), pa.clone());
            let warm = warmed.forward(b.clone(), pb.clone());

            let cold_v = cold.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
            let warm_v = warm.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
            assert_ne!(
                cold_v, warm_v,
                "{name}: history should change the estimate, so `reset` has something to undo",
            );
        }
    }

    #[test]
    fn test_kinds_batch_rows_independently() {
        // Row 1 must see its own signal, not row 0's.
        let device = Default::default();
        let (solo, solo_p) = frame(1, 0.0, &device);
        let (other, other_p) = frame(1, 17.0, &device);

        let paired = Tensor::cat(vec![solo.clone(), other], 0);
        let paired_p = Tensor::cat(vec![solo_p.clone(), other_p], 0);

        for (name, cfg) in all_configs() {
            let mut alone: PitchSourceKind<B> = cfg.init_source(1, &device);
            let mut together: PitchSourceKind<B> = cfg.init_source(2, &device);

            let a = alone.forward(solo.clone(), solo_p.clone());
            let t = together.forward(paired.clone(), paired_p.clone());

            let a_v = a.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
            let t_v = t.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
            assert!(
                (a_v[0] - t_v[0]).abs() < 1e-3,
                "{name}: row 0 alone {} vs batched {}",
                a_v[0],
                t_v[0],
            );
        }
    }

    #[test]
    fn test_config_round_trips_through_serde() {
        // `PitchSourceConfig` exists so the choice can live in a serialized
        // config rather than only in code; that is worth checking.
        for (name, cfg) in all_configs() {
            let json = serde_json::to_string(&cfg).expect("serialize");
            let back: PitchSourceConfig = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(
                core::mem::discriminant(&cfg),
                core::mem::discriminant(&back),
                "{name} changed variant through serde",
            );
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

/// How the driver obtains feature `40`.
///
/// A `Config` enum whose variants wrap each implementation's own config,
/// following [`StftWindowConfig`](crate::ops::signal::StftWindowConfig): the
/// contract is [`PitchSourceInit`], and this dispatches to it.
///
/// [`Self::default`] selects [`Self::Tensor`], which is both the faithful
/// choice and the one that keeps the front end device-resident.
#[derive(Config, Debug)]
pub enum PitchSourceConfig {
    /// Pin feature `40` to a constant and skip the branch entirely.
    ///
    /// Never inspects its input, so the sequence path stays on-device with no
    /// synchronization at all. Features `0..40` are exact regardless.
    Zero,

    /// The host scalar estimator, one instance per stream.
    ///
    /// The reference port, and the permanent oracle the device stages are
    /// validated against. Carries no configuration of its own — the geometry
    /// is fixed by the reference. Costs a device-to-host readback per call.
    Host,

    /// The device-side estimator. The default.
    ///
    /// Selecting [`PitchAntiAliasConfig::Recurrence`] inside chooses the
    /// literal-transcription tier instead of the optimized one; see
    /// [`TensorPitchConfig::reference`].
    ///
    /// [`PitchAntiAliasConfig::Recurrence`]: super::tensor::PitchAntiAliasConfig::Recurrence
    Tensor(TensorPitchConfig),
}

impl Default for PitchSourceConfig {
    fn default() -> Self {
        Self::Tensor(TensorPitchConfig::new())
    }
}

impl<B: Backend> PitchSourceInit<B> for PitchSourceConfig {
    type Source = PitchSourceKind<B>;

    fn try_init_source(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> BunsenResult<Self::Source> {
        Ok(match self {
            Self::Zero => PitchSourceKind::Zero(ZeroPitch),
            Self::Host => {
                PitchSourceKind::Host(HostPitch::new(HostPitchEstimator::new(), batch_size))
            }
            Self::Tensor(cfg) => {
                PitchSourceKind::Tensor(cfg.try_init(device)?.init_state(batch_size, device))
            }
        })
    }
}

/// A pitch source selected by [`PitchSourceConfig`].
///
/// An enum rather than a boxed trait object: the contract returns a *stateful*
/// source, and an associated type must be one type across every variant. This
/// keeps dispatch static and the state concrete enough to inspect.
#[derive(Debug, Clone)]
pub enum PitchSourceKind<B: Backend> {
    /// See [`ZeroPitch`].
    Zero(ZeroPitch),
    /// See [`HostPitch`].
    Host(HostPitch<HostPitchEstimator>),
    /// See [`TensorPitchContext`].
    Tensor(TensorPitchContext<B>),
}

impl<B: Backend> PitchSource<B> for PitchSourceKind<B> {
    fn forward(
        &mut self,
        raw: Tensor<B, 2>,
        bin_power: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        match self {
            Self::Zero(s) => s.forward(raw, bin_power),
            Self::Host(s) => s.forward(raw, bin_power),
            Self::Tensor(s) => s.forward(raw, bin_power),
        }
    }

    fn forward_sequence(
        &mut self,
        raw: Tensor<B, 3>,
        bin_power: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        match self {
            Self::Zero(s) => s.forward_sequence(raw, bin_power),
            Self::Host(s) => s.forward_sequence(raw, bin_power),
            Self::Tensor(s) => s.forward_sequence(raw, bin_power),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Zero(s) => PitchSource::<B>::reset(s),
            Self::Host(s) => PitchSource::<B>::reset(s),
            Self::Tensor(s) => PitchSource::<B>::reset(s),
        }
    }
}
