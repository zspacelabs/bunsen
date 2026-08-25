//! # Host-side pitch sources, adapted to the device seam.
//!
//! [`HostPitch`] wraps a per-stream [`PitchScalarSource`] — notably
//! [`HostPitchEstimator`](super::HostPitchEstimator), the reference port —
//! and presents it as a [`PitchSource`].
//!
//! ## The readback lives here, and only here
//!
//! The reference pitch algorithm is a serial recurrence over scalars, so
//! stepping it means bringing the raw hops and the bin powers back to the
//! host. [`HostPitch`] is the one place in the front end that synchronizes,
//! which is the point of isolating it: everything upstream and downstream of
//! this file is device-resident tensor code.
//!
//! The cost is **one** readback per call, not one per hop —
//! [`forward_sequence`](HostPitch::forward_sequence) reads the whole
//! `[steps, batch, ..]` block once, walks it host-side, and uploads a single
//! `[steps, batch, 1]` column.

use burn::prelude::*;

use super::source::{
    PitchScalarSource,
    PitchSource,
    PitchSourceInit,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    prelude::{
        TensorDataToVecAsExt,
        TensorElemOpExt,
    },
};

/// Adapts a per-stream host [`PitchScalarSource`] to the device seam.
///
/// Holds one scalar estimator per batch row, cloned from a prototype. Built by
/// [`HostPitch::new`] or, through the driver, by [`HostPitchInit`].
#[derive(Debug, Clone, PartialEq)]
pub struct HostPitch<P: PitchScalarSource> {
    /// The per-stream estimators; one entry per batch row.
    pub sources: Vec<P>,
}

impl<P: PitchScalarSource + Clone> HostPitch<P> {
    /// Builds a host adapter over `batch_size` independent streams.
    ///
    /// # Arguments
    /// * `prototype`: cloned once per batch row.
    /// * `batch_size`: the number of independent streams; must be non-zero.
    ///
    /// # Panics
    /// If `batch_size` is zero.
    pub fn new(
        prototype: P,
        batch_size: usize,
    ) -> Self {
        assert_ne!(batch_size, 0, "HostPitch batch_size must be non-zero");
        Self {
            sources: vec![prototype; batch_size],
        }
    }
}

impl<P: PitchScalarSource> HostPitch<P> {
    /// The batch size; each entry is an independent stream.
    pub fn batch_size(&self) -> usize {
        self.sources.len()
    }
}

impl<B: Backend, P: PitchScalarSource> PitchSource<B> for HostPitch<P> {
    fn forward(
        &mut self,
        raw: Tensor<B, 2>,
        bin_power: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, hop_size] = raw.dims();
        let [power_batch, n_bins] = bin_power.dims();
        assert_eq!(batch, self.batch_size(), "HostPitch batch mismatch");
        assert_eq!(power_batch, batch, "raw and bin_power disagree on batch");

        let device = raw.device();
        let raw_host: Vec<f32> = raw.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
        let power_host: Vec<f32> = bin_power
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();

        let values: Vec<f32> = (0..batch)
            .map(|b| {
                self.sources[b].frame_pitch(
                    &raw_host[b * hop_size..(b + 1) * hop_size],
                    &power_host[b * n_bins..(b + 1) * n_bins],
                )
            })
            .collect();

        Tensor::from_data(TensorData::new(values, [batch, 1]), &device)
    }

    /// Pitch is a per-stream recurrence, so the frames are walked in order,
    /// one host-side call per stream per step — but the device is read back
    /// only once, for the whole sequence.
    fn forward_sequence(
        &mut self,
        raw: Tensor<B, 3>,
        bin_power: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [steps, batch, hop_size] = raw.dims();
        let [power_steps, power_batch, n_bins] = bin_power.dims();
        assert_eq!(batch, self.batch_size(), "HostPitch batch mismatch");
        assert_eq!(power_batch, batch, "raw and bin_power disagree on batch");
        assert_eq!(power_steps, steps, "raw and bin_power disagree on steps");

        let device = raw.device();
        let raw_host: Vec<f32> = raw.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
        let power_host: Vec<f32> = bin_power
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();

        let mut values = Vec::with_capacity(steps * batch);
        for step in 0..steps {
            for b in 0..batch {
                let raw_at = (step * batch + b) * hop_size;
                let pow_at = (step * batch + b) * n_bins;
                values.push(self.sources[b].frame_pitch(
                    &raw_host[raw_at..raw_at + hop_size],
                    &power_host[pow_at..pow_at + n_bins],
                ));
            }
        }

        Tensor::from_data(TensorData::new(values, [steps, batch, 1]), &device)
    }

    fn reset(&mut self) {
        for source in &mut self.sources {
            source.reset();
        }
    }
}

/// Builds a [`HostPitch`] from a scalar prototype.
///
/// Third-party [`PitchScalarSource`] implementations reach the driver
/// through this; the ten-vad reference estimator has its own
/// [`PitchSourceInit`] impl so that
/// `init_context_with(cfg, HostPitchEstimator::new(), device)` reads
/// naturally.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostPitchInit<P>(pub P);

impl<B: Backend, P: PitchScalarSource + Clone> PitchSourceInit<B> for HostPitchInit<P> {
    type Source = HostPitch<P>;

    fn try_init_source(
        &self,
        batch_size: usize,
        _device: &B::Device,
    ) -> BunsenResult<Self::Source> {
        if batch_size == 0 {
            return Err(BunsenError::Invalid(
                "HostPitch batch_size must be non-zero".to_string(),
            ));
        }
        Ok(HostPitch::new(self.0.clone(), batch_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::PerformanceBackend;

    type B = PerformanceBackend;

    /// Reports the hop's first sample, so a test can tell rows apart, and
    /// counts calls so a test can tell whether it was stepped once per frame.
    #[derive(Debug, Clone, Default, PartialEq)]
    struct EchoPitch {
        calls: usize,
        last: f32,
    }

    impl PitchScalarSource for EchoPitch {
        fn frame_pitch(
            &mut self,
            raw_hop: &[f32],
            _bin_power: &[f32],
        ) -> f32 {
            self.calls += 1;
            self.last = raw_hop[0];
            self.last
        }

        fn reset(&mut self) {
            self.calls = 0;
            self.last = 0.0;
        }
    }

    #[test]
    fn test_forward_routes_each_row_to_its_own_source() {
        let device = Default::default();
        let mut pitch = HostPitch::new(EchoPitch::default(), 3);

        let raw = Tensor::<B, 2>::from_floats([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], &device);
        let power = Tensor::<B, 2>::ones([3, 4], &device);

        let out = PitchSource::<B>::forward(&mut pitch, raw, power);
        out.into_data()
            .assert_eq(&TensorData::from([[10.0f32], [20.0], [30.0]]), true);

        for source in &pitch.sources {
            assert_eq!(source.calls, 1);
        }
    }

    #[test]
    fn test_forward_sequence_matches_stepwise() {
        let device = Default::default();
        let steps = 4;
        let batch = 2;

        let raw = Tensor::<B, 3>::from_floats(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
                [[7.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            ],
            &device,
        );
        let power = Tensor::<B, 3>::ones([steps, batch, 5], &device);

        let mut seq = HostPitch::new(EchoPitch::default(), batch);
        let seq_out = PitchSource::<B>::forward_sequence(&mut seq, raw.clone(), power.clone());

        let mut step = HostPitch::new(EchoPitch::default(), batch);
        let mut rows = Vec::new();
        for s in 0..steps {
            let r = raw
                .clone()
                .slice_dim(0, s as isize..(s + 1) as isize)
                .squeeze_dim::<2>(0);
            let p = power
                .clone()
                .slice_dim(0, s as isize..(s + 1) as isize)
                .squeeze_dim::<2>(0);
            rows.push(PitchSource::<B>::forward(&mut step, r, p));
        }
        let step_out: Tensor<B, 3> = Tensor::stack(rows, 0);

        seq_out.into_data().assert_eq(&step_out.into_data(), true);
        // The residual state must match too, not just the output.
        assert_eq!(seq.sources, step.sources);
    }

    #[test]
    fn test_reset_rewinds_every_row() {
        let device = Default::default();
        let mut pitch = HostPitch::new(EchoPitch::default(), 2);

        let raw = Tensor::<B, 2>::from_floats([[5.0], [6.0]], &device);
        let power = Tensor::<B, 2>::ones([2, 2], &device);
        PitchSource::<B>::forward(&mut pitch, raw, power);
        assert!(pitch.sources.iter().all(|s| s.calls == 1));

        PitchSource::<B>::reset(&mut pitch);
        assert!(pitch.sources.iter().all(|s| *s == EchoPitch::default()));
    }

    #[test]
    fn test_init_rejects_zero_batch() {
        let device = Default::default();
        let init = HostPitchInit(EchoPitch::default());
        assert!(PitchSourceInit::<B>::try_init_source(&init, 0, &device).is_err());
        assert!(PitchSourceInit::<B>::try_init_source(&init, 2, &device).is_ok());
    }

    #[test]
    fn test_init_clones_the_prototype_per_row() {
        let device = Default::default();
        let init = HostPitchInit(EchoPitch::default());
        let built = PitchSourceInit::<B>::init_source(&init, 3, &device);
        assert_eq!(built.batch_size(), 3);
    }
}
