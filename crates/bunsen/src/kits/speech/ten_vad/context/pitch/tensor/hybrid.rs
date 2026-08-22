//! # The staged-migration harness.
//!
//! [`HybridPitch`] runs the pre-filter design on the device and the remaining
//! three stages on the host, so a device stage can be measured against the C
//! reference **end to end** while everything downstream of it is held fixed.
//!
//! That is the point: the stage-level differential tests say a device stage
//! reproduces its host counterpart to a tolerance, but they cannot say whether
//! that tolerance survives the tracker's `argmax` and voicing threshold. This
//! can, by driving the real golden through a pipeline that differs from the
//! pinned one in exactly one stage.
//!
//! The split is exact rather than approximate: the pre-filter design carries
//! no state, and nothing downstream rewrites the coefficients it produces, so
//! [`TenVadPitchEstimator::frame_pitch_with_lpc`] resumes from precisely the
//! point [`frame_pitch`](TenVadPitchScalarSource::frame_pitch) would have
//! reached.
//!
//! Test-only, and deliberately so — it is scaffolding for the migration, not
//! a configuration anyone should ship. It comes out when the last stage lands.

use burn::prelude::*;

use super::{
    super::{
        TenVadPitchEstimator,
        TenVadPitchSource,
        TenVadPitchSourceInit,
        coeff::LPC_ORDER,
    },
    prefilter::{
        PitchPrefilter,
        PitchPrefilterConfig,
    },
};
use crate::{
    errors::{
        BunsenResult,
        WithOkOrPanic,
    },
    prelude::{
        TensorDataToVecAsExt,
        TensorElemOpExt,
    },
};

/// Device pre-filter design, host everything else.
///
/// See the module docs. Built by [`HybridPitch::new`].
#[derive(Debug, Clone)]
pub(crate) struct HybridPitch<B: Backend> {
    /// The device-side stage under test.
    pub prefilter: PitchPrefilter<B>,

    /// The host estimators, one per stream, resumed past their own stage 1.
    pub sources: Vec<TenVadPitchEstimator>,
}

impl<B: Backend> HybridPitch<B> {
    /// Builds a hybrid source over `batch_size` independent streams.
    ///
    /// # Panics
    /// If `batch_size` is zero.
    pub fn new(
        prefilter: PitchPrefilter<B>,
        batch_size: usize,
    ) -> Self {
        assert_ne!(batch_size, 0, "HybridPitch batch_size must be non-zero");
        Self {
            prefilter,
            sources: vec![TenVadPitchEstimator::new(); batch_size],
        }
    }

    /// The batch size.
    pub fn batch_size(&self) -> usize {
        self.sources.len()
    }

    /// Steps every stream over one hop's worth of already-read-back host data.
    ///
    /// # Arguments
    /// * `raw_host`: `rows * hop_size` raw samples, row-major.
    /// * `lpc_host`: `rows * LPC_ORDER` device-designed coefficients.
    /// * `rows`: how many `(step, stream)` pairs are present.
    /// * `hop_size`: samples per hop.
    /// * `batch`: the stream count, so row `r` maps to stream `r % batch`.
    fn step_rows(
        &mut self,
        raw_host: &[f32],
        lpc_host: &[f32],
        rows: usize,
        hop_size: usize,
        batch: usize,
    ) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                let mut lpc = [0.0f32; LPC_ORDER];
                lpc.copy_from_slice(&lpc_host[row * LPC_ORDER..(row + 1) * LPC_ORDER]);
                self.sources[row % batch]
                    .frame_pitch_with_lpc(&raw_host[row * hop_size..(row + 1) * hop_size], &lpc)
            })
            .collect()
    }
}

impl<B: Backend> TenVadPitchSource<B> for HybridPitch<B> {
    fn forward(
        &mut self,
        raw: Tensor<B, 2>,
        bin_power: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, hop_size] = raw.dims();
        assert_eq!(batch, self.batch_size(), "HybridPitch batch mismatch");

        let device = raw.device();
        let lpc = self.prefilter.forward(bin_power);

        let raw_host: Vec<f32> = raw.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
        let lpc_host: Vec<f32> = lpc.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();

        let values = self.step_rows(&raw_host, &lpc_host, batch, hop_size, batch);
        Tensor::from_data(TensorData::new(values, [batch, 1]), &device)
    }

    fn forward_sequence(
        &mut self,
        raw: Tensor<B, 3>,
        bin_power: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [steps, batch, hop_size] = raw.dims();
        let n_bins = bin_power.dims()[2];
        assert_eq!(batch, self.batch_size(), "HybridPitch batch mismatch");

        let device = raw.device();

        // Stage 1 is stateless, so the whole sequence designs in one pass.
        let lpc = self
            .prefilter
            .forward(bin_power.reshape([steps * batch, n_bins]));

        let raw_host: Vec<f32> = raw.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
        let lpc_host: Vec<f32> = lpc.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();

        // Row-major `[steps, batch]` means row `r` is stream `r % batch`, which
        // walks each stream's hops in order.
        let values = self.step_rows(&raw_host, &lpc_host, steps * batch, hop_size, batch);
        Tensor::from_data(TensorData::new(values, [steps, batch, 1]), &device)
    }

    fn reset(&mut self) {
        use super::super::TenVadPitchScalarSource;
        for source in &mut self.sources {
            source.reset();
        }
    }
}

/// Builds a [`HybridPitch`] for the driver.
///
/// Holds only the stage's config: the prefilter's tables are device-resident,
/// so they cannot be built until `try_init_source` is handed a device.
#[derive(Debug, Clone)]
pub(crate) struct HybridPitchInit(pub PitchPrefilterConfig);

impl HybridPitchInit {
    /// Builds an init over the default ten-vad pre-filter geometry.
    pub fn new() -> Self {
        Self(PitchPrefilterConfig::new())
    }
}

impl<B: Backend> TenVadPitchSourceInit<B> for HybridPitchInit {
    type Source = HybridPitch<B>;

    fn try_init_source(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> BunsenResult<Self::Source> {
        Ok(HybridPitch::new(self.0.try_init(device)?, batch_size))
    }
}
