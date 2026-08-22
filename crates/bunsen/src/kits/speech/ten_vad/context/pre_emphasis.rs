//! # Streaming pre-emphasis filter.
//!
//! The first-order high-pass the ten-vad front end applies to the audio
//! before the STFT:
//!
//! ```text
//! y[n] = x[n] - coeff * x[n-1]
//! ```
//!
//! The reference keeps two parallel FIFOs (`ALGO_TRACE.md` §3.3): raw samples
//! feed the pitch estimator, pre-emphasized samples feed the STFT. Only the
//! STFT branch is filtered, and the filter is continuous across hop
//! boundaries — the previous hop's last **raw** sample is carried forward, so
//! a stream chopped into hops yields exactly the same output as the whole
//! stream filtered at once.
//!
//! The pieces are:
//! * [`PreEmphasisConfig`] — the coefficient.
//! * [`PreEmphasisContext`] — a streaming state (the carried sample) bound to a
//!   batch; built by [`PreEmphasisConfig::init`].
//!
//! Note: this is the most generalizable block in the ten-vad front end — a
//! first-order FIR with carry, with nothing ten-vad-specific but its default
//! coefficient. It lives here rather than in [`crate::ops::signal`] until a
//! second caller justifies the general form.

use burn::{
    config::Config,
    prelude::*,
};

#[cfg(any(test, debug_assertions))]
use crate::contracts::unpack_shape_contract;
use crate::{
    kits::speech::ten_vad::context::coeff::PRE_EMPHASIS_COEFF,
    prelude::TensorOpExt,
};

/// Config for [`PreEmphasisContext`].
///
/// Builds [`PreEmphasisContext`] via [`init`](Self::init).
#[derive(Config, Debug, Copy)]
pub struct PreEmphasisConfig {
    /// The pre-emphasis coefficient.
    ///
    /// Defaults to the ten-vad reference value,
    /// [`PRE_EMPHASIS_COEFF`].
    #[config(default = "PRE_EMPHASIS_COEFF")]
    pub coeff: f32,
}

impl Default for PreEmphasisConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl PreEmphasisConfig {
    /// Initializes a zeroed [`PreEmphasisContext`] for `batch_size` streams.
    ///
    /// The carried sample starts at zero, so the first sample of the first
    /// hop passes through unfiltered.
    ///
    /// # Arguments
    /// * `batch_size`: the number of independent streams; must be non-zero.
    pub fn init<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> PreEmphasisContext<B> {
        assert_ne!(batch_size, 0, "PreEmphasis batch_size must be non-zero");
        PreEmphasisContext {
            coeff: self.coeff,
            prev: Tensor::zeros([batch_size], device),
        }
    }
}

/// Streaming pre-emphasis state.
///
/// Holds the coefficient and the `[batch]` last raw sample of each stream.
///
/// Built by [`PreEmphasisConfig::init`].
#[derive(Debug, Clone)]
pub struct PreEmphasisContext<B: Backend> {
    coeff: f32,

    /// The `[batch]` last raw input sample of each stream.
    ///
    /// Zero before the stream starts.
    pub prev: Tensor<B, 1>,
}

impl<B: Backend> PreEmphasisContext<B> {
    /// The pre-emphasis coefficient.
    pub fn coeff(&self) -> f32 {
        self.coeff
    }

    /// The batch size; each batch row is an independent stream.
    pub fn batch_size(&self) -> usize {
        self.prev.dims()[0]
    }

    /// Resets the carried sample to zero.
    pub fn reset(&mut self) {
        self.prev = Tensor::zeros_like(&self.prev);
    }

    /// Filters one hop, carrying the boundary sample forward.
    ///
    /// # Arguments
    /// * `hop`: `[batch, samples]` new raw samples, with `samples` non-zero.
    ///
    /// # Returns
    /// `[batch, samples]` pre-emphasized samples.
    pub fn forward(
        &mut self,
        hop: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        #[cfg(any(test, debug_assertions))]
        let [samples] = unpack_shape_contract!(
            ["batch", "samples"],
            &hop,
            &["samples"],
            &[("batch", self.batch_size())],
        );
        #[cfg(not(any(test, debug_assertions)))]
        let samples = hop.dims()[1];

        assert_ne!(samples, 0, "PreEmphasis hop must be non-empty");

        // `[batch, samples]`: the input delayed by one, with the carried
        // sample filling the first slot.
        // [batch, 1]
        let carry: Tensor<B, 2> = self.prev.extract().unsqueeze_dim(1);
        let delayed = if samples == 1 {
            carry
        } else {
            let head = hop.clone().slice_dim(1, ..(samples - 1) as isize);
            Tensor::cat(vec![carry, head], 1)
        };

        // Carry the last *raw* sample, not the filtered one.
        self.prev = hop.clone().select_dim::<1>(1, samples - 1);

        hop - delayed.mul_scalar(self.coeff)
    }

    /// Filters `steps` consecutive hops at once.
    ///
    /// Equivalent to `steps` calls of [`forward`](Self::forward): the hops are
    /// concatenated into one per-stream signal, filtered in a single pass, and
    /// split back apart. The filter is causal with a one-sample memory, so
    /// splitting the pass at hop boundaries changes nothing.
    ///
    /// # Arguments
    /// * `hops`: `[steps, batch, samples]` consecutive raw hops.
    ///
    /// # Returns
    /// `[steps, batch, samples]` pre-emphasized hops.
    pub fn forward_sequence(
        &mut self,
        hops: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        let [steps, samples] = unpack_shape_contract!(
            ["steps", "batch", "samples"],
            &hops,
            &["steps", "samples"],
            &[("batch", self.batch_size())],
        );
        #[cfg(not(any(test, debug_assertions)))]
        let [steps, _, samples] = hops.dims();

        assert_ne!(steps, 0, "PreEmphasis hops must be non-empty");

        let batch = self.batch_size();

        // [batch, steps * samples]
        let stream = hops.swap_dims(0, 1).flatten::<2>(1, 2);

        // [batch, steps * samples]
        let out = self.forward(stream);

        // [steps, batch, samples]
        out.reshape([batch, steps, samples]).swap_dims(0, 1)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
        backend::BackendTypes,
    };

    use super::*;
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;
    type F = <B as BackendTypes>::FloatElem;

    /// Host reference: the scalar filter, over one stream, with carry.
    fn host_pre_emphasis(
        signal: &[f32],
        coeff: f32,
        prev: &mut f32,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(signal.len());
        for (i, &x) in signal.iter().enumerate() {
            let back = if i == 0 { *prev } else { signal[i - 1] };
            out.push(x - coeff * back);
        }
        if let Some(&last) = signal.last() {
            *prev = last;
        }
        out
    }

    #[test]
    fn test_config_default() {
        let cfg = PreEmphasisConfig::new();
        assert_eq!(cfg.coeff, PRE_EMPHASIS_COEFF);
        assert_eq!(cfg.coeff, 0.97);

        let cfg = PreEmphasisConfig::new().with_coeff(0.5);
        assert_eq!(cfg.coeff, 0.5);
    }

    #[test]
    fn test_init_state() {
        let device = Default::default();
        let ctx: PreEmphasisContext<B> = PreEmphasisConfig::new().init(3, &device);

        assert_eq!(ctx.batch_size(), 3);
        assert_eq!(ctx.coeff(), PRE_EMPHASIS_COEFF);
        assert_eq!(ctx.prev.dims(), [3]);

        // The carried sample starts at zero.
        ctx.prev
            .to_data()
            .assert_eq(&Tensor::<B, 1>::zeros([3], &device).to_data(), true);
    }

    #[test]
    #[should_panic(expected = "batch_size must be non-zero")]
    fn test_init_rejects_zero_batch() {
        let device = Default::default();
        let _: PreEmphasisContext<B> = PreEmphasisConfig::new().init(0, &device);
    }

    #[test]
    fn test_forward_matches_host_reference() {
        let device = Default::default();
        let coeff = PRE_EMPHASIS_COEFF;
        let mut ctx: PreEmphasisContext<B> = PreEmphasisConfig::new().init(1, &device);

        // Two consecutive hops: the second must see the first's last sample.
        let hops = [vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0f32, -6.0, 7.0, -8.0]];

        let mut host_prev = 0.0f32;
        for hop in &hops {
            let expected = host_pre_emphasis(hop, coeff, &mut host_prev);

            let input =
                Tensor::<B, 2>::from_data(TensorData::new(hop.clone(), [1, hop.len()]), &device);
            let out = ctx.forward(input);

            out.to_data_as::<F>().assert_approx_eq::<F>(
                &TensorData::new(expected, [1, hop.len()]).convert::<F>(),
                Tolerance::default(),
            );
        }

        // The carried sample is the last *raw* input, not the filtered one.
        ctx.prev.to_data_as::<F>().assert_approx_eq::<F>(
            &TensorData::new(vec![-8.0f32], [1]).convert::<F>(),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_forward_single_sample_hop() {
        // A one-sample hop is entirely carry-driven; it exercises the branch
        // where there is no in-hop history at all.
        let device = Default::default();
        let mut ctx: PreEmphasisContext<B> =
            PreEmphasisConfig::new().with_coeff(0.5).init(1, &device);

        let first = Tensor::<B, 2>::from_data(TensorData::new(vec![4.0f32], [1, 1]), &device);
        let out = ctx.forward(first);
        // 4.0 - 0.5 * 0.0
        out.to_data_as::<F>().assert_approx_eq::<F>(
            &TensorData::new(vec![4.0f32], [1, 1]).convert::<F>(),
            Tolerance::default(),
        );

        let second = Tensor::<B, 2>::from_data(TensorData::new(vec![10.0f32], [1, 1]), &device);
        let out = ctx.forward(second);
        // 10.0 - 0.5 * 4.0
        out.to_data_as::<F>().assert_approx_eq::<F>(
            &TensorData::new(vec![8.0f32], [1, 1]).convert::<F>(),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_batch_rows_are_independent() {
        let device = Default::default();
        let coeff = PRE_EMPHASIS_COEFF;
        let batch = 3;
        let samples = 5;

        let rows: Vec<Vec<f32>> = (0..batch)
            .map(|b| (0..samples).map(|i| (b * 10 + i) as f32).collect())
            .collect();

        let mut ctx: PreEmphasisContext<B> = PreEmphasisConfig::new().init(batch, &device);
        let input =
            Tensor::<B, 2>::from_data(TensorData::new(rows.concat(), [batch, samples]), &device);
        let out = ctx.forward(input);

        let expected: Vec<f32> = rows
            .iter()
            .flat_map(|row| {
                let mut prev = 0.0f32;
                host_pre_emphasis(row, coeff, &mut prev)
            })
            .collect();

        out.to_data_as::<F>().assert_approx_eq::<F>(
            &TensorData::new(expected, [batch, samples]).convert::<F>(),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_forward_sequence_matches_stepwise() {
        let device = Default::default();
        let steps = 4;
        let batch = 2;
        let samples = 6;

        let hops = Tensor::<B, 3>::random([steps, batch, samples], Distribution::Default, &device);

        let mut seq_ctx: PreEmphasisContext<B> = PreEmphasisConfig::new().init(batch, &device);
        let mut step_ctx = seq_ctx.clone();

        let seq_out = seq_ctx.forward_sequence(hops.clone());
        assert_eq!(seq_out.dims(), [steps, batch, samples]);

        let mut step_outs = Vec::with_capacity(steps);
        for step in 0..steps {
            step_outs.push(step_ctx.forward(hops.clone().select_dim::<2>(0, step)));
        }
        let step_out: Tensor<B, 3> = Tensor::stack(step_outs, 0);

        let tol = Tolerance::<F>::default();
        seq_out
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_out.to_data_as::<F>(), tol);

        // The residual carry must agree too, or the next call diverges.
        seq_ctx
            .prev
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_ctx.prev.to_data_as::<F>(), tol);
    }

    #[test]
    fn test_reset() {
        let device = Default::default();
        let mut ctx: PreEmphasisContext<B> = PreEmphasisConfig::new().init(1, &device);

        let hop = Tensor::<B, 2>::from_data(TensorData::new(vec![3.0f32, 9.0], [1, 2]), &device);
        let first = ctx.forward(hop.clone());

        ctx.reset();
        ctx.prev
            .to_data()
            .assert_eq(&Tensor::<B, 1>::zeros([1], &device).to_data(), true);

        // After a reset the same hop reproduces the very first output.
        let again = ctx.forward(hop);
        again
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&first.to_data_as::<F>(), Tolerance::default());
    }
}
