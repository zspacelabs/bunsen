//! # ten-vad streaming driver.
//!
//! The mutable context that carries a ten-vad stream across calls, and the
//! `context_forward` family that drives raw audio through it.
//!
//! [`TenVad::forward`] is the stateless call through the network: it consumes
//! an already-widened `[batch, d_ctx, n_freq]` feature stack. Everything
//! needed to *build* that stack from audio — the pre-emphasis carry, the
//! sliding STFT queue, the pitch recurrence, and the rolling frame history —
//! lives in [`TenVadContext`], and
//! [`context_forward`](TenVad::context_forward) is what ties the two
//! together. This mirrors the split in
//! [`SileroVad`](crate::kits::speech::silero_vad::SileroVad).
//!
//! ## Mutable, not moved
//!
//! [`SileroVadContext`] is a burn `Module` moved in and out by value. This
//! context cannot be: it owns a [`SlidingStftContext`] and one
//! [`TenVadPitchSource`] per stream, neither of which is a tensor. It is a
//! plain struct driven through `&mut`, matching [`SlidingStftContext`]'s own
//! style.
//!
//! ## Batch size
//!
//! The stock ten-vad ONNX graph pins its LSTM batch to 1, so this driver is
//! built and tested at `batch_size = 1`. See [`TenVad::forward`] for what the
//! leading axis actually means and why multi-stream batching needs a patched
//! graph.
//!
//! [`SileroVadContext`]: crate::kits::speech::silero_vad::SileroVadContext

use burn::{
    config::Config,
    prelude::*,
};

use crate::{
    blocks::rnn::lstm::ExtLstmState,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::ten_vad::{
        TenVad,
        TenVadMeta,
        context::{
            coeff::{
                D_CTX,
                SAMPLE_RATE,
            },
            features::{
                TenVadFeatureConfig,
                TenVadFeatureContext,
                TenVadFeatureMeta,
            },
            pitch::{
                TenVadPitchEstimator,
                TenVadPitchSource,
            },
        },
    },
    ops::signal::SlidingStftContext,
    prelude::TensorOpExt,
};

/// Common meta for [`TenVadContextConfig`] and [`TenVadContext`].
pub trait TenVadContextMeta {
    /// The sample rate, in Hz, this context expects.
    fn sample_rate(&self) -> usize;

    /// The batch size; each batch row is an independent stream.
    fn batch_size(&self) -> usize;

    /// The hop size, in samples; one hop yields one probability.
    fn hop_size(&self) -> usize;

    /// The frame context depth carried into the model.
    fn d_ctx(&self) -> usize;

    /// The feature width of one frame.
    fn n_freq(&self) -> usize;
}

/// Config for [`TenVadContext`].
///
/// Defaults match the ten-vad reference driver: one 16 kHz stream, hop 256,
/// a 3-frame context stack.
///
/// Builds [`TenVadContext`] via [`TenVad::init_context`]. Implements
/// [`TenVadContextMeta`].
#[derive(Config, Debug)]
pub struct TenVadContextConfig {
    /// The sample rate, in Hz.
    #[config(default = "SAMPLE_RATE")]
    pub sample_rate: usize,

    /// The number of independent streams.
    ///
    /// The stock ONNX graph pins this to 1; see the module docs.
    #[config(default = "1")]
    pub batch_size: usize,

    /// The frame context depth.
    #[config(default = "D_CTX")]
    pub d_ctx: usize,

    /// The feature front-end geometry.
    #[config(default = "TenVadFeatureConfig::new()")]
    pub features: TenVadFeatureConfig,
}

impl TenVadContextMeta for TenVadContextConfig {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn hop_size(&self) -> usize {
        self.features.hop_size()
    }

    fn d_ctx(&self) -> usize {
        self.d_ctx
    }

    fn n_freq(&self) -> usize {
        self.features.n_freq()
    }
}

impl TenVadContextConfig {
    /// Validates the context geometry.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the batch size or context depth is zero, if
    /// the sample rate disagrees with the front end, or if the front-end
    /// geometry is itself invalid.
    pub fn validate(&self) -> BunsenResult<()> {
        self.features.validate()?;

        if self.batch_size == 0 {
            return Err(BunsenError::Invalid(
                "TenVadContext batch_size must be non-zero".to_string(),
            ));
        }
        if self.d_ctx == 0 {
            return Err(BunsenError::Invalid(
                "TenVadContext d_ctx must be non-zero".to_string(),
            ));
        }
        if self.sample_rate != self.features.sample_rate() {
            return Err(BunsenError::Invalid(format!(
                "TenVadContext sample_rate ({}) != feature sample_rate ({})",
                self.sample_rate,
                self.features.sample_rate(),
            )));
        }
        Ok(())
    }
}

/// The mutable driving context for a ten-vad stream.
///
/// Carries everything the model cannot: the audio front-end state
/// ([`TenVadFeatureContext`]), the rolling `[batch, d_ctx, n_freq]` feature
/// stack, and the two LSTM states.
///
/// Built by [`TenVad::init_context`]. Implements [`TenVadContextMeta`].
#[derive(Debug, Clone)]
pub struct TenVadContext<B: Backend, P: TenVadPitchSource = TenVadPitchEstimator> {
    /// The audio front-end streaming state.
    pub features: TenVadFeatureContext<B, P>,

    /// The `[batch, d_ctx, n_freq]` rolling feature stack.
    ///
    /// Row `d_ctx - 1` is the most recent frame; row `0` the oldest. This is
    /// exactly what [`TenVad::forward`] consumes.
    pub stack: Tensor<B, 3>,

    /// The `[batch, d_hidden]` first-LSTM state.
    pub state1: ExtLstmState<B, 2>,

    /// The `[batch, d_hidden]` second-LSTM state.
    pub state2: ExtLstmState<B, 2>,
}

impl<B: Backend, P: TenVadPitchSource> TenVadContextMeta for TenVadContext<B, P> {
    fn sample_rate(&self) -> usize {
        self.features.sample_rate()
    }

    fn batch_size(&self) -> usize {
        self.stack.dims()[0]
    }

    fn hop_size(&self) -> usize {
        self.features.hop_size()
    }

    fn d_ctx(&self) -> usize {
        self.stack.dims()[1]
    }

    fn n_freq(&self) -> usize {
        self.stack.dims()[2]
    }
}

impl<B: Backend, P: TenVadPitchSource> TenVadContext<B, P> {
    /// The recurrent hidden width.
    pub fn d_hidden(&self) -> usize {
        self.state1.hidden.dims()[1]
    }

    /// The sliding STFT queue, for inspection.
    pub fn stft(&self) -> &SlidingStftContext<B> {
        &self.features.stft
    }

    /// Resets the whole context to the start-of-stream condition.
    ///
    /// Zeroes the feature stack, both LSTM states, the STFT queue, the
    /// pre-emphasis carry, and every pitch source.
    pub fn reset(&mut self) {
        self.features.reset();
        self.stack = Tensor::zeros_like(&self.stack);
        self.state1 = ExtLstmState::initial(self.state1.hidden.dims(), &self.stack.device());
        self.state2 = ExtLstmState::initial(self.state2.hidden.dims(), &self.stack.device());
    }

    /// Rolls one feature frame into the context stack.
    ///
    /// Drops the oldest frame and appends `feats`, matching the reference's
    /// shift-left-and-write (`ALGO_TRACE.md` §3.7).
    ///
    /// This is the widened-context construction itself: the result is what
    /// [`TenVad::forward`] expects, so callers driving the model by hand (or
    /// cross-checking it against the ONNX graph) can use it directly.
    ///
    /// # Arguments
    /// * `feats`: `[batch, n_freq]` the newest feature frame.
    ///
    /// # Returns
    /// The updated `[batch, d_ctx, n_freq]` stack.
    pub fn push_features(
        &mut self,
        feats: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "n_freq"],
            &feats,
            &[("batch", self.batch_size()), ("n_freq", self.n_freq())],
        );

        // [batch, 1, n_freq]
        let newest: Tensor<B, 3> = feats.unsqueeze_dim(1);

        // Inplace, so the old stack's storage can be reused.
        let older = self.stack.extract().slice_dim(1, 1..);
        self.stack = Tensor::cat(vec![older, newest], 1);

        self.stack.clone()
    }
}

impl<B: Backend> TenVad<B> {
    /// Builds a zeroed driving context with the reference pitch estimator.
    ///
    /// This is the faithful front end: all 41 features match the reference
    /// implementation. Feature `40` is a host-side recurrence, so every frame
    /// synchronizes the raw hop and the bin powers back from the device.
    ///
    /// To trade that fidelity for an entirely on-device sequence path, pass
    /// [`ZeroPitch`](super::ZeroPitch) to
    /// [`init_context_with`](Self::init_context_with); it pins feature `40` to
    /// a constant and leaves the other 40 exact.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the config is invalid, or if its context
    /// depth or feature width disagrees with this model.
    pub fn init_context(
        &self,
        cfg: &TenVadContextConfig,
        device: &B::Device,
    ) -> BunsenResult<TenVadContext<B, TenVadPitchEstimator>> {
        self.init_context_with(cfg, TenVadPitchEstimator::new(), device)
    }

    /// Builds a zeroed driving context over a specific pitch source.
    ///
    /// # Arguments
    /// * `cfg`: the context geometry.
    /// * `pitch`: the prototype pitch source, cloned once per stream.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the config is invalid, or if its context
    /// depth or feature width disagrees with this model.
    pub fn init_context_with<P: TenVadPitchSource + Clone>(
        &self,
        cfg: &TenVadContextConfig,
        pitch: P,
        device: &B::Device,
    ) -> BunsenResult<TenVadContext<B, P>> {
        cfg.validate()?;

        if cfg.d_ctx() != self.d_ctx() {
            return Err(BunsenError::Invalid(format!(
                "TenVadContext d_ctx ({}) != model d_ctx ({})",
                cfg.d_ctx(),
                self.d_ctx(),
            )));
        }
        if cfg.n_freq() != self.n_freq() {
            return Err(BunsenError::Invalid(format!(
                "TenVadContext n_freq ({}) != model n_freq ({})",
                cfg.n_freq(),
                self.n_freq(),
            )));
        }

        let batch = cfg.batch_size();
        let state_shape = [batch, self.d_hidden()];

        Ok(TenVadContext {
            features: cfg.features.try_init_context(batch, pitch, device)?,
            stack: Tensor::zeros([batch, cfg.d_ctx(), cfg.n_freq()], device),
            state1: ExtLstmState::initial(state_shape, device),
            state2: ExtLstmState::initial(state_shape, device),
        })
    }

    /// Drives one hop of audio through the front end and the model.
    ///
    /// Extracts the frame's features, rolls them into the context stack, and
    /// runs one [`forward`](Self::forward) with the carried LSTM states. The
    /// context is advanced in place.
    ///
    /// # Arguments
    /// * `hop`: `[batch, hop_size]` mono audio in `[-1, 1]`, at this model's
    ///   sample rate.
    /// * `ctx`: the driving context, advanced in place.
    ///
    /// # Returns
    /// `[batch]` speech probabilities in `[0, 1]`.
    pub fn context_forward<P: TenVadPitchSource>(
        &self,
        hop: Tensor<B, 2>,
        ctx: &mut TenVadContext<B, P>,
    ) -> Tensor<B, 1> {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "hop_size"],
            &hop,
            &[("batch", ctx.batch_size()), ("hop_size", ctx.hop_size()),],
        );

        // [batch, n_freq]
        let feats = ctx.features.forward(hop);

        // [batch, d_ctx, n_freq]
        let stack = ctx.push_features(feats);

        let (probs, state1, state2) =
            self.forward(stack, Some(ctx.state1.clone()), Some(ctx.state2.clone()));
        ctx.state1 = state1;
        ctx.state2 = state2;

        // [batch, 1] -> [batch]
        probs.squeeze_dim(1)
    }

    /// Drives `steps` consecutive hops through the front end and the model.
    ///
    /// Equivalent to `steps` calls of
    /// [`context_forward`](Self::context_forward), but the entire front end
    /// runs batched across the sequence: one pre-emphasis pass, one `stft`
    /// call, one filterbank matmul, and every step's context stack
    /// materialized in a single concatenation. Only the recurrence itself is
    /// stepped.
    ///
    /// # Arguments
    /// * `hop_seq`: `[steps, batch, hop_size]` consecutive mono audio hops in
    ///   `[-1, 1]`, with `steps` non-zero.
    /// * `ctx`: the driving context, advanced in place.
    ///
    /// # Returns
    /// `[steps, batch]` speech probabilities in `[0, 1]`.
    pub fn context_forward_sequence<P: TenVadPitchSource>(
        &self,
        hop_seq: Tensor<B, 3>,
        ctx: &mut TenVadContext<B, P>,
    ) -> Tensor<B, 2> {
        #[cfg(any(test, debug_assertions))]
        let [steps] = crate::contracts::unpack_shape_contract!(
            ["steps", "batch", "hop_size"],
            &hop_seq,
            &["steps"],
            &[("batch", ctx.batch_size()), ("hop_size", ctx.hop_size()),],
        );
        #[cfg(not(any(test, debug_assertions)))]
        let steps = hop_seq.dims()[0];

        assert_ne!(steps, 0, "TenVad hop_seq must be non-empty");

        let d_ctx = ctx.d_ctx();

        // [steps, batch, n_freq]
        let feats = ctx.features.forward_sequence(hop_seq);

        // The carried stack followed by every new frame, so that step `s`'s
        // widened context is a slice of one tensor.
        // [batch, d_ctx + steps, n_freq]
        let history = Tensor::cat(vec![ctx.stack.extract(), feats.swap_dims(0, 1)], 1);

        // The carry-out stack is the final `d_ctx` frames.
        ctx.stack = history.clone().slice_dim(1, steps as isize..);

        let mut probs = Vec::with_capacity(steps);
        for step in 0..steps {
            // Step `s` sees frames `s + 1 ..= s + d_ctx` of the history.
            // [batch, d_ctx, n_freq]
            let stack = history
                .clone()
                .slice_dim(1, (step + 1) as isize..(step + 1 + d_ctx) as isize);

            let (prob, state1, state2) =
                self.forward(stack, Some(ctx.state1.clone()), Some(ctx.state2.clone()));
            ctx.state1 = state1;
            ctx.state2 = state2;

            probs.push(prob);
        }

        // [steps, batch, 1] -> [steps, batch]
        Tensor::stack::<3>(probs, 0).squeeze_dim(2)
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
        burner::module::ModuleInit,
        kits::speech::ten_vad::{
            TenVadStructureConfig,
            context::coeff::N_FREQ,
        },
        prelude::*,
        support::testing::CpuBackend,
    };

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    fn model() -> (TenVad<B>, burn::backend::flex::FlexDevice) {
        let device = Default::default();
        let vad: TenVad<B> = TenVadStructureConfig::default().init(&device);
        (vad, device)
    }

    #[test]
    fn test_config_meta() {
        let cfg = TenVadContextConfig::new();

        assert_eq!(cfg.sample_rate(), 16000);
        assert_eq!(cfg.batch_size(), 1);
        assert_eq!(cfg.hop_size(), 256);
        assert_eq!(cfg.d_ctx(), 3);
        assert_eq!(cfg.n_freq(), N_FREQ);

        cfg.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        for bad in [
            TenVadContextConfig::new().with_batch_size(0),
            TenVadContextConfig::new().with_d_ctx(0),
            // The declared rate must agree with the front end's.
            TenVadContextConfig::new().with_sample_rate(8000),
        ] {
            assert!(
                matches!(bad.validate(), Err(BunsenError::Invalid(_))),
                "expected Invalid: {bad:?}",
            );
        }
    }

    #[test]
    fn test_init_context_shapes_and_zeroing() {
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();
        let ctx = vad.init_context(&cfg, &device).unwrap();

        assert_eq!(ctx.sample_rate(), 16000);
        assert_eq!(ctx.batch_size(), 1);
        assert_eq!(ctx.hop_size(), 256);
        assert_eq!(ctx.d_ctx(), vad.d_ctx());
        assert_eq!(ctx.n_freq(), vad.n_freq());
        assert_eq!(ctx.d_hidden(), vad.d_hidden());

        assert_eq!(ctx.stack.dims(), [1, 3, 41]);
        assert_eq!(ctx.state1.hidden.dims(), [1, 64]);
        assert_eq!(ctx.state2.cell.dims(), [1, 64]);

        // Everything starts at zero, as in the reference driver.
        let zeros = Tensor::<B, 3>::zeros([1, 3, 41], &device);
        ctx.stack.to_data().assert_eq(&zeros.to_data(), true);

        let zeros2 = Tensor::<B, 2>::zeros([1, 64], &device);
        ctx.state1
            .hidden
            .to_data()
            .assert_eq(&zeros2.to_data(), true);
        ctx.state1.cell.to_data().assert_eq(&zeros2.to_data(), true);
        ctx.state2
            .hidden
            .to_data()
            .assert_eq(&zeros2.to_data(), true);
        ctx.state2.cell.to_data().assert_eq(&zeros2.to_data(), true);

        // The STFT queue starts zeroed too.
        assert_eq!(ctx.stft().queue.dims(), [1, 768]);
    }

    #[test]
    fn test_init_context_rejects_model_mismatch() {
        let (vad, device) = model();

        // A context depth the model does not consume.
        let bad = TenVadContextConfig::new().with_d_ctx(5);
        assert!(matches!(
            vad.init_context(&bad, &device),
            Err(BunsenError::Invalid(_)),
        ));

        // A feature width the model does not consume. The front end accepts
        // it (it is under the table bound); the driver must not.
        let bad = TenVadContextConfig::new().with_features(
            TenVadFeatureConfig::new().with_mel(
                crate::kits::speech::ten_vad::context::mel::TenVadMelConfig::new()
                    .with_n_mels(6)
                    .with_fft_size(1024),
            ),
        );
        bad.validate().unwrap();
        assert!(matches!(
            vad.init_context(&bad, &device),
            Err(BunsenError::Invalid(_)),
        ));
    }

    #[test]
    fn test_push_features_rolls_the_stack() {
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();
        let mut ctx = vad.init_context(&cfg, &device).unwrap();

        let d_ctx = ctx.d_ctx();
        let n_freq = ctx.n_freq();

        // Push distinguishable frames: frame `f` is all-`f`.
        let mut pushed: Vec<f32> = Vec::new();
        for f in 1..=(d_ctx + 2) {
            let value = f as f32;
            pushed.push(value);

            let feats = Tensor::<B, 2>::full([1, n_freq], value, &device);
            let stack = ctx.push_features(feats);
            assert_eq!(stack.dims(), [1, d_ctx, n_freq]);

            let host: Vec<f32> = stack.to_data_as::<f32>().to_vec_as::<f32>().unwrap();

            // The stack holds the last `d_ctx` frames, oldest first; slots
            // not yet filled are still zero.
            for row in 0..d_ctx {
                let age = d_ctx - 1 - row;
                let expected = if age < pushed.len() {
                    pushed[pushed.len() - 1 - age]
                } else {
                    0.0
                };
                for col in 0..n_freq {
                    assert_eq!(
                        host[row * n_freq + col],
                        expected,
                        "after {f} pushes, row {row} col {col}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_context_forward_shapes_and_range() {
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();
        let mut ctx = vad.init_context(&cfg, &device).unwrap();

        let hop = Tensor::<B, 2>::random([1, cfg.hop_size()], Distribution::Default, &device);
        let probs = vad.context_forward(hop, &mut ctx);

        assert_eq!(probs.dims(), [1]);

        let host: Vec<f32> = probs.to_data_as::<f32>().to_vec_as::<f32>().unwrap();
        assert!(host.iter().all(|&p| (0.0..=1.0).contains(&p)), "{host:?}");
    }

    #[test]
    fn test_context_forward_sequence_shapes_and_range() {
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();
        let mut ctx = vad.init_context(&cfg, &device).unwrap();

        let steps = 7;
        let hops =
            Tensor::<B, 3>::random([steps, 1, cfg.hop_size()], Distribution::Default, &device);
        let probs = vad.context_forward_sequence(hops, &mut ctx);

        assert_eq!(probs.dims(), [steps, 1]);

        let host: Vec<f32> = probs.to_data_as::<f32>().to_vec_as::<f32>().unwrap();
        assert!(host.iter().all(|&p| (0.0..=1.0).contains(&p)), "{host:?}");
    }

    #[test]
    fn test_sequence_matches_stepwise() {
        // The whole point of the sequence form: same answer, fewer passes.
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();

        let steps = 9;
        let hops =
            Tensor::<B, 3>::random([steps, 1, cfg.hop_size()], Distribution::Default, &device);

        let mut seq_ctx = vad.init_context(&cfg, &device).unwrap();
        let seq_probs = vad.context_forward_sequence(hops.clone(), &mut seq_ctx);

        let mut step_ctx = vad.init_context(&cfg, &device).unwrap();
        let mut step_probs = Vec::with_capacity(steps);
        for step in 0..steps {
            let hop = hops.clone().select_dim::<2>(0, step);
            step_probs.push(vad.context_forward(hop, &mut step_ctx));
        }
        let step_probs: Tensor<B, 2> = Tensor::stack(step_probs, 0);

        let tol = Tolerance::<F>::permissive();
        seq_probs
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_probs.to_data_as::<F>(), tol);

        // Every carried field must agree, or the next call diverges.
        seq_ctx
            .stack
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_ctx.stack.to_data_as::<F>(), tol);
        seq_ctx
            .state1
            .hidden
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_ctx.state1.hidden.to_data_as::<F>(), tol);
        seq_ctx
            .state1
            .cell
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_ctx.state1.cell.to_data_as::<F>(), tol);
        seq_ctx
            .state2
            .hidden
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_ctx.state2.hidden.to_data_as::<F>(), tol);
        seq_ctx
            .state2
            .cell
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_ctx.state2.cell.to_data_as::<F>(), tol);
        seq_ctx
            .features
            .stft
            .queue
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_ctx.features.stft.queue.to_data_as::<F>(), tol);
    }

    #[test]
    fn test_single_step_sequence_matches_context_forward() {
        // The `steps == 1` boundary: the sequence path's history slicing must
        // degenerate to a plain roll-and-run.
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();

        let hop = Tensor::<B, 2>::random([1, cfg.hop_size()], Distribution::Default, &device);

        let mut seq_ctx = vad.init_context(&cfg, &device).unwrap();
        let seq_probs =
            vad.context_forward_sequence(hop.clone().unsqueeze_dim::<3>(0), &mut seq_ctx);

        let mut step_ctx = vad.init_context(&cfg, &device).unwrap();
        let step_probs = vad.context_forward(hop, &mut step_ctx);

        let tol = Tolerance::<F>::permissive();
        seq_probs
            .squeeze_dim::<1>(0)
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_probs.to_data_as::<F>(), tol);
        seq_ctx
            .stack
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_ctx.stack.to_data_as::<F>(), tol);
    }

    #[test]
    fn test_sequence_resumes_across_chunks() {
        // Splitting a stream into two sequence calls must match one call over
        // the whole thing -- that is what makes the context a *continuation*.
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();

        let steps = 8;
        let hops =
            Tensor::<B, 3>::random([steps, 1, cfg.hop_size()], Distribution::Default, &device);

        let mut whole_ctx = vad.init_context(&cfg, &device).unwrap();
        let whole = vad.context_forward_sequence(hops.clone(), &mut whole_ctx);

        let mut split_ctx = vad.init_context(&cfg, &device).unwrap();
        let head = hops.clone().slice_dim(0, ..3);
        let tail = hops.slice_dim(0, 3..);
        let a = vad.context_forward_sequence(head, &mut split_ctx);
        let b = vad.context_forward_sequence(tail, &mut split_ctx);
        let split = Tensor::cat(vec![a, b], 0);

        let tol = Tolerance::<F>::permissive();
        whole
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&split.to_data_as::<F>(), tol);
        whole_ctx
            .stack
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&split_ctx.stack.to_data_as::<F>(), tol);
        whole_ctx
            .state2
            .hidden
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&split_ctx.state2.hidden.to_data_as::<F>(), tol);
    }

    #[test]
    fn test_reset_rewinds_the_stream() {
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();
        let mut ctx = vad.init_context(&cfg, &device).unwrap();

        let hop = Tensor::<B, 2>::random([1, cfg.hop_size()], Distribution::Default, &device);

        let first = vad.context_forward(hop.clone(), &mut ctx);

        vad.context_forward(hop.clone(), &mut ctx);
        vad.context_forward(hop.clone(), &mut ctx);
        ctx.reset();

        let again = vad.context_forward(hop, &mut ctx);
        again
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&first.to_data_as::<F>(), Tolerance::permissive());
    }

    #[test]
    #[should_panic(expected = "hop_seq must be non-empty")]
    fn test_sequence_rejects_empty_input() {
        let (vad, device) = model();
        let cfg = TenVadContextConfig::new();
        let mut ctx = vad.init_context(&cfg, &device).unwrap();

        let empty = Tensor::<B, 3>::zeros([0, 1, cfg.hop_size()], &device);
        vad.context_forward_sequence(empty, &mut ctx);
    }
}
