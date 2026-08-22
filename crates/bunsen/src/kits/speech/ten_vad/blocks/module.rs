//! # ten-vad model.
//!
//! [ten-vad][t] is a small, streaming voice-activity-detection model: given a
//! short stack of consecutive feature frames and the previous recurrent
//! states, it emits a per-frame speech probability and the next states.
//!
//! [t]: https://github.com/TEN-framework/ten-vad
//!
//! The pipeline is:
//!
//! 1. a `[1, 3, 41]` feature stack through a 2D conv stem ([`ConvSeq2d`] +
//!    [`MaxPool2d`] + a depthwise/pointwise [`ConvSeq2d`]), producing an
//!    `[f_ctx, d_features]` embedding,
//! 2. two stacked single-step [`Lstm`] blocks, whose outputs are concatenated,
//! 3. a two-layer `ReLU` / sigmoid [`Linear`] head producing the speech
//!    probability.
//!
//! [`TenVad::forward`] is the stateless call through the network. It starts at
//! an already-widened feature stack; everything that turns audio into those
//! 41 bins — pre-emphasis, the sliding STFT, the mel filterbank, the
//! normalization, and the rolling frame history — lives in
//! [`context`](crate::kits::speech::ten_vad::context), and is driven through
//! [`TenVadContext`] by
//! [`context_forward`](TenVad::context_forward) /
//! [`context_forward_sequence`](TenVad::context_forward_sequence).
//!
//! [`TenVadContext`]: crate::kits::speech::ten_vad::TenVadContext

use burn::{
    nn::{
        Linear,
        LinearConfig,
        Lstm,
        LstmConfig,
        PaddingConfig2d,
        activation::ActivationConfig,
        conv::Conv2dConfig,
        pool::{
            MaxPool2d,
            MaxPool2dConfig,
        },
    },
    prelude::*,
    tensor::activation::{
        relu,
        sigmoid,
    },
};
#[cfg(feature = "store")]
use burn_store::{
    BurnpackStore,
    ModuleSnapshot,
};

use crate::{
    blocks::{
        conv::{
            ConvBlock2dConfig,
            ConvBlock2dMeta,
            ConvSeq2d,
            ConvSeq2dConfig,
        },
        rnn::lstm::ExtLstmState,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`TenVad`].
pub trait TenVadMeta {
    /// The context length.
    fn d_ctx(&self) -> usize {
        3
    }

    /// The internal feature context length.
    fn f_ctx(&self) -> usize {
        2 * self.d_ctx() - 1
    }

    /// The number of frequency bins.
    fn n_freq(&self) -> usize {
        41
    }

    /// The dimension of the feature space.
    fn d_features(&self) -> usize;

    /// The dimension of the embedding space.
    fn d_hidden(&self) -> usize;
}

/// Config for [`TenVad`].
///
/// Builds [`TenVad`].
#[derive(Config, Debug)]
pub struct TenVadStructureConfig {
    /// Embedding first `ConvSeq2d` block.
    pub cs1: ConvSeq2dConfig,

    /// Embedding `MaxPool2d` block.
    pub pool: MaxPool2dConfig,

    /// Embedding second `ConvSeq2d` block.
    pub cs2: ConvSeq2dConfig,

    /// The first Lstm block.
    pub lstm1: LstmConfig,

    /// The second Lstm block.
    pub lstm2: LstmConfig,

    /// Projection first `Linear` block.
    pub linear1: LinearConfig,

    /// Projection second `Linear` block.
    pub linear2: LinearConfig,
}

impl Default for TenVadStructureConfig {
    fn default() -> Self {
        let d_hidden = 64;
        Self {
            cs1: ConvSeq2dConfig {
                blocks: vec![
                    ConvBlock2dConfig::new(Conv2dConfig::new([1, 1], [3, 3]).with_bias(false))
                        .with_act(None),
                    ConvBlock2dConfig::new(Conv2dConfig::new([1, 16], [1, 1]))
                        .with_act(Some(ActivationConfig::Relu)),
                ],
            },
            pool: MaxPool2dConfig::new([1, 3]).with_strides([1, 2]),
            cs2: ConvSeq2dConfig {
                blocks: vec![
                    ConvBlock2dConfig::new(
                        Conv2dConfig::new([16, 16], [1, 3])
                            .with_stride([2, 2])
                            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
                            .with_groups(16)
                            .with_bias(false),
                    )
                    .with_act(None),
                    ConvBlock2dConfig::new(Conv2dConfig::new([16, 16], [1, 1]))
                        .with_act(Some(ActivationConfig::Relu)),
                    ConvBlock2dConfig::new(
                        Conv2dConfig::new([16, 16], [1, 3])
                            .with_stride([2, 2])
                            .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 1))
                            .with_groups(16)
                            .with_bias(false),
                    )
                    .with_act(None),
                    ConvBlock2dConfig::new(Conv2dConfig::new([16, 16], [1, 1]))
                        .with_act(Some(ActivationConfig::Relu)),
                ],
            },
            lstm1: LstmConfig::new(80, d_hidden, true)
                .with_batch_first(false)
                .with_input_forget(false),
            lstm2: LstmConfig::new(d_hidden, d_hidden, true)
                .with_batch_first(false)
                .with_input_forget(false),
            linear1: LinearConfig::new(2 * d_hidden, d_hidden / 2).with_bias(true),
            linear2: LinearConfig::new(d_hidden / 2, 1).with_bias(true),
        }
    }
}

impl<B: Backend> ModuleInit<B, TenVad<B>> for TenVadStructureConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<TenVad<B>> {
        Ok(TenVad {
            cs1: self.cs1.try_init(device)?,
            pool: self.pool.init(),
            cs2: self.cs2.try_init(device)?,
            lstm1: self.lstm1.init(device),
            lstm2: self.lstm2.init(device),
            linear1: self.linear1.init(device),
            linear2: self.linear2.init(device),
        })
    }
}

/// ten-vad module.
///
/// Built by [`TenVadStructureConfig`].
#[derive(Module, Debug)]
pub struct TenVad<B: Backend> {
    /// Embedding first `ConvSeq2d` block.
    pub cs1: ConvSeq2d<B>,

    /// Embedding `MaxPool2d` block.
    pub pool: MaxPool2d,

    /// Embedding second `ConvSeq2d` block.
    pub cs2: ConvSeq2d<B>,

    /// The first Lstm block.
    pub lstm1: Lstm<B>,

    /// The second Lstm block.
    pub lstm2: Lstm<B>,

    /// Projection first `Linear` block.
    pub linear1: Linear<B>,

    /// Projection second `Linear` block.
    pub linear2: Linear<B>,
}

impl<B: Backend> TenVadMeta for TenVad<B> {
    fn d_features(&self) -> usize {
        self.cs1.blocks.last().unwrap().out_channels()
    }

    fn d_hidden(&self) -> usize {
        self.lstm1.d_hidden
    }
}

#[cfg(feature = "store")]
impl<B: Backend> TenVad<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(
        file: P,
        device: &B::Device,
    ) -> Self {
        let mut model = TenVadStructureConfig::default().try_init(device).unwrap();
        let mut store = BurnpackStore::from_file(file);
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(
        bytes: burn::tensor::Bytes,
        device: &B::Device,
    ) -> Self {
        let mut model = TenVadStructureConfig::default().try_init(device).unwrap();
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> TenVad<B> {
    /// Stateless forward pass over one feature stack.
    ///
    /// This is the model half only: `input` must already be the widened,
    /// normalized context stack. See
    /// [`context_forward`](Self::context_forward) to drive raw audio.
    ///
    /// # The leading axis is time, not stream-batch
    ///
    /// In the reference ONNX graph the leading dimension of the feature input
    /// lands on the LSTM's *sequence* axis, with the LSTM batch fixed at 1 by a
    /// graph constant (`new_shape__177 = [-1, 1, 80]`); `ALGO_TRACE.md` §8
    /// documents this and verifies it empirically. This method pins it to `1`
    /// and is the single-step form;
    /// [`forward_sequence`](Self::forward_sequence) is the same computation
    /// over a run of steps.
    ///
    /// **Multi-stream batching is structurally impossible** against the stock
    /// graph: batched states fail shape validation outright. It requires
    /// patching two reshape constants (§8.3), i.e. a different model file.
    ///
    /// # Arguments
    /// * `input`: `[1, d_ctx, n_freq]` the widened feature stack.
    /// * `state1`: `[1, d_hidden]` first-LSTM state, or `None` to start zeroed.
    /// * `state2`: `[1, d_hidden]` second-LSTM state, or `None` to start
    ///   zeroed.
    ///
    /// # Returns
    /// `(probabilities, state1, state2)`, with:
    /// * `probabilities`: `[1, 1]` speech probability in `[0, 1]`
    /// * `state1` / `state2`: `[1, d_hidden]` next LSTM states
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state1: Option<ExtLstmState<B, 2>>,
        state2: Option<ExtLstmState<B, 2>>,
    ) -> (Tensor<B, 2>, ExtLstmState<B, 2>, ExtLstmState<B, 2>) {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["steps", "d_ctx", "n_freq"],
            &input,
            &[
                ("steps", 1),
                ("d_ctx", self.d_ctx()),
                ("n_freq", self.n_freq())
            ]
        );

        self.forward_sequence(input, state1, state2)
    }

    /// Stateless forward pass over a run of consecutive feature stacks.
    ///
    /// Equivalent to `steps` calls of [`forward`](Self::forward), final states
    /// included — the reference verified exactly that against its own graph
    /// (`ALGO_TRACE.md` §8.2) — but it runs as **one** pass instead of `steps`.
    ///
    /// Only the recurrence is inherently sequential, and it is the one part
    /// that does not become a Rust-side loop:
    ///
    /// | stage | over a run of `steps` |
    /// |---|---|
    /// | [`frame_features`](Self::frame_features) | one conv pass, `steps` images |
    /// | `lstm_step` | one call; burn walks the recurrence internally |
    /// | `output_head` | one pass, `steps` rows |
    ///
    /// So a `steps`-hop run costs a constant number of dispatches rather than
    /// a number proportional to `steps`. The reference measured ~46x on CPU at
    /// `steps = 1875` (1756 ms sequential vs 38 ms batched).
    ///
    /// # The periodic reset
    ///
    /// This threads one unbroken recurrence through the whole input, so a
    /// caller reproducing the reference's periodic state reset must chunk at
    /// reset boundaries and zero the states between chunks — §8.2's own usage
    /// note. [`context_forward_sequence`](Self::context_forward_sequence) does
    /// that for you.
    ///
    /// # Arguments
    /// * `input`: `[steps, d_ctx, n_freq]` consecutive widened feature stacks,
    ///   with `steps` non-zero.
    /// * `state1`: `[1, d_hidden]` first-LSTM state, or `None` to start zeroed.
    /// * `state2`: `[1, d_hidden]` second-LSTM state, or `None` to start
    ///   zeroed.
    ///
    /// # Returns
    /// `(probabilities, state1, state2)`, with:
    /// * `probabilities`: `[steps, 1]` speech probabilities in `[0, 1]`, in
    ///   order
    /// * `state1` / `state2`: `[1, d_hidden]` states after the final step
    pub fn forward_sequence(
        &self,
        input: Tensor<B, 3>,
        state1: Option<ExtLstmState<B, 2>>,
        state2: Option<ExtLstmState<B, 2>>,
    ) -> (Tensor<B, 2>, ExtLstmState<B, 2>, ExtLstmState<B, 2>) {
        assert_eq!(state1.is_some(), state2.is_some());
        assert_ne!(input.dims()[0], 0, "TenVad input must be non-empty");

        // [steps, f_ctx, d_features]
        let x = self.frame_features(input);

        // [1, steps, 2 * d_hidden]
        let (x, state1, state2) = self.lstm_step(x, state1, state2);

        // [steps, 1]
        let x = self.output_head(x);

        (x, state1, state2)
    }

    /// Runs the conv stem over a feature stack.
    ///
    /// Purely per-step: each stack is convolved on its own, so this runs once
    /// over a whole sequence rather than once per step. That is the other half
    /// of what makes [`forward_sequence`](Self::forward_sequence) worth having.
    ///
    /// # Arguments
    /// * `x`: `[steps, d_ctx, n_freq]` the widened feature stacks.
    ///
    /// # Returns
    /// `[steps, f_ctx, d_features]` embeddings.
    ///
    /// # The `[-1, 1, d_ctx, n_freq]` reshape
    ///
    /// This matches the reference graph, which reshapes `input_1` per item so
    /// that the leading dimension is a true conv batch (`ALGO_TRACE.md` §8.1).
    /// An earlier form here used `[1, -1, d_ctx, n_freq]`, which puts the
    /// leading axis on the *channel* dimension instead — the two agree only at
    /// `steps == 1`, and above it the stem would not even typecheck against
    /// `cs1`'s single input channel.
    pub fn frame_features(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        #[cfg(any(test, debug_assertions))]
        let [steps] = crate::contracts::unpack_shape_contract!(
            ["steps", "d_ctx", "n_freq"],
            &x,
            &["steps"],
            &[("d_ctx", self.d_ctx()), ("n_freq", self.n_freq())]
        );

        // [steps, 1, d_ctx, n_freq]: one single-channel image per step.
        let x = x.reshape([-1, 1, self.d_ctx() as isize, self.n_freq() as isize]);
        let x = self.cs1.forward(x);
        let x = self.pool.forward(x);
        let x = self.cs2.forward(x);

        // The stem collapses the context axis to 1.
        let x = x.squeeze_dim(2);
        let x = x.permute([0, 2, 1]);

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["steps", "f_ctx", "d_features"],
            &x,
            &[
                ("steps", steps),
                ("f_ctx", self.f_ctx()),
                ("d_features", self.d_features())
            ]
        );
        x
    }

    /// Runs both LSTMs over a whole sequence, threading the states through.
    ///
    /// This is the model's only recurrence, and it is the reason the reference
    /// graph's leading axis behaves the way it does: `new_shape__177 =
    /// [-1, 1, 80]` lands it on the LSTM's *sequence* axis with the LSTM batch
    /// pinned to 1 (`ALGO_TRACE.md` §8.1). Both [`Lstm`]s here are configured
    /// `batch_first(false)`, so `[steps, 1, ..]` is that same layout, and burn
    /// walks the recurrence inside one call — `steps` sequential steps of the
    /// cell, not `steps` dispatches from Rust.
    ///
    /// The reference verified this equals `steps` separate batch-1 calls,
    /// final states included (`ALGO_TRACE.md` §8.2).
    ///
    /// # Arguments
    /// * `x`: `[steps, f_ctx, d_features]` per-step embeddings.
    /// * `state1` / `state2`: `[1, d_hidden]` states, or `None` to start
    ///   zeroed. These stay batch-1 whatever `steps` is — they are the state of
    ///   *one* stream, before and after the run.
    ///
    /// # Returns
    /// `([1, steps, 2 * d_hidden]` concatenated outputs, next `state1`, next
    /// `state2)`.
    fn lstm_step(
        &self,
        x: Tensor<B, 3>,
        state1: Option<ExtLstmState<B, 2>>,
        state2: Option<ExtLstmState<B, 2>>,
    ) -> (Tensor<B, 3>, ExtLstmState<B, 2>, ExtLstmState<B, 2>) {
        assert_eq!(state1.is_some(), state2.is_some());
        #[cfg(any(test, debug_assertions))]
        let steps = {
            use crate::contracts::assert_shape_contract;
            let [steps] = crate::contracts::unpack_shape_contract!(
                ["steps", "f_ctx", "d_features"],
                &x,
                &["steps"],
                &[("f_ctx", self.f_ctx()), ("d_features", self.d_features())]
            );
            for state in [&state1, &state2].into_iter().flatten() {
                assert_shape_contract!(
                    ["batch", "d_hidden"],
                    state.shape(),
                    &[("batch", 1), ("d_hidden", self.d_hidden())],
                );
            }
            steps
        };

        // [steps, 1, f_ctx * d_features]: the reference's `new_shape__177`,
        // which is `[seq, batch, input]` for a `batch_first(false)` LSTM.
        let x = x.reshape([-1, 1, (self.f_ctx() * self.d_features()) as isize]);
        let (x, state1) = self.lstm1.forward(x, state1.map(Into::into));

        // [1, steps, d_hidden]: `new_shape__176`.
        let y = x.reshape([1, -1, self.d_hidden() as isize]);

        // [steps, 1, d_hidden]: the graph's `[1, 0, 2]` transpose, putting the
        // second LSTM back on the same sequence-major layout.
        let x = y.clone().swap_dims(0, 1);
        let (x, state2) = self.lstm2.forward(x, state2.map(Into::into));
        let x = x.swap_dims(0, 1);

        // [1, steps, 2 * d_hidden]
        let x = Tensor::cat([x, y].into(), 2);
        let state1: ExtLstmState<B, 2> = state1.into();
        let state2: ExtLstmState<B, 2> = state2.into();
        #[cfg(any(test, debug_assertions))]
        {
            use crate::contracts::assert_shape_contract;
            assert_shape_contract!(
                [1, "steps", 2 * "d_hidden"],
                &x,
                &[("steps", steps), ("d_hidden", self.d_hidden())]
            );
            for state in [&state1, &state2] {
                assert_shape_contract!(
                    ["batch", "d_hidden"],
                    &state.shape(),
                    &[("batch", 1), ("d_hidden", self.d_hidden())],
                );
            }
        }
        (x, state1, state2)
    }

    /// Projects the recurrent output to a probability per step.
    ///
    /// Purely per-step: every row of `x` goes through the same two linears, so
    /// this runs once over a whole sequence rather than once per step. That is
    /// half of what makes [`forward_sequence`](Self::forward_sequence) worth
    /// having.
    ///
    /// # Arguments
    /// * `x`: `[1, steps, 2 * d_hidden]` the concatenated LSTM outputs.
    ///
    /// # Returns
    /// `[steps, 1]` speech probabilities in `[0, 1]`.
    fn output_head(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        #[cfg(any(test, debug_assertions))]
        let [steps] = crate::contracts::unpack_shape_contract!(
            [1, "steps", 2 * "d_hidden"],
            &x,
            &["steps"],
            &[("d_hidden", self.d_hidden())]
        );

        // The leading axis is a pinned batch of 1, so folding it away leaves
        // one row per step, in order.
        // [steps, 2 * d_hidden]
        let x = x.reshape([-1, (self.d_hidden() * 2) as isize]);

        // [steps, d_hidden / 2]
        let x = relu(self.linear1.forward(x));

        // [steps, 1]
        let x = sigmoid(self.linear2.forward(x));

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(["steps", 1], &x, &[("steps", steps)]);
        x
    }
}
