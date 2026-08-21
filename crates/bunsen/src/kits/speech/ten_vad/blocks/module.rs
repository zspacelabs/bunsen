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
    /// # The `a` axis
    ///
    /// `a` is **not** a stream-batch axis, and this implementation pins it to
    /// `1`. In the reference ONNX graph the leading dimension of the feature
    /// input lands on the LSTM's *sequence* axis, with the LSTM batch fixed at
    /// 1 by a graph constant (`new_shape__177 = [-1, 1, 80]`); the reference
    /// `ALGO_TRACE.md` §8 documents this and verifies it empirically.
    ///
    /// Two consequences, both deliberate and both currently out of scope:
    ///
    /// * **Sequence batching is available in the graph but not here.** Feeding
    ///   `T` stacked context frames as one call is bit-identical to `T`
    ///   sequential calls (§8.2), and far faster. Reaching it from bunsen needs
    ///   [`frame_features`](Self::frame_features) to reshape `[-1, 1, d_ctx,
    ///   n_freq]`, as the reference graph does, rather than the `[1, -1, d_ctx,
    ///   n_freq]` it currently uses — the two agree only at `a == 1`, which is
    ///   why the shape contract pins it there.
    /// * **Multi-stream batching is structurally impossible** against the stock
    ///   graph: batched states fail shape validation outright. It requires
    ///   patching two reshape constants (§8.3), i.e. a different model file.
    ///
    /// # Arguments
    /// * `input`: `[a, d_ctx, n_freq]` the widened feature stack, `a == 1`.
    /// * `state1`: `[a, d_hidden]` first-LSTM state, or `None` to start zeroed.
    /// * `state2`: `[a, d_hidden]` second-LSTM state, or `None` to start
    ///   zeroed.
    ///
    /// # Returns
    /// `(probabilities, state1, state2)`, with:
    /// * `probabilities`: `[a, 1]` speech probabilities in `[0, 1]`
    /// * `state1` / `state2`: `[a, d_hidden]` next LSTM states
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state1: Option<ExtLstmState<B, 2>>,
        state2: Option<ExtLstmState<B, 2>>,
    ) -> (Tensor<B, 2>, ExtLstmState<B, 2>, ExtLstmState<B, 2>) {
        // TODO: debug the 1, 1 dims; relationship to batch, seq.
        assert_eq!(state1.is_some(), state2.is_some());
        #[cfg(any(test, debug_assertions))]
        {
            use crate::contracts::assert_shape_contract;
            let [a] = crate::contracts::unpack_shape_contract!(
                ["a", "d_ctx", "n_freq"],
                &input,
                &["a"],
                &[("a", 1), ("d_ctx", self.d_ctx()), ("n_freq", self.n_freq())]
            );
            if let Some(state1) = &state1 {
                assert_shape_contract!(
                    ["a", "d_hidden"],
                    state1.shape(),
                    &[("a", a), ("d_hidden", self.d_hidden())],
                );
            }
            if let Some(state2) = &state2 {
                assert_shape_contract!(
                    ["a", "d_hidden"],
                    state2.shape(),
                    &[("a", a), ("d_hidden", self.d_hidden())],
                );
            }
        }
        let x = self.frame_features(input);

        let (x, state1, state2) = self.lstm_step(x, state1, state2);

        let x = self.output_head(x);

        (x, state1, state2)
    }

    /// Runs the conv stem over a feature stack.
    ///
    /// # Arguments
    /// * `x`: `[a, d_ctx, n_freq]` the widened feature stack.
    ///
    /// # Returns
    /// `[a, f_ctx, d_features]` embeddings.
    ///
    /// Note the `[1, -1, d_ctx, n_freq]` reshape below: the reference graph
    /// uses `[-1, 1, d_ctx, n_freq]`. The two agree at `a == 1`, which is all
    /// this model is contracted for; see [`forward`](Self::forward).
    pub fn frame_features(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        // TODO: debug the 1, 1 dims; relationship to batch, seq.
        #[cfg(any(test, debug_assertions))]
        let [a] = crate::contracts::unpack_shape_contract!(
            ["a", "d_ctx", "n_freq"],
            &x,
            &["a"],
            &[("d_ctx", self.d_ctx()), ("n_freq", self.n_freq())]
        );

        // this *appears* batch-able.
        let x = x.reshape([1, -1, 3, self.n_freq() as isize]);
        let x = self.cs1.forward(x);
        let x = self.pool.forward(x);
        let x = self.cs2.forward(x);
        let x = x.squeeze_dim(2);
        let x = x.permute([0, 2, 1]);

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["a", "f_ctx", "d_features"],
            &x,
            &[
                ("a", a),
                ("f_ctx", self.f_ctx()),
                ("d_features", self.d_features())
            ]
        );
        x
    }

    fn lstm_step(
        &self,
        x: Tensor<B, 3>,
        state1: Option<ExtLstmState<B, 2>>,
        state2: Option<ExtLstmState<B, 2>>,
    ) -> (Tensor<B, 3>, ExtLstmState<B, 2>, ExtLstmState<B, 2>) {
        // TODO: debug the 1, 1 dims; relationship to batch, seq.
        assert_eq!(state1.is_some(), state2.is_some());
        #[cfg(any(test, debug_assertions))]
        let a = {
            use crate::contracts::assert_shape_contract;
            let [a] = crate::contracts::unpack_shape_contract!(
                ["a", "f_ctx", "d_features"],
                &x,
                &["a"],
                &[("f_ctx", self.f_ctx()), ("d_features", self.d_features())]
            );
            if let Some(state1) = &state1 {
                assert_shape_contract!(
                    ["a", "d_hidden"],
                    state1.shape(),
                    &[("a", a), ("d_hidden", self.d_hidden())],
                );
            }
            if let Some(state2) = &state2 {
                assert_shape_contract!(
                    ["a", "d_hidden"],
                    state2.shape(),
                    &[("a", a), ("d_hidden", self.d_hidden())],
                );
            }
            a
        };

        // converting this to batch seems odd.
        // The existing re-shape seems to feed [batch=-1, seq=1, features=80];
        // which is a weird way to run this ...?
        let x = x.reshape([-1, 1, (self.f_ctx() * self.d_features()) as isize]);
        let (x, state1) = self.lstm1.forward(x, state1.map(Into::into));
        let y = x.reshape([1, -1, self.d_hidden() as isize]);

        let x = y.clone().swap_dims(0, 1);

        let (x, state2) = self.lstm2.forward(x, state2.map(Into::into));
        let x = x.swap_dims(0, 1);

        let x = Tensor::cat([x, y].into(), 2);
        let state1: ExtLstmState<B, 2> = state1.into();
        let state2: ExtLstmState<B, 2> = state2.into();
        #[cfg(any(test, debug_assertions))]
        {
            use crate::contracts::assert_shape_contract;
            assert_shape_contract!(
                ["a", 1, 2 * "d_hidden"],
                &x,
                &[("a", a), ("d_hidden", self.d_hidden())]
            );
            assert_shape_contract!(
                ["a", "d_hidden"],
                &state1.shape(),
                &[("a", 1), ("d_hidden", self.d_hidden())],
            );
            assert_shape_contract!(
                ["a", "d_hidden"],
                &state2.shape(),
                &[("a", 1), ("d_hidden", self.d_hidden())],
            );
        }
        (x, state1, state2)
    }

    fn output_head(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        // TODO: debug the 1, 1 dims; relationship to batch, seq.
        #[cfg(any(test, debug_assertions))]
        let [a] = crate::contracts::unpack_shape_contract!(
            ["a", 1, 2 * "d_hidden"],
            &x,
            &["a"],
            &[("d_hidden", self.d_hidden())]
        );

        let half_hidden = self.d_hidden() / 2;
        let twice_hidden = self.d_hidden() * 2;

        // this *appears* batch-able.
        let mut shape1: [usize; 3] = x.dims();
        shape1[2] = self.d_hidden() / 2;
        // [a * 1, 2 * d_hidden]
        let x = x.reshape([-1, twice_hidden as isize]);
        let x = self.linear1.forward(x);
        let x = relu(x);
        // [a * 1, d_hidden / 2]
        let x = x.reshape(shape1);

        let mut shape2: [usize; 3] = x.dims();
        shape2[2] = 1;
        // [a * 1, d_hidden / 2]
        let x = x.reshape([-1, half_hidden as isize]);
        let x = self.linear2.forward(x);
        let x = sigmoid(x);
        let x = x.reshape(shape2);
        // [a, 1]
        let x = x.squeeze_dim(2);

        #[cfg(any(test, debug_assertions))]
        // TODO: really?
        crate::contracts::assert_shape_contract!(["a", 1], &x, &[("a", a)]);
        x
    }
}
