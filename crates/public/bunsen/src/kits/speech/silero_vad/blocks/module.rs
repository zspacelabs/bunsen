//! # Silero VAD model.

use burn::{
    config::Config,
    module::Module,
    nn::{
        LinearLayout,
        PaddingConfig1d,
        activation::ActivationConfig,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    prelude::{
        Backend,
        Tensor,
        s,
    },
    tensor::{
        activation::{
            relu,
            sigmoid,
        },
        ops::PadMode,
    },
};

use crate::{
    blocks::{
        conv::{
            ConvBlock1dConfig,
            ConvBlock1dMeta,
            ConvSeq1d,
            ConvSeq1dConfig,
            ConvSeq1dMeta,
        },
        rnn::lstm::{
            FusedLstm,
            FusedLstmConfig,
        },
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    kits::speech::silero_vad::blocks::context::SileroVadContext,
    prelude::TensorOpExt,
};

/// [`SileroVad`] Signal Config: the top policy of `SileroVad`'s Stacked
/// Config.
///
/// Describes the model by its signal: sample rate, frequency bins, and widths.
/// [`try_to_stft`](Self::try_to_stft) (or its panicking twin,
/// [`to_stft`](Self::to_stft)) refines it into the [`SileroVadStftConfig`]
/// policy, which spells out the STFT geometry. It implements
/// [`ToStructureConfig`], lowering straight to [`SileroVadStructureConfig`]
/// through that step, and gets [`ModuleInit`] from the trait's blanket impl.
#[derive(Config, Debug)]
pub struct SileroVadSignalConfig {
    /// The sample rate (in Hz) this model expects, e.g. `16000`.
    pub sample_rate: usize,

    /// Number of frequency bins.
    pub n_freq: usize,

    /// The recurrent hidden / cell width of the LSTM.
    #[config(default = "128")]
    pub d_hidden: usize,

    /// The encoder bottleneck dimension.
    #[config(default = "64")]
    pub d_bottleneck: usize,
}

impl SileroVadSignalConfig {
    /// The canonical 16 kHz model config.
    pub fn standard_16khz() -> Self {
        Self::new(16000, 129)
    }

    /// The canonical 8 kHz model config.
    pub fn standard_8khz() -> Self {
        Self::new(8000, 65)
    }

    /// Refines this policy to a [`SileroVadStftConfig`]: an STFT stride of
    /// `n_freq - 1`, a kernel twice the stride, and input padding of half the
    /// stride.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] when `n_freq` is below 2, which leaves no
    /// stride.
    pub fn try_to_stft(&self) -> BunsenResult<SileroVadStftConfig> {
        if self.n_freq < 2 {
            return Err(BunsenError::Invalid(format!(
                "SileroVad needs at least 2 frequency bins, for an STFT stride (n_freq - 1) above 0; got n_freq = {}",
                self.n_freq,
            )));
        }

        let stft_stride = self.n_freq - 1;
        let stft_kernel = stft_stride * 2;
        let input_pad = stft_stride / 2;

        Ok(SileroVadStftConfig::new(
            self.sample_rate,
            self.n_freq,
            input_pad,
            stft_kernel,
            stft_stride,
        )
        .with_d_hidden(self.d_hidden)
        .with_d_bottleneck(self.d_bottleneck))
    }

    /// Refines this policy to a [`SileroVadStftConfig`]; the panicking twin of
    /// [`try_to_stft`](Self::try_to_stft).
    ///
    /// # Panics
    ///
    /// When `n_freq` is below 2.
    pub fn to_stft(&self) -> SileroVadStftConfig {
        self.try_to_stft().ok_or_panic()
    }
}

impl ToStructureConfig for SileroVadSignalConfig {
    type Structure = SileroVadStructureConfig;

    /// Refines through [`try_to_stft`](SileroVadSignalConfig::try_to_stft),
    /// then lowers the STFT policy.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] when `n_freq` is below 2, or when `d_hidden`
    /// or `d_bottleneck` is 0.
    fn try_to_structure(&self) -> BunsenResult<SileroVadStructureConfig> {
        self.try_to_stft()?.try_to_structure()
    }
}

/// [`SileroVad`] Stft Config: the policy of `SileroVad`'s Stacked Config
/// that spells out the STFT geometry.
///
/// [`SileroVadSignalConfig::to_stft`] derives one from the signal; set it
/// directly for a non-standard STFT. It implements [`ToStructureConfig`],
/// lowering to [`SileroVadStructureConfig`], and gets [`ModuleInit`] from the
/// trait's blanket impl.
#[derive(Config, Debug)]
pub struct SileroVadStftConfig {
    /// The sample rate (in Hz) this model expects, e.g. `16000`.
    pub sample_rate: usize,

    /// Number of frequency bins; above 0.
    pub n_freq: usize,

    /// The reflect-padding applied to the right of the input before the STFT.
    pub input_pad: usize,

    /// STFT kernel size; above 0.
    pub stft_kernel: usize,

    /// STFT stride; above 0.
    pub stft_stride: usize,

    /// The recurrent hidden / cell width of the LSTM; above 0.
    #[config(default = "128")]
    pub d_hidden: usize,

    /// The encoder bottleneck dimension; above 0.
    #[config(default = "64")]
    pub d_bottleneck: usize,
}

impl ToStructureConfig for SileroVadStftConfig {
    type Structure = SileroVadStructureConfig;

    /// Lowers the STFT policy.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] when `n_freq`, `stft_kernel`, `stft_stride`,
    /// `d_hidden` or `d_bottleneck` is 0. Each would build a model that
    /// panics in its first `forward` (or, with a zero `d_bottleneck` on
    /// wgpu, returns NaN).
    fn try_to_structure(&self) -> BunsenResult<SileroVadStructureConfig> {
        for (name, value) in [
            ("n_freq", self.n_freq),
            ("stft_kernel", self.stft_kernel),
            ("stft_stride", self.stft_stride),
            ("d_hidden", self.d_hidden),
            ("d_bottleneck", self.d_bottleneck),
        ] {
            if value == 0 {
                return Err(BunsenError::Invalid(format!(
                    "SileroVad needs {name} above 0; got {name} = 0"
                )));
            }
        }

        Ok(SileroVadStructureConfig {
            sample_rate: self.sample_rate,
            input_pad: self.input_pad,
            stft: Conv1dConfig::new(1, 2 * self.n_freq, self.stft_kernel)
                .with_stride(self.stft_stride)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(false),
            encoder: encoder_config(self.n_freq, self.d_hidden, self.d_bottleneck),
            lstm: FusedLstmConfig::new(self.d_hidden).with_layout(LinearLayout::Col),
            decoder: Conv1dConfig::new(self.d_hidden, 1, 1)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(true),
        })
    }
}

/// [`SileroVad`] Meta.
///
/// Implemented by:
/// * [`SileroVadStructureConfig`]
/// * [`SileroVad`]
pub trait SileroVadMeta {
    /// The sample rate (in Hz) this model expects, e.g. `16000`.
    fn sample_rate(&self) -> usize;

    /// The number of magnitude frequency bins feeding the encoder.
    ///
    /// This is half the STFT conv's output channels.
    fn n_freq(&self) -> usize;

    /// The processing chunk size.
    ///
    /// This is 2x the STFT kernel size.
    fn chunk_size(&self) -> usize {
        2 * self.stft_kernel()
    }

    /// The reflect-padding applied to the right of the input before the STFT
    /// conv.
    ///
    /// This is generally half the STFT stride.
    fn input_pad(&self) -> usize;

    /// The kernel size of the STFT conv.
    ///
    /// This is generally 2x the STFT stride.
    fn stft_kernel(&self) -> usize;

    /// The stride of the STFT conv.
    ///
    /// This is generally `n_freq` - 1.
    fn stft_stride(&self) -> usize;

    /// The recurrent hidden / cell width (the encoder output width, and the
    /// LSTM state width).
    fn d_hidden(&self) -> usize;

    /// The bottleneck width of the encoder.
    fn d_bottleneck(&self) -> usize;

    /// The combined LSTM gate width; four gates of [`d_hidden`].
    ///
    /// [`d_hidden`]: SileroVadMeta::d_hidden
    fn gate_size(&self) -> usize {
        4 * self.d_hidden()
    }
}

/// Builds the canonical 4-block `ReLU` conv encoder for `n_freq` input bins.
///
/// Channel flow: `n_freq -> d_hidden -> d_bottleneck -> d_bottleneck ->
/// d_hidden`, with the middle two blocks striding by 2.
///
/// Blocks default to no norm and `ReLU` activation.
pub fn encoder_config(
    n_freq: usize,
    d_hidden: usize,
    d_bottleneck: usize,
) -> ConvSeq1dConfig {
    let block = |in_channels: usize, out_channels: usize, stride: usize| {
        ConvBlock1dConfig::new(
            Conv1dConfig::new(in_channels, out_channels, 3)
                .with_stride(stride)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .with_bias(true),
        )
        .with_act(Some(ActivationConfig::Relu))
    };
    ConvSeq1dConfig::new(vec![
        block(n_freq, d_hidden, 1),
        block(d_hidden, d_bottleneck, 2),
        block(d_bottleneck, d_bottleneck, 2),
        block(d_bottleneck, d_hidden, 1),
    ])
}

/// [`SileroVad`] Structure Config.
///
/// The fully explicit structural config for a single-rate Silero VAD model.
/// Both policies, [`SileroVadSignalConfig`] and [`SileroVadStftConfig`], lower
/// to it.
///
/// Implements [`SileroVadMeta`]; built into a [`SileroVad`] via
/// [`ModuleInit`].
#[derive(Config, Debug)]
pub struct SileroVadStructureConfig {
    /// The sample rate (in Hz) this model expects.
    pub sample_rate: usize,

    /// The reflect-padding applied to the right of the input before the STFT
    /// conv.
    pub input_pad: usize,

    /// The STFT analysis conv: `1 -> 2 * n_freq` channels.
    pub stft: Conv1dConfig,

    /// The 4-block `ReLU` conv encoder.
    pub encoder: ConvSeq1dConfig,

    /// The config for the LSTM.
    pub lstm: FusedLstmConfig,

    /// The `1x1` output-head conv: `d_hidden -> 1`.
    pub decoder: Conv1dConfig,
}

impl SileroVadMeta for SileroVadStructureConfig {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn n_freq(&self) -> usize {
        self.stft.channels_out / 2
    }

    fn input_pad(&self) -> usize {
        self.input_pad
    }

    fn stft_kernel(&self) -> usize {
        self.stft.kernel_size
    }

    fn stft_stride(&self) -> usize {
        self.stft.stride
    }

    fn d_hidden(&self) -> usize {
        self.encoder.out_channels()
    }

    fn d_bottleneck(&self) -> usize {
        self.encoder.blocks.last().unwrap().in_channels()
    }
}

impl SileroVadStructureConfig {
    /// Validates the structural consistency of the model.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the encoder input does not match the
    /// magnitude bin count, or if the LSTM / head widths are inconsistent.
    pub fn validate(&self) -> BunsenResult<()> {
        self.encoder.validate()?;

        let hidden = self.d_hidden();
        if self.encoder.in_channels() != self.n_freq() {
            return Err(BunsenError::Invalid(format!(
                "SileroVad encoder in_channels ({}) != n_freq ({})",
                self.encoder.in_channels(),
                self.n_freq(),
            )));
        }
        if self.decoder.channels_in != hidden || self.decoder.channels_out != 1 {
            return Err(BunsenError::Invalid(format!(
                "SileroVad decoder must map hidden ({hidden}) -> 1, got {} -> {}",
                self.decoder.channels_in, self.decoder.channels_out,
            )));
        }
        Ok(())
    }
}

impl<B: Backend> ModuleInit<B, SileroVad<B>> for SileroVadStructureConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<SileroVad<B>> {
        self.validate()?;
        Ok(SileroVad {
            sample_rate: self.sample_rate,
            input_pad: self.input_pad,
            stft: self.stft.init(device),
            encoder: self.encoder.try_init(device)?,
            lstm: self.lstm.init(device),
            decoder: self.decoder.init(device),
        })
    }
}

/// Silero VAD model for a single sample rate.
///
/// [Silero VAD][s] is a small, streaming voice-activity-detection model:
/// given a short chunk of mono audio and the previous recurrent state, it
/// emits a per-chunk speech probability and the next state.
///
/// [s]: https://github.com/snakers4/silero-vad
///
/// A model is built for a single sample rate: the rate is a property of
/// the model (and its loaded weights), not a forward-time argument. A
/// checkpoint carries both rates as a
/// [`SileroVadCollection`](super::SileroVadCollection), which routes by rate.
///
/// # Pipeline
///
/// The chunk is reflect-padded on the right by
/// [`input_pad`](SileroVadMeta::input_pad) samples, then:
///
/// 1. an STFT-style analysis [`Conv1d`] (`1 -> 2 * n_freq` channels), whose
///    output halves are combined as `sqrt(real^2 + imag^2)` into `n_freq`
///    magnitude bins;
/// 2. a 4-block `ReLU` [`ConvSeq1d`] encoder producing a `d_hidden`-wide
///    feature frame;
/// 3. a single-step LSTM cell (two gate projections: one over the recurrent
///    hidden state, one over the encoder feature);
/// 4. an output head, a `ReLU`, a `1x1` [`Conv1d`] and a sigmoid, producing the
///    speech probability.
///
/// The recurrent state is `[2, batch, d_hidden]`, the LSTM hidden and cell
/// states stacked along dim 0. Each batch row is an independent stream.
///
/// # Streams
///
/// The model holds no stream state; the caller passes it in and gets the
/// next back. [`forward`](Self::forward) runs one chunk per call against a
/// bare recurrent state (matching the ONNX graph), and
/// [`forward_sequence`](Self::forward_sequence) runs a sequence of chunks,
/// `[steps, batch, samples]`, carrying the state across them. The context
/// forms, [`context_forward`](Self::context_forward) and
/// [`context_forward_sequence`](Self::context_forward_sequence), carry a
/// [`SileroVadContext`] instead: the recurrent state and the tail of the
/// last chunk, which each chunk is prefixed with, as upstream's streaming
/// wrapper does. [`SileroVadContextConfig`](super::SileroVadContextConfig)
/// opens one per stream.
///
/// Implements [`SileroVadMeta`]; built by [`SileroVadStructureConfig`]
/// (directly, or through [`SileroVadSignalConfig`] and
/// [`SileroVadStftConfig`]), or loaded with its weights through
/// [`default_silero_factory`](crate::kits::speech::silero_vad::pretrained::default_silero_factory).
///
/// The cross-checks against the ONNX reference live in the
/// `silero-model-validation` crate. On the burn CUDA backend alone the model
/// diverges from those golden tests, which points at a backend bug.
#[derive(Module, Debug)]
pub struct SileroVad<B: Backend> {
    sample_rate: usize,
    input_pad: usize,

    /// The STFT analysis conv.
    pub stft: Conv1d<B>,

    /// The `ReLU` conv encoder.
    pub encoder: ConvSeq1d<B>,

    /// The lstm.
    pub lstm: FusedLstm<B>,

    /// The `1x1` output-head conv.
    pub decoder: Conv1d<B>,
}

impl<B: Backend> SileroVadMeta for SileroVad<B> {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn n_freq(&self) -> usize {
        self.stft.weight.dims()[0] / 2
    }

    fn input_pad(&self) -> usize {
        self.input_pad
    }

    fn stft_kernel(&self) -> usize {
        self.stft.kernel_size
    }

    fn stft_stride(&self) -> usize {
        self.stft.stride
    }

    fn d_hidden(&self) -> usize {
        self.encoder.out_channels()
    }

    fn d_bottleneck(&self) -> usize {
        self.encoder.blocks.last().unwrap().in_channels()
    }
}

impl<B: Backend> SileroVad<B> {
    /// Allocates a zeroed recurrent state of shape `[2, batch, d_hidden]`.
    pub fn init_state(
        &self,
        batch: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        Tensor::zeros([2, batch, self.d_hidden()], device)
    }

    /// Construct an initial continuation context: a zero tail of
    /// `context_size` samples per row, and a zeroed state.
    ///
    /// # Panics
    ///
    /// When `context_size` is 0;
    /// [`SileroVadContextConfig::try_init`](super::SileroVadContextConfig::try_init)
    /// reports that as an error instead.
    pub fn init_context(
        &self,
        batch: usize,
        context_size: usize,
        device: &B::Device,
    ) -> SileroVadContext<B> {
        assert_context_size(context_size);
        SileroVadContext {
            sample_rate: self.sample_rate(),
            context: Tensor::zeros([batch, context_size], device),
            state: self.init_state(batch, device),
        }
    }

    /// Iterative forward sequence, with context.
    ///
    /// # Arguments
    /// # Arguments
    /// * `chunk_seq`: `[step, batch, samples]` input.
    /// * `context`: previous continuation context.
    ///
    /// # Returns
    /// `(probabilities, context)`, with:
    /// * `probabilities` : `[steps, batch]`
    /// * `context`: continuation context
    ///
    /// # Panics
    ///
    /// When the context's rate is not the model's, or the context is 0
    /// samples wide (built by hand: its constructors refuse one).
    pub fn context_forward_sequence(
        &self,
        chunk_seq: Tensor<B, 3>,
        context: SileroVadContext<B>,
    ) -> (Tensor<B, 2>, SileroVadContext<B>) {
        let SileroVadContext {
            sample_rate,
            context,
            state,
        } = context;
        assert_eq!(sample_rate, self.sample_rate());

        cfg_select! {
            any(test, debug_assertions) => {
                use crate::contracts::{
                    assert_shape_contract_periodically,
                    unpack_shape_contract,
                };
                let [steps, batch] = unpack_shape_contract!(
                    ["steps", "batch", "samples"],
                    &chunk_seq,
                    &["steps", "batch"],
                    &[("samples", self.chunk_size())]
                );
                let [context_size] = unpack_shape_contract!(
                    ["batch", "context_size"],
                    &context,
                    &["context_size"],
                    &[("batch", batch)],
                );
                assert_shape_contract_periodically!(
                    [2, "batch", "d_hidden"],
                    &state,
                    &[("batch", batch), ("d_hidden", self.d_hidden())]
                );
            }
            _ => {
                let steps = chunk_seq.dims()[0];
                let context_size = context.dims()[1];
            }
        }
        assert_context_size(context_size);

        // [1, batch, context_size]
        let context: Tensor<B, 3> = context.unsqueeze_dim(0);

        // [steps, batch, context_size]
        let context: Tensor<B, 3> = if steps <= 1 {
            context
        } else {
            let tails = chunk_seq
                .clone()
                .slice(s![0..-1, .., -(context_size as isize)..]);
            Tensor::cat(vec![context, tails], 0)
        };

        // [steps, batch, context_size + samples]
        let ext_chunk_seq: Tensor<B, 3> = Tensor::cat(vec![context, chunk_seq.clone()], 2);
        let context = ext_chunk_seq
            .clone()
            .slice(s![-1, .., -(context_size as isize)..])
            .squeeze_dim::<2>(0);

        let (out, state) = self.forward_sequence(ext_chunk_seq, state);

        (
            out,
            SileroVadContext {
                sample_rate,
                context,
                state,
            },
        )
    }

    /// Single-step forward pass with context.
    ///
    /// # Arguments
    /// * `chunk`: `[batch, samples]` input.
    /// * `context`: previous continuation context.
    ///
    /// # Returns
    /// `(probabilities, context)`, with:
    /// * `probabilities` : `[batch]`
    /// * `context`: the continuation context.
    ///
    /// # Panics
    ///
    /// When the context's rate is not the model's, or the context is 0
    /// samples wide (built by hand: its constructors refuse one).
    pub fn context_forward(
        &self,
        chunk: Tensor<B, 2>,
        context: SileroVadContext<B>,
    ) -> (Tensor<B, 1>, SileroVadContext<B>) {
        let SileroVadContext {
            sample_rate,
            context,
            state,
        } = context;
        assert_eq!(sample_rate, self.sample_rate());

        cfg_select! {
            any(test, debug_assertions) => {
                use crate::contracts::{
                    assert_shape_contract_periodically,
                    unpack_shape_contract,
                };
                let [batch] = unpack_shape_contract!(
                    ["batch", "samples"],
                    &chunk,
                    &["batch"],
                    &[("samples", self.chunk_size())]
                );
                let [context_size] = unpack_shape_contract!(
                    ["batch", "context_size"],
                    &context,
                    &["context_size"],
                    &[("batch", batch)],
                );
                assert_shape_contract_periodically!(
                    [2, "batch", "d_hidden"],
                    &state,
                    &[("batch", batch), ("d_hidden", self.d_hidden())]
                );
            }
            _ => {
                let context_size = context.dims()[1];
            }
        }
        assert_context_size(context_size);

        let ext_input = Tensor::cat(vec![context, chunk], 1);
        let context = ext_input.clone().slice(s![.., -(context_size as isize)..]);

        let (out, state) = self.forward(ext_input, state);

        (
            out,
            SileroVadContext {
                sample_rate,
                context,
                state,
            },
        )
    }

    /// Optimized [`Self::forward`] sequence.
    ///
    /// This is not context-aware, this is just a faster implementation
    /// of calling `self.forward` on an iterative sequence of churks.
    ///
    /// # Arguments
    /// * `input` - `[steps, batch, samples]` consecutive mono audio chunks.
    /// * `state` - `[2, batch, d_hidden]` recurrent state for the single
    ///   stream.
    ///
    /// # Returns
    /// `(probabilities, context, state)`, with:
    /// * `probabilities` : `[steps, batch]`
    /// * `state`: `[2, batch, d_hidden]`
    pub fn forward_sequence(
        &self,
        chunk_seq: Tensor<B, 3>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        cfg_select! {
            any(test, debug_assertions) => {
                let [steps, batch] = crate::contracts::unpack_shape_contract!(
                    ["steps", "batch", "samples"],
                    &chunk_seq,
                    &["steps", "batch"],
                );
                crate::contracts::assert_shape_contract_periodically!(
                    [2, "batch", "d_hidden"],
                    &state,
                    &[("batch", batch), ("d_hidden", self.d_hidden())]
                );
            }
            _ => {
                let [steps, batch, _] = chunk_seq.dims();
            }
        }

        // [steps, batch, d_hidden].
        let mut seq_features = self.frame_features(chunk_seq.flatten::<2>(0, 1)).reshape([
            steps,
            batch,
            self.d_hidden(),
        ]);

        // [batch, d_hidden]
        let (mut hidden, mut cell) = Self::unpack_state(state);

        macro_rules! process_steps {
            (mut $acc:ident) => {{
                for step in 0..steps {
                    // [batch, d_hidden]
                    let features = seq_features.clone().select_dim::<2>(0, step);

                    // [batch, d_hidden]
                    (hidden, cell) = self.lstm_step(features, hidden, cell);

                    // Collect the hidden states.
                    // [1, batch, d_hidden]
                    let step_hidden = hidden.clone().unsqueeze_dim::<3>(0);
                    $acc = $acc.slice_assign(s![step, .., ..], step_hidden);
                }
                $acc
            }};
        }

        // [steps, batch, d_hidden]
        let seq_hidden: Tensor<B, 3> = if B::ad_enabled(&seq_features.device()) {
            // Differentiable Sequence.
            let mut seq_hidden = Tensor::zeros_like(&seq_features);
            process_steps!(mut seq_hidden)
        } else {
            // Non-differentiable Optimization.
            // TODO: Verify that this fires.
            //
            // As there is only one reference to seq_features, the slice_assign
            // *should* convert to an in-place update, and reuse the memory.
            //
            // This is not differentiable.
            process_steps!(mut seq_features)
        };

        let out = self
            .output_head(seq_hidden.flatten(0, 1))
            .reshape([steps, batch]);

        let state = Self::pack_state(hidden, cell);

        (out, state)
    }

    /// Single-step forward pass; one chunk per batch row.
    ///
    /// Each batch row is an independent stream with its own recurrent state.
    ///
    /// # Arguments
    ///
    /// * `chunk` - `[batch, samples]` mono audio chunks (at this model's
    ///   [`sample_rate`](SileroVadMeta::sample_rate)).
    /// * `state` - `[2, batch, d_hidden]` recurrent state (see
    ///   [`init_state`](Self::init_state)).
    ///
    /// # Returns
    /// `(probabilities, context, state)`, with:
    /// * `probabilities` : `[batch]`
    /// * `state`: `[2, batch, d_hidden]`
    pub fn forward(
        &self,
        chunk: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>) {
        #[cfg(any(test, debug_assertions))]
        {
            let [batch] =
                crate::contracts::unpack_shape_contract!(["batch", "samples"], &chunk, &["batch"]);
            crate::contracts::assert_shape_contract_periodically!(
                [2, "batch", "d_hidden"],
                &state,
                &[("batch", batch), ("d_hidden", self.d_hidden())]
            );
        }

        // [batch, d_hidden]
        let features = self.frame_features(chunk);

        // [batch, d_hidden]
        let (hidden, cell) = Self::unpack_state(state);

        // [batch, d_hidden]
        let (hidden, cell) = self.lstm_step(features, hidden, cell);

        (
            self.output_head(hidden.clone()),
            Self::pack_state(hidden, cell),
        )
    }

    /// Extracts the encoder feature frame for each row of `input`.
    ///
    /// # Arguments
    ///
    /// * `input` - `[batch, samples]` mono audio chunks.
    ///
    /// # Returns
    ///
    /// `[batch, d_hidden]` feature frames (the encoder output at frame 0).
    pub fn frame_features(
        &self,
        input: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        #[cfg(any(test, debug_assertions))]
        let [batch] =
            crate::contracts::unpack_shape_contract!(["batch", "samples"], &input, &["batch"]);

        // Reflect-pad, then add the channel axis.
        // [batch, 1, samples + pad]
        let x: Tensor<B, 3> = input
            .pad([(0, self.input_pad)], PadMode::Reflect)
            .unsqueeze_dim::<3>(1);

        // STFT magnitude: split the [n, 2F, T] conv into real / imaginary
        // halves and combine as sqrt(real^2 + imag^2) -> [n, F, T].

        // [batch, 2 * n_freq, T]
        let x = self.stft.forward(x);
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", 2 * "n_freq", "T"],
            &x,
            &[("batch", batch), ("n_freq", self.n_freq())],
        );

        // [batch, n_freq, T]
        let [real_2, imag_2] = x.square().chunk(2, 1).try_into().unwrap();
        let mag = (real_2 + imag_2).sqrt();

        // Encode, then take the first (and, for a single chunk, only) frame.
        let x = self.encoder.forward(mag).select_dim::<2>(2, 0);

        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_hidden"],
            &x,
            &[("batch", batch), ("d_hidden", self.d_hidden())],
        );

        x
    }

    /// Splits a packed `[2, batch, d_hidden]` state into `(hidden, cell)`.
    /// Of shape `[batch, d_hidden]`.
    pub fn unpack_state(state: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [hidden, cell] = state.chunk(2, 0).try_into().unwrap();
        (hidden.squeeze_dim::<2>(0), cell.squeeze_dim::<2>(0))
    }

    /// Stacks `(hidden, cell)` into a packed `[2, batch, d_hidden]` state.
    pub fn pack_state(
        hidden: Tensor<B, 2>,
        cell: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        Tensor::stack(vec![hidden, cell], 0)
    }

    /// Runs one LSTM step.
    ///
    /// # Arguments
    ///
    /// * `feature` - `[batch, d_hidden]` encoder feature frame.
    /// * `cell` - `[batch, d_hidden]` previous cell state.
    /// * `hidden` - `[batch, d_hidden]` previous hidden state.
    ///
    /// # Returns
    ///
    /// The `(hidden, cell)` next states, each `[batch, d_hidden]`.
    pub fn lstm_step(
        &self,
        features: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
        cell: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.lstm.step(features, hidden, cell)
    }

    /// Runs the `1x1` conv + sigmoid output head.
    ///
    /// # Arguments
    ///
    /// * `hidden` - `[batch, d_hidden]` LSTM hidden states.
    ///
    /// # Returns
    ///
    /// `[batch]` speech probabilities in `[0, 1]`.
    pub fn output_head(
        &self,
        hidden: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let x: Tensor<B, 3> = hidden.unsqueeze_dim::<3>(2);
        let x = relu(x);
        let x = self.decoder.forward(x);
        let x = sigmoid(x);
        let x = x.squeeze_dim::<2>(1);
        let x = x.mean_dim(1);
        x.squeeze_dim::<1>(1)
    }
}

/// Refuses a zero-width context. Its tail slice, `-(0)..`, would take the
/// whole input, so each chunk after the first would be prefixed with all of
/// the one before.
fn assert_context_size(context_size: usize) {
    assert!(
        context_size > 0,
        "a SileroVadContext needs a context_size above 0; got 0"
    );
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
        support::testing::{
            DeviceMemoryGuard,
            PerformanceBackend,
            default_device,
        },
    };

    type B = PerformanceBackend;

    #[test]
    fn test_config_meta() {
        {
            let cfg = SileroVadSignalConfig::standard_16khz();
            assert_eq!(cfg.sample_rate, 16000);
            assert_eq!(cfg.n_freq, 129);

            let cfg = cfg.to_stft();
            assert_eq!(cfg.sample_rate, 16000);
            assert_eq!(cfg.n_freq, 129);
            assert_eq!(cfg.stft_stride, 128);
            assert_eq!(cfg.stft_kernel, 256);
            assert_eq!(cfg.input_pad, 64);
            assert_eq!(cfg.d_hidden, 128);
            assert_eq!(cfg.d_bottleneck, 64);

            let cfg = cfg.to_structure();
            assert_eq!(cfg.sample_rate(), 16000);
            assert_eq!(cfg.n_freq(), 129);
            assert_eq!(cfg.chunk_size(), 512);
            assert_eq!(cfg.input_pad(), 64);
            assert_eq!(cfg.stft_kernel(), 256);
            assert_eq!(cfg.stft_stride(), 128);
            assert_eq!(cfg.gate_size(), 512);
            assert_eq!(cfg.encoder.in_channels(), cfg.n_freq());
            assert_eq!(cfg.d_hidden(), 128);
            assert_eq!(cfg.d_bottleneck(), 64);
            cfg.validate().unwrap();
        }

        {
            let cfg = SileroVadSignalConfig::standard_8khz();
            assert_eq!(cfg.sample_rate, 8000);
            assert_eq!(cfg.n_freq, 65);

            let cfg = cfg.to_stft();
            assert_eq!(cfg.sample_rate, 8000);
            assert_eq!(cfg.n_freq, 65);
            assert_eq!(cfg.stft_stride, 64);
            assert_eq!(cfg.stft_kernel, 128);
            assert_eq!(cfg.input_pad, 32);
            assert_eq!(cfg.d_hidden, 128);
            assert_eq!(cfg.d_bottleneck, 64);

            let cfg = cfg.to_structure();
            assert_eq!(cfg.sample_rate(), 8000);
            assert_eq!(cfg.n_freq(), 65);
            assert_eq!(cfg.chunk_size(), 256);
            assert_eq!(cfg.input_pad(), 32);
            assert_eq!(cfg.stft_kernel(), 128);
            assert_eq!(cfg.stft_stride(), 64);
            assert_eq!(cfg.gate_size(), 512);
            assert_eq!(cfg.encoder.in_channels(), cfg.n_freq());
            assert_eq!(cfg.d_hidden(), 128);
            assert_eq!(cfg.d_bottleneck(), 64);
            cfg.validate().unwrap();
        }
    }

    #[test]
    fn test_validate_rejects_mismatch() {
        // An encoder whose input does not match the magnitude bins is invalid.
        let bad = SileroVadStructureConfig {
            encoder: encoder_config(64, 128, 64),
            ..SileroVadSignalConfig::standard_16khz().to_structure()
        };
        assert!(matches!(bad.validate(), Err(BunsenError::Invalid(_))));
    }

    /// With no frequency bins, the STFT stride (`n_freq - 1`) underflows:
    /// an error from `try_to_stft` and `try_to_structure`, not a panic.
    #[test]
    fn test_try_to_structure_rejects_zero_freq_bins() {
        let signal = SileroVadSignalConfig::new(16000, 0);
        assert!(matches!(signal.try_to_stft(), Err(BunsenError::Invalid(_))));
        assert!(matches!(
            signal.try_to_structure(),
            Err(BunsenError::Invalid(_))
        ));
    }

    /// With one frequency bin, the STFT stride (`n_freq - 1`) is 0: an error
    /// from `try_to_stft` and `try_to_structure`, not a model whose STFT conv
    /// panics on its first `forward`.
    #[test]
    fn test_try_to_structure_rejects_one_freq_bin() {
        let signal = SileroVadSignalConfig::new(16000, 1);
        assert!(matches!(signal.try_to_stft(), Err(BunsenError::Invalid(_))));
        assert!(matches!(
            signal.try_to_structure(),
            Err(BunsenError::Invalid(_))
        ));
    }

    /// An STFT policy set directly with a stride of 0 is an error from
    /// `try_to_structure`, not a model whose STFT conv panics on its first
    /// `forward`.
    #[test]
    fn test_stft_try_to_structure_rejects_zero_stride() {
        let stft = SileroVadStftConfig {
            stft_stride: 0,
            ..SileroVadSignalConfig::standard_16khz().to_stft()
        };
        assert!(matches!(
            stft.try_to_structure(),
            Err(BunsenError::Invalid(_))
        ));
    }

    /// An STFT policy set directly with a zero kernel, bin count or width is
    /// an error from `try_to_structure`, not a model that fails in its first
    /// `forward`: a zero `n_freq` or `stft_kernel` would panic there, and a
    /// zero `d_hidden` or `d_bottleneck` would panic there too or, on wgpu
    /// with a zero `d_bottleneck`, give NaN probabilities.
    #[test]
    fn test_stft_try_to_structure_rejects_zero_sizes() {
        let standard = SileroVadSignalConfig::standard_16khz().to_stft();
        for (field, stft) in [
            (
                "n_freq",
                SileroVadStftConfig {
                    n_freq: 0,
                    ..standard.clone()
                },
            ),
            (
                "stft_kernel",
                SileroVadStftConfig {
                    stft_kernel: 0,
                    ..standard.clone()
                },
            ),
            ("d_hidden", standard.clone().with_d_hidden(0)),
            ("d_bottleneck", standard.clone().with_d_bottleneck(0)),
        ] {
            assert!(
                matches!(stft.try_to_structure(), Err(BunsenError::Invalid(_))),
                "{field} = 0 lowered"
            );
        }

        // The signal policy reaches the same check through its refinement.
        let signal = SileroVadSignalConfig::standard_16khz().with_d_hidden(0);
        assert!(matches!(
            signal.try_to_structure(),
            Err(BunsenError::Invalid(_))
        ));
    }

    #[test]
    #[serial_test::serial]
    fn test_config_meta_matches_module() {
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        for (cfg, n_freq, chunk_size) in [
            (
                SileroVadSignalConfig::standard_16khz().to_structure(),
                129,
                512,
            ),
            (
                SileroVadSignalConfig::standard_8khz().to_structure(),
                65,
                256,
            ),
        ] {
            assert_eq!(cfg.chunk_size(), chunk_size);
            assert_eq!(cfg.n_freq(), n_freq);

            let model: SileroVad<B> = cfg.init(&device);
            assert_eq!(model.sample_rate(), cfg.sample_rate());
            assert_eq!(model.chunk_size(), cfg.chunk_size());
            assert_eq!(model.d_hidden(), cfg.d_hidden());
            assert_eq!(model.input_pad(), cfg.input_pad());
            assert_eq!(model.stft_kernel(), cfg.stft_kernel());
            assert_eq!(model.stft_stride(), cfg.stft_stride());
            assert_eq!(model.gate_size(), cfg.gate_size());
            assert_eq!(model.d_bottleneck(), cfg.d_bottleneck());
        }
    }

    /// Asserts that `a` and `b` answer every [`SileroVadMeta`] method alike.
    fn assert_meta_agrees(
        a: &impl SileroVadMeta,
        b: &impl SileroVadMeta,
    ) {
        assert_eq!(a.sample_rate(), b.sample_rate());
        assert_eq!(a.n_freq(), b.n_freq());
        assert_eq!(a.chunk_size(), b.chunk_size());
        assert_eq!(a.input_pad(), b.input_pad());
        assert_eq!(a.stft_kernel(), b.stft_kernel());
        assert_eq!(a.stft_stride(), b.stft_stride());
        assert_eq!(a.d_hidden(), b.d_hidden());
        assert_eq!(a.d_bottleneck(), b.d_bottleneck());
        assert_eq!(a.gate_size(), b.gate_size());
    }

    /// Each policy builds the same `SileroVad` through its structure as
    /// through the blanket `init`.
    #[test]
    #[serial_test::serial]
    fn test_policy_pathways_agree() {
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        // The signal policy, at a non-standard rate and widths.
        let signal = SileroVadSignalConfig::new(4000, 33)
            .with_d_hidden(32)
            .with_d_bottleneck(16);
        let structure = signal.to_structure();
        assert_eq!(structure.stft_stride(), 32);
        assert_eq!(structure.chunk_size(), 128);

        let lowered: SileroVad<B> = structure.init(&device);
        let direct: SileroVad<B> = signal.init(&device);
        assert_meta_agrees(&direct, &lowered);
        assert_meta_agrees(&direct, &structure);

        // The STFT policy, with a geometry `to_stft` would not choose.
        let stft = SileroVadStftConfig::new(4000, 33, 8, 48, 24)
            .with_d_hidden(32)
            .with_d_bottleneck(16);
        let structure = stft.to_structure();
        assert_eq!(structure.stft_stride(), 24);
        assert_eq!(structure.chunk_size(), 96);

        let lowered: SileroVad<B> = structure.init(&device);
        let direct: SileroVad<B> = stft.init(&device);
        assert_meta_agrees(&direct, &lowered);
        assert_meta_agrees(&direct, &structure);
    }

    #[test]
    #[serial_test::serial]
    fn test_forward_shapes_and_range() {
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        for cfg in [
            SileroVadSignalConfig::standard_16khz().to_structure(),
            SileroVadSignalConfig::standard_8khz().to_structure(),
        ] {
            let model: SileroVad<B> = cfg.init(&device);
            let batch = 3;

            let context = 64;

            let input = Tensor::<B, 2>::random(
                [batch, context + model.chunk_size()],
                Distribution::Default,
                &device,
            );
            let state = model.init_state(batch, &device);

            let (prob, next_state) = model.forward(input, state);

            assert_eq!(prob.dims(), [batch]);
            assert_eq!(next_state.dims(), [2, batch, 128]);

            // Probabilities are sigmoid outputs in [0, 1].
            let probs: Vec<f32> = prob.into_data().to_vec().unwrap();
            assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_forward_sequence_shapes() {
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let batch = 8;
        let steps = 5;
        let context = 64;

        for cfg in [
            SileroVadSignalConfig::standard_16khz().to_structure(),
            SileroVadSignalConfig::standard_8khz().to_structure(),
        ] {
            let model: SileroVad<B> = cfg.init(&device);
            let input = Tensor::random(
                [steps, batch, context + model.chunk_size()],
                Distribution::Default,
                &device,
            );
            let state = model.init_state(batch, &device);

            let (probs, next_state) = model.forward_sequence(input, state);

            assert_eq!(probs.dims(), [steps, batch]);
            assert_eq!(next_state.dims(), [2, batch, 128]);
        }
    }

    fn check_sequence_matches_stepwise<B: Backend, F>()
    where
        F: num_traits::Float + burn::tensor::Element,
    {
        // Streaming a single stream must match looping the single-step forward
        // while carrying state.
        let device = default_device();
        let model: SileroVad<B> = SileroVadSignalConfig::standard_16khz()
            .to_structure()
            .init(&device);

        let steps = 5;
        let batch = 8;
        let context = 64;

        let input = Tensor::random(
            [steps, batch, context + model.chunk_size()],
            Distribution::Default,
            &device,
        );

        let mut state = model.init_state(batch, &device);

        let (seq_probs, seq_state) = model.forward_sequence(input.clone(), state.clone());

        // Reference: feed each chunk through the single-step forward, one
        // stream.
        let mut step_probs = Vec::with_capacity(steps);
        for step in 0..steps {
            let chunk = input.clone().slice_dim(0, step).squeeze_dim::<2>(0);

            let (prob, next_state) = model.forward(chunk, state);
            state = next_state;
            step_probs.push(prob);
        }
        let step_probs: Tensor<B, 2> = Tensor::stack(step_probs, 0);

        let tol = Tolerance::<F>::default();
        seq_probs
            .into_data_as::<F>()
            .assert_approx_eq::<F>(&step_probs.into_data_as::<F>(), tol);
        seq_state
            .into_data_as::<F>()
            .assert_approx_eq::<F>(&state.into_data_as::<F>(), tol);
    }

    #[test]
    #[serial_test::serial]
    fn test_sequence_matches_stepwise_no_ad() {
        type F = <B as BackendTypes>::FloatElem;
        check_sequence_matches_stepwise::<B, F>();
    }

    #[test]
    #[serial_test::serial]
    fn test_sequence_matches_stepwise_autodiff() {
        type F = <B as BackendTypes>::FloatElem;
        check_sequence_matches_stepwise::<burn::backend::Autodiff<B>, F>();
    }

    /// A context of width 0, built by hand: `init_context` refuses one.
    fn zero_width_context(
        model: &SileroVad<B>,
        device: &<B as BackendTypes>::Device,
    ) -> SileroVadContext<B> {
        SileroVadContext {
            sample_rate: model.sample_rate(),
            context: Tensor::zeros([1, 0], device),
            state: model.init_state(1, device),
        }
    }

    /// A zero `context_size` is refused when the context is built. Its tail
    /// slice, `-(0)..`, is the whole input: each chunk after the first would
    /// get the entire previous chunk as context.
    #[test]
    #[serial_test::serial]
    #[should_panic(expected = "context_size above 0")]
    fn test_init_context_refuses_zero_width() {
        let device = default_device();
        let model: SileroVad<B> = SileroVadSignalConfig::standard_16khz()
            .to_structure()
            .init(&device);
        let _context = model.init_context(1, 0, &device);
    }

    /// `context_forward` refuses a zero-width context built by hand, rather
    /// than prefixing the next chunk with the whole of this one.
    #[test]
    #[serial_test::serial]
    #[should_panic(expected = "context_size above 0")]
    fn test_context_forward_refuses_a_zero_width_context() {
        let device = default_device();
        let model: SileroVad<B> = SileroVadSignalConfig::standard_16khz()
            .to_structure()
            .init(&device);
        let chunk = Tensor::random([1, model.chunk_size()], Distribution::Default, &device);
        let _ = model.context_forward(chunk, zero_width_context(&model, &device));
    }

    /// `context_forward_sequence` refuses a zero-width context built by
    /// hand, with a message that says why.
    #[test]
    #[serial_test::serial]
    #[should_panic(expected = "context_size above 0")]
    fn test_context_forward_sequence_refuses_a_zero_width_context() {
        let device = default_device();
        let model: SileroVad<B> = SileroVadSignalConfig::standard_16khz()
            .to_structure()
            .init(&device);
        let chunks = Tensor::random([2, 1, model.chunk_size()], Distribution::Default, &device);
        let _ = model.context_forward_sequence(chunks, zero_width_context(&model, &device));
    }
}
