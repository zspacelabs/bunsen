//! # Silero VAD model.
//!
//! [Silero VAD][s] is a small, streaming voice-activity-detection model: given
//! a short chunk of mono audio and the previous recurrent state, it emits a
//! per-chunk speech probability and the next state.
//!
//! [s]: https://github.com/snakers4/silero-vad
//!
//! A [`SileroVad`] model is
//! built for a single sample rate &mdash; the rate is a property of the model
//! (and its loaded weights), not a forward-time argument. Multi-rate routing,
//! if needed, belongs at a higher level.
//!
//! The pipeline is:
//!
//! 1. an STFT-style analysis [`Conv1d`] (`1 -> 2 * n_freq` channels), whose
//!    output halves are combined as `sqrt(real^2 + imag^2)` into `n_freq`
//!    magnitude bins,
//! 2. a 4-block `ReLU` [`ConvSeq1d`] encoder producing a `hidden`-wide feature
//!    frame,
//! 3. a single-step LSTM cell (two gate projections: one over the recurrent
//!    hidden state, one over the encoder feature),
//! 4. a `1x1` [`Conv1d`] + sigmoid output head producing the speech
//!    probability.
//!
//! The recurrent state is packed as `[2, batch, d_hidden]`, stacking the LSTM
//! hidden and cell states along dim 0.
//!
//! [`SileroVad::forward`] runs one chunk per call (matching the ONNX
//! graph), while [`SileroVad::forward_sequence`] streams a whole
//! chunk-sequence through a single stream, carrying state across chunks.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
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
            tanh,
        },
        ops::PadMode,
    },
};

use crate::{
    blocks::conv::{
        ConvBlock1dConfig,
        ConvBlock1dMeta,
        ConvSeq1d,
        ConvSeq1dConfig,
        ConvSeq1dMeta,
    },
    burner::module::ModuleInit,
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// [`SileroVad`] Signal Config.
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

    /// Converts to [`SileroVadStftConfig`].
    pub fn to_stft(&self) -> SileroVadStftConfig {
        let stft_stride = self.n_freq - 1;
        let stft_kernel = stft_stride * 2;
        let input_pad = stft_stride / 2;

        SileroVadStftConfig::new(
            self.sample_rate,
            self.n_freq,
            input_pad,
            stft_kernel,
            stft_stride,
        )
        .with_d_hidden(self.d_hidden)
        .with_d_bottleneck(self.d_bottleneck)
    }

    /// Converts to [`SileroVadStructureConfig`].
    pub fn to_structure(&self) -> SileroVadStructureConfig {
        self.to_stft().to_structure()
    }
}

impl<B: Backend> ModuleInit<B, SileroVad<B>> for SileroVadSignalConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<SileroVad<B>> {
        self.to_stft().try_init(device)
    }
}

/// [`SileroVad`] Stft Config.
#[derive(Config, Debug)]
pub struct SileroVadStftConfig {
    /// The sample rate (in Hz) this model expects, e.g. `16000`.
    pub sample_rate: usize,

    /// Number of frequency bins.
    pub n_freq: usize,

    /// The reflect-padding applied to the right of the input before the STFT.
    pub input_pad: usize,

    /// STFT kernel size.
    pub stft_kernel: usize,

    /// STFT stride.
    pub stft_stride: usize,

    /// The recurrent hidden / cell width of the LSTM.
    #[config(default = "128")]
    pub d_hidden: usize,

    /// The encoder bottleneck dimension.
    #[config(default = "64")]
    pub d_bottleneck: usize,
}

impl SileroVadStftConfig {
    /// Convert this config into a [`SileroVadStructureConfig`].
    fn to_structure(&self) -> SileroVadStructureConfig {
        SileroVadStructureConfig {
            sample_rate: self.sample_rate,
            input_pad: self.input_pad,
            stft: Conv1dConfig::new(1, 2 * self.n_freq, self.stft_kernel)
                .with_stride(self.stft_stride)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(false),
            encoder: encoder_config(self.n_freq, self.d_hidden, self.d_bottleneck),
            gate_config: lstm_gate_config(self.d_hidden),
            decoder: Conv1dConfig::new(self.d_hidden, 1, 1)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(true),
        }
    }
}

impl<B: Backend> ModuleInit<B, SileroVad<B>> for SileroVadStftConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<SileroVad<B>> {
        self.to_structure().try_init(device)
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
    fn chunk_size(&self) -> usize {
        self.n_freq() * 2
    }

    /// The reflect-padding applied to the right of the input before the STFT
    /// conv.
    ///
    /// This is generally 2x the STFT stride.
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

/// Builds an LSTM gate projection: `d_hidden -> 4 * d_hidden`, column layout.
///
/// The column layout matches the ONNX export so the original weights load
/// without transposition.
pub fn lstm_gate_config(d_hidden: usize) -> LinearConfig {
    LinearConfig::new(d_hidden, 4 * d_hidden)
        .with_bias(true)
        .with_layout(LinearLayout::Col)
}

/// [`SileroVad`] Structure Config.
///
/// The fully explicit structural config for a single-rate Silero VAD model.
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

    /// The shared LSTM Gate (hidden, feature -> gates) projection config.
    pub gate_config: LinearConfig,

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
            lstm_features: self.gate_config.init(device),
            lstm_hidden: self.gate_config.init(device),
            decoder: self.decoder.init(device),
        })
    }
}

/// Silero VAD model for a single sample rate.
///
/// Implements [`SileroVadMeta`]; built by
/// [`SileroVadStructureConfig`].
///
/// The forward
/// primitives ([`frame_features`], [`lstm_step`], [`output_head`]) are shared
/// by the single-step [`forward`] and the streaming [`forward_sequence`].
///
/// [`frame_features`]: SileroVad::frame_features
/// [`lstm_step`]: SileroVad::lstm_step
/// [`output_head`]: SileroVad::output_head
/// [`forward`]: SileroVad::forward
/// [`forward_sequence`]: SileroVad::forward_sequence
#[derive(Module, Debug)]
pub struct SileroVad<B: Backend> {
    sample_rate: usize,
    input_pad: usize,

    /// The STFT analysis conv.
    pub stft: Conv1d<B>,

    /// The `ReLU` conv encoder.
    pub encoder: ConvSeq1d<B>,

    /// The LSTM input (feature -> gates) projection.
    pub lstm_features: Linear<B>,

    /// The LSTM recurrent (hidden -> gates) projection.
    pub lstm_hidden: Linear<B>,

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
        self.stft.weight.dims()[1]
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

    /// Single-step forward pass; one chunk per batch row.
    ///
    /// Each batch row is an independent stream with its own recurrent state.
    ///
    /// # Arguments
    ///
    /// * `input` - `[batch, samples]` mono audio chunks (at this model's
    ///   [`sample_rate`](SileroVadMeta::sample_rate)).
    /// * `state` - `[2, batch, d_hidden]` recurrent state (see
    ///   [`init_state`](Self::init_state)).
    ///
    /// # Returns
    ///
    /// `(probabilities, state)`, with `probabilities` of shape `[batch, 1]` and
    /// the next `state` of shape `[2, batch, d_hidden]`.
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>) {
        #[cfg(any(test, debug_assertions))]
        {
            let [batch] = unpack_shape_contract!(["batch", "samples"], &input, &["batch"],);
            assert_shape_contract_periodically!(
                ["2", "batch", "d_hidden"],
                &state,
                &[("2", 2), ("batch", batch), ("d_hidden", self.d_hidden())]
            );
        }

        // [batch, d_hidden]
        let features = self.frame_features(input);

        // [batch, d_hidden]
        let (hidden, cell) = Self::unpack_state(state);

        // [batch, d_hidden]
        let (hidden, cell) = self.lstm_step(features, hidden, cell);

        (
            self.output_head(hidden.clone()),
            Self::pack_state(hidden, cell),
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
    /// `(probabilities, state)`, with `probabilities` of shape `[steps, batch]`
    /// (one per chunk) and the next `state` of shape `[2, batch, d_hidden]`.
    pub fn forward_sequence(
        &self,
        input: Tensor<B, 3>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        cfg_select! {
            any(test, debug_assertions) => {
                let [steps, batch] = unpack_shape_contract!(
                    ["steps", "batch", "samples"],
                    &input,
                    &["steps", "batch"],
                );
                assert_shape_contract_periodically!(
                    ["2", "batch", "d_hidden"],
                    &state,
                    &[("2", 2), ("batch", batch), ("d_hidden", self.d_hidden())]
                );
            }
            _ => {
                let [steps, batch, _] = input.dims();
            }
        }

        // [steps, batch, d_hidden].
        let mut seq_features =
            self.frame_features(input.flatten::<2>(0, 1))
                .reshape([steps, batch, self.d_hidden()]);

        // [batch, d_hidden]
        let (mut hidden, mut cell) = Self::unpack_state(state);

        macro_rules! process_steps {
            (mut $acc:ident) => {{
                for step in 0..steps {
                    // [batch, d_hidden]
                    let features = seq_features.clone().slice_dim(0, step).squeeze_dim::<2>(0);

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
        let [batch] = unpack_shape_contract!(["batch", "samples"], &input, &["batch"],);

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
        assert_shape_contract_periodically!(
            ["batch", "2" * "n_freq", "T"],
            &x,
            &[("2", 2), ("batch", batch), ("n_freq", self.n_freq())],
        );

        // [batch, n_freq, T]
        let [real_2, imag_2] = x.square().chunk(2, 1).try_into().unwrap();
        let mag = (real_2 + imag_2).sqrt();

        // Encode, then take the first (and, for a single chunk, only) frame.
        let x = self
            .encoder
            .forward(mag)
            .slice_dim(2, 0)
            .squeeze_dim::<2>(2);

        #[cfg(any(test, debug_assertions))]
        assert_shape_contract_periodically!(
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
        cfg_select! {
            any(test, debug_assertions) => {
                let [batch] = unpack_shape_contract!(
                    ["batch", "d_hidden"],
                    &features,
                    &["batch"],
                    &[("d_hidden", self.d_hidden())],
                );
                assert_shape_contract_periodically!(
                    ["batch", "d_hidden"],
                    &cell,
                    &[("batch", batch), ("d_hidden", self.d_hidden())]
                );
                assert_shape_contract_periodically!(
                    ["batch", "d_hidden"],
                    &hidden,
                    &[("batch", batch), ("d_hidden", self.d_hidden())]
                );
            }
        }

        // Gates: recurrent projection of `hidden` plus input projection of
        // `feature`, split into [input, forget, cell, output] gates.
        let gates = self.lstm_features.forward(features) + self.lstm_hidden.forward(hidden);

        #[cfg(any(test, debug_assertions))]
        assert_shape_contract_periodically!(
            ["batch", "4" * "d_hidden"],
            &gates,
            &[("batch", batch), ("4", 4), ("d_hidden", self.d_hidden())]
        );

        /*
        let [g_i, g_f, g_c, g_o] = gates.chunk(4, 1).try_into().unwrap();

        let input_values = sigmoid(g_i);
        let forget_values = sigmoid(g_f);
         */

        // Funny chunking to reduce sigmoid() calls.
        let [g_i_f, g_c_o] = gates.chunk(2, 1).try_into().unwrap();
        let [g_c, g_o] = g_c_o.chunk(2, 1).try_into().unwrap();

        let [input_values, forget_values] = sigmoid(g_i_f).chunk(2, 1).try_into().unwrap();
        let candidate_cell_values = tanh(g_c);
        let output_values = sigmoid(g_o);

        let cell = forget_values * cell + input_values * candidate_cell_values;
        let hidden = output_values * tanh(cell.clone());
        (hidden, cell)
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

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
        backend::BackendTypes,
    };

    use super::*;
    use crate::support::testing::PerformanceBackend;

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

    #[test]
    fn test_config_meta_matches_module() {
        let device = Default::default();

        for (cfg, n_freq) in [
            (SileroVadSignalConfig::standard_16khz().to_structure(), 129),
            (SileroVadSignalConfig::standard_8khz().to_structure(), 65),
        ] {
            let model: SileroVad<B> = cfg.init(&device);
            assert_eq!(model.sample_rate(), cfg.sample_rate());
            assert_eq!(model.d_hidden(), cfg.d_hidden());
            assert_eq!(model.n_freq(), n_freq);
            assert_eq!(model.input_pad(), cfg.input_pad());
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_forward_shapes_and_range() {
        let device = Default::default();

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
        let device = Default::default();

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
        let device = Default::default();
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

        // Reference: feed each chunk through the single-step forward, one stream.
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
            .into_data()
            .assert_approx_eq::<F>(&step_probs.into_data(), tol);
        seq_state
            .into_data()
            .assert_approx_eq::<F>(&state.into_data(), tol);
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
}
