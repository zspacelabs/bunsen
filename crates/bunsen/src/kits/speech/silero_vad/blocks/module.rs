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
//! The recurrent state is packed as `[2, batch, hidden]`, stacking the LSTM
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
    /// Derive a common configuration for the given sample rate and number of
    /// frequency bins.
    ///
    /// This sets:
    /// * `stft_stride = n_freq - 1`
    /// * `stft_kernel = stft_stride * 2`
    /// * `input_pad = stft_stride / 2`
    pub fn from_signal(
        sample_rate: usize,
        n_freq: usize,
    ) -> Self {
        let stft_stride = n_freq - 1;
        let stft_kernel = stft_stride * 2;
        let input_pad = stft_stride / 2;
        Self::new(sample_rate, input_pad, stft_kernel, stft_stride, n_freq)
    }

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
    /// Allocates a zeroed recurrent state of shape `[2, batch, hidden]`.
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
    /// * `state` - `[2, batch, hidden]` recurrent state (see
    ///   [`init_state`](Self::init_state)).
    ///
    /// # Returns
    ///
    /// `(probabilities, state)`, with `probabilities` of shape `[batch, 1]` and
    /// the next `state` of shape `[2, batch, hidden]`.
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        #[cfg(any(test, debug_assertions))]
        {
            let [batch, _samples] = input.dims();
            crate::contracts::assert_shape_contract_periodically!(
                ["2", "batch", "d_hidden"],
                &state,
                &[("batch", batch), ("d_hidden", self.d_hidden())]
            );
        }

        let features = self.frame_features(input);
        let (cell, hidden) = Self::unpack_state(state);
        let (cell, hidden) = self.lstm_step(features, cell, hidden);

        (
            self.output_head(hidden.clone()),
            Self::pack_state(cell, hidden),
        )
    }

    /// Streaming forward pass over a single stream's chunk-sequence.
    ///
    /// The rows of `input` are consecutive chunks of one stream; the LSTM is
    /// run across them in order, carrying state from chunk to chunk.
    ///
    /// # Arguments
    ///
    /// * `input` - `[steps, samples]` consecutive mono audio chunks.
    /// * `state` - `[2, 1, hidden]` recurrent state for the single stream.
    ///
    /// # Returns
    ///
    /// `(probabilities, state)`, with `probabilities` of shape `[steps, 1]`
    /// (one per chunk) and the next `state` of shape `[2, 1, hidden]`.
    pub fn forward_sequence(
        &self,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        // One feature frame per chunk: [steps, hidden].
        let mut seq_buf = self.frame_features(input);

        cfg_select! {
            any(test, debug_assertions) => {
                use crate::contracts::unpack_shape_contract;
                use crate::contracts::assert_shape_contract_periodically;

                let d_hidden = self.d_hidden();

                let [steps] = unpack_shape_contract!(
                    ["steps", "d_hidden"],
                    &seq_buf,
                    &["steps"],
                    &[("d_hidden", d_hidden)]
                );
                assert_shape_contract_periodically!(["2", "1", "d_hidden"], &state, &[("d_hidden", d_hidden)]);
            }
            _ => {
                let steps = seq_buf.dims()[0];
            }
        }

        let (mut cell, mut hidden) = Self::unpack_state(state);
        for step in 0..steps {
            let features = seq_buf.clone().slice_dim(0, step);
            (cell, hidden) = self.lstm_step(features, cell, hidden);

            // We expect this to be a hidden in-place update,
            // as there are no other references to seq_buf.
            seq_buf = seq_buf.slice_assign(s![step, ..], hidden.clone());
        }

        // Batch the output head over all steps at once.
        (self.output_head(seq_buf), Self::pack_state(cell, hidden))
    }

    /// Extracts the encoder feature frame for each row of `input`.
    ///
    /// # Arguments
    ///
    /// * `input` - `[n, samples]` mono audio chunks.
    ///
    /// # Returns
    ///
    /// `[n, hidden]` feature frames (the encoder output at frame 0).
    pub fn frame_features(
        &self,
        input: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Reflect-pad, then add the channel axis: [n, 1, samples + pad].
        let x = input.pad([(0, 0), (0, self.input_pad)], PadMode::Reflect);
        let x: Tensor<B, 3> = x.unsqueeze_dim::<3>(1);

        // STFT magnitude: split the [n, 2F, T] conv into real / imaginary
        // halves and combine as sqrt(real^2 + imag^2) -> [n, F, T].
        let [real_2, imag_2] = self
            .stft
            .forward(x)
            .square()
            .chunk(2, 1)
            .try_into()
            .unwrap();
        let mag = (real_2 + imag_2).sqrt();

        // Encode, then take the first (and, for a single chunk, only) frame.
        self.encoder
            .forward(mag)
            .slice_dim(2, 0)
            .squeeze_dim::<2>(2)
    }

    /// Splits a packed `[2, batch, hidden]` state into `(cell, hidden)`.
    /// Of shape `[batch, hidden]`.
    pub fn unpack_state(state: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = state.clone().slice_dim(0, 0).squeeze_dim::<2>(0);
        let cell = state.slice_dim(0, 1).squeeze_dim::<2>(0);
        (cell, hidden)
    }

    /// Stacks `(cell, hidden)` into a packed `[2, batch, hidden]` state.
    pub fn pack_state(
        cell: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        Tensor::stack(vec![hidden, cell], 0)
    }

    /// Runs one LSTM step.
    ///
    /// # Arguments
    ///
    /// * `feature` - `[n, hidden]` encoder feature frame.
    /// * `hidden` - `[n, hidden]` previous hidden state.
    /// * `cell` - `[n, hidden]` previous cell state.
    ///
    /// # Returns
    ///
    /// The `(cell, hidden)` next states, each `[n, hidden]`.
    pub fn lstm_step(
        &self,
        features: Tensor<B, 2>,
        cell: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Gates: recurrent projection of `hidden` plus input projection of
        // `feature`, split into [input, forget, cell, output] gates.
        let gates = self.lstm_features.forward(features) + self.lstm_hidden.forward(hidden);

        let [g_i, g_f, g_c, g_o] = gates.chunk(4, 1).try_into().unwrap();

        let input_values = sigmoid(g_i);
        let forget_values = sigmoid(g_f);
        let candidate_cell_values = tanh(g_c);
        let output_values = sigmoid(g_o);

        let new_cell = forget_values * cell + input_values * candidate_cell_values;
        let new_hidden = output_values * tanh(new_cell.clone());
        (new_cell, new_hidden)
    }

    /// Runs the `1x1` conv + sigmoid output head.
    ///
    /// # Arguments
    ///
    /// * `hidden` - `[n, hidden]` LSTM hidden states.
    ///
    /// # Returns
    ///
    /// `[n, 1]` speech probabilities in `[0, 1]`.
    pub fn output_head(
        &self,
        hidden: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let x: Tensor<B, 3> = hidden.unsqueeze_dim::<3>(2);
        let x = relu(x);
        let x = self.decoder.forward(x);
        let x = sigmoid(x);
        let x = x.squeeze_dim::<2>(1);
        x.mean_dim(1)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;

    /// A valid chunk length for the given sample rate (standard Silero chunk).
    pub fn chunk_samples(sample_rate: usize) -> usize {
        match sample_rate {
            16000 => 512,
            8000 => 256,
            other => panic!("no test chunk for {other}"),
        }
    }

    #[test]
    fn test_config_meta() {
        let cfg16 = SileroVadSignalConfig::standard_16khz().to_structure();
        assert_eq!(cfg16.sample_rate(), 16000);
        assert_eq!(cfg16.input_pad(), 64);
        assert_eq!(cfg16.n_freq(), 129);
        assert_eq!(cfg16.d_hidden(), 128);
        assert_eq!(cfg16.gate_size(), 512);
        // The encoder consumes the magnitude bins.
        assert_eq!(cfg16.encoder.in_channels(), cfg16.n_freq());
        cfg16.validate().unwrap();

        let cfg8 = SileroVadSignalConfig::standard_8khz().to_structure();
        assert_eq!(cfg8.sample_rate(), 8000);
        assert_eq!(cfg8.input_pad(), 32);
        assert_eq!(cfg8.n_freq(), 65);
        assert_eq!(cfg8.d_hidden(), 128);
        assert_eq!(cfg8.encoder.in_channels(), cfg8.n_freq());
        cfg8.validate().unwrap();
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
    fn test_forward_shapes_and_range() {
        let device = Default::default();

        for cfg in [
            SileroVadSignalConfig::standard_16khz().to_structure(),
            SileroVadSignalConfig::standard_8khz().to_structure(),
        ] {
            let model: SileroVad<B> = cfg.init(&device);
            let batch = 3;
            let input = Tensor::<B, 2>::random(
                [batch, chunk_samples(model.sample_rate())],
                Distribution::Default,
                &device,
            );
            let state = model.init_state(batch, &device);

            let (prob, next_state) = model.forward(input, state);

            assert_eq!(prob.dims(), [batch, 1]);
            assert_eq!(next_state.dims(), [2, batch, 128]);

            // Probabilities are sigmoid outputs in [0, 1].
            let probs: Vec<f32> = prob.into_data().to_vec().unwrap();
            assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }

    #[test]
    fn test_forward_sequence_shapes() {
        let device = Default::default();

        for cfg in [
            SileroVadSignalConfig::standard_16khz().to_structure(),
            SileroVadSignalConfig::standard_8khz().to_structure(),
        ] {
            let model: SileroVad<B> = cfg.init(&device);
            let steps = 5;
            let input = Tensor::<B, 2>::random(
                [steps, chunk_samples(model.sample_rate())],
                Distribution::Default,
                &device,
            );
            let state = model.init_state(1, &device);

            let (probs, next_state) = model.forward_sequence(input, state);

            assert_eq!(probs.dims(), [steps, 1]);
            assert_eq!(next_state.dims(), [2, 1, 128]);
        }
    }

    #[test]
    fn test_sequence_matches_stepwise() {
        // Streaming a single stream must match looping the single-step forward
        // while carrying state.
        let device = Default::default();
        let model: SileroVad<B> = SileroVadSignalConfig::standard_16khz()
            .to_structure()
            .init(&device);

        let steps = 4;
        let input = Tensor::<B, 2>::random(
            [steps, chunk_samples(model.sample_rate())],
            Distribution::Default,
            &device,
        );

        let (seq_probs, seq_state) =
            model.forward_sequence(input.clone(), model.init_state(1, &device));

        // Reference: feed each chunk through the single-step forward, one stream.
        let mut state = model.init_state(1, &device);
        let mut step_probs = Vec::with_capacity(steps);
        for step in 0..steps {
            let chunk = input.clone().slice(s![step..step + 1, ..]);
            let (prob, next_state) = model.forward(chunk, state);
            state = next_state;
            step_probs.push(prob);
        }
        let step_probs = Tensor::cat(step_probs, 0);

        let tol = Tolerance::<f32>::default();
        seq_probs
            .into_data()
            .assert_approx_eq::<f32>(&step_probs.into_data(), tol);
        seq_state
            .into_data()
            .assert_approx_eq::<f32>(&state.into_data(), tol);
    }
}
