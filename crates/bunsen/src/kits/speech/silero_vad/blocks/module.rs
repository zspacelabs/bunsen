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
//! [`SileroVad::forward`] runs one chunk per call (matching the ONNX graph),
//! while [`SileroVad::forward_sequence`] streams a whole chunk-sequence through
//! a single stream, carrying state across chunks.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
        LinearLayout,
        PaddingConfig1d,
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
    blocks::conv::{
        ConvBlock1dConfig,
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

/// The recurrent hidden / cell width of the Silero VAD LSTM.
const HIDDEN: usize = 128;

/// [`SileroVad`] Meta.
///
/// Implemented by:
/// * [`SileroVadConfig`]
/// * [`SileroVad`]
pub trait SileroVadMeta {
    /// The sample rate (in Hz) this model expects, e.g. `16000`.
    fn sample_rate(&self) -> usize;

    /// The reflect-padding applied to the right of the input before the STFT
    /// conv.
    fn input_pad(&self) -> usize;

    /// The number of magnitude frequency bins feeding the encoder.
    ///
    /// This is half the STFT conv's output channels.
    fn n_freq(&self) -> usize;

    /// The recurrent hidden / cell width (the encoder output width, and the
    /// LSTM state width).
    fn hidden_size(&self) -> usize;

    /// The combined LSTM gate width; four gates of [`hidden_size`].
    ///
    /// [`hidden_size`]: SileroVadMeta::hidden_size
    fn gate_size(&self) -> usize {
        4 * self.hidden_size()
    }
}

/// [`SileroVad`] Config.
///
/// The fully-explicit structural config for a single-rate Silero VAD model.
/// Build the canonical pipelines with [`SileroVadConfig::standard_16khz`] /
/// [`SileroVadConfig::standard_8khz`].
///
/// Implements [`SileroVadMeta`]; built into a [`SileroVad`] via [`ModuleInit`].
#[derive(Config, Debug)]
pub struct SileroVadConfig {
    /// The sample rate (in Hz) this model expects.
    pub sample_rate: usize,

    /// The reflect-padding applied to the right of the input before the STFT
    /// conv.
    pub input_pad: usize,

    /// The STFT analysis conv: `1 -> 2 * n_freq` channels.
    pub stft: Conv1dConfig,

    /// The 4-block `ReLU` conv encoder.
    pub encoder: ConvSeq1dConfig,

    /// The LSTM recurrent (hidden -> gates) projection.
    pub hidden_gate: LinearConfig,

    /// The LSTM input (feature -> gates) projection.
    pub input_gate: LinearConfig,

    /// The `1x1` output-head conv: `hidden -> 1`.
    pub decoder: Conv1dConfig,
}

impl SileroVadMeta for SileroVadConfig {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn input_pad(&self) -> usize {
        self.input_pad
    }

    fn n_freq(&self) -> usize {
        self.stft.channels_out / 2
    }

    fn hidden_size(&self) -> usize {
        self.encoder.out_channels()
    }
}

impl SileroVadConfig {
    /// The canonical 16 kHz model.
    pub fn standard_16khz() -> Self {
        Self::standard(16000, 64, 256, 128, 129)
    }

    /// The canonical 8 kHz model.
    pub fn standard_8khz() -> Self {
        Self::standard(8000, 32, 128, 64, 65)
    }

    /// Builds a standard model from its rate-specific dimensions.
    ///
    /// * `stft_kernel` / `stft_stride` size the analysis conv; it maps `1`
    ///   input channel to `2 * n_freq` output channels (real and imaginary
    ///   halves).
    /// * The encoder, LSTM, and output head are shared in structure across
    ///   rates.
    fn standard(
        sample_rate: usize,
        input_pad: usize,
        stft_kernel: usize,
        stft_stride: usize,
        n_freq: usize,
    ) -> Self {
        Self {
            sample_rate,
            input_pad,
            stft: Conv1dConfig::new(1, 2 * n_freq, stft_kernel)
                .with_stride(stft_stride)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(false),
            encoder: encoder_config(n_freq),
            hidden_gate: lstm_gate_config(HIDDEN),
            input_gate: lstm_gate_config(HIDDEN),
            decoder: Conv1dConfig::new(HIDDEN, 1, 1)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(true),
        }
    }

    /// Validates the structural consistency of the model.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the encoder input does not match the
    /// magnitude bin count, or if the LSTM / head widths are inconsistent.
    pub fn validate(&self) -> BunsenResult<()> {
        self.encoder.validate()?;

        let hidden = self.hidden_size();
        if self.encoder.in_channels() != self.n_freq() {
            return Err(BunsenError::Invalid(format!(
                "SileroVad encoder in_channels ({}) != n_freq ({})",
                self.encoder.in_channels(),
                self.n_freq(),
            )));
        }
        if self.hidden_gate.d_input != hidden || self.input_gate.d_input != hidden {
            return Err(BunsenError::Invalid(format!(
                "SileroVad gate inputs ({}, {}) must both equal hidden ({hidden})",
                self.hidden_gate.d_input, self.input_gate.d_input,
            )));
        }
        if self.hidden_gate.d_output != self.gate_size()
            || self.input_gate.d_output != self.gate_size()
        {
            return Err(BunsenError::Invalid(format!(
                "SileroVad gate outputs ({}, {}) must both equal gate_size ({})",
                self.hidden_gate.d_output,
                self.input_gate.d_output,
                self.gate_size(),
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

impl<B: Backend> ModuleInit<B, SileroVad<B>> for SileroVadConfig {
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
            hidden_gate: self.hidden_gate.init(device),
            input_gate: self.input_gate.init(device),
            decoder: self.decoder.init(device),
        })
    }
}

/// Silero VAD model for a single sample rate.
///
/// Implements [`SileroVadMeta`]; built by [`SileroVadConfig`].
///
/// See the [module docs](self) for the pipeline structure. The forward
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

    /// The LSTM recurrent (hidden -> gates) projection.
    pub hidden_gate: Linear<B>,

    /// The LSTM input (feature -> gates) projection.
    pub input_gate: Linear<B>,

    /// The `1x1` output-head conv.
    pub decoder: Conv1d<B>,
}

impl<B: Backend> SileroVadMeta for SileroVad<B> {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn input_pad(&self) -> usize {
        self.input_pad
    }

    fn n_freq(&self) -> usize {
        self.stft.weight.dims()[0] / 2
    }

    fn hidden_size(&self) -> usize {
        self.encoder.out_channels()
    }
}

impl<B: Backend> SileroVad<B> {
    /// Allocates a zeroed recurrent state of shape `[2, batch, hidden]`.
    pub fn init_state(
        &self,
        batch: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        Tensor::zeros([2, batch, self.hidden_size()], device)
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
        let x: Tensor<B, 3> = x.unsqueeze_dims::<3>(&[1]);

        // STFT magnitude: split the [n, 2F, T] conv into real / imaginary
        // halves and combine as sqrt(real^2 + imag^2) -> [n, F, T].
        let x = self.stft.forward(x);
        let f = x.dims()[1] / 2;
        let real = x.clone().slice(s![.., 0..f, ..]);
        let imag = x.slice(s![.., f.., ..]);
        let mag = (real.powi_scalar(2) + imag.powi_scalar(2)).sqrt();

        // Encode, then take the first (and, for a single chunk, only) frame.
        let encoded = self.encoder.forward(mag);
        encoded.slice(s![.., .., 0]).squeeze_dim::<2>(2)
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
    /// The `(hidden, cell)` next states, each `[n, hidden]`.
    pub fn lstm_step(
        &self,
        feature: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
        cell: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.hidden_size();

        // Gates: recurrent projection of `hidden` plus input projection of
        // `feature`, split into [input, forget, cell, output] gates.
        let gates = self.hidden_gate.forward(hidden) + self.input_gate.forward(feature);
        let parts = gates.split_with_sizes([h, h, h, h].into(), 1);
        let [g_i, g_f, g_g, g_o]: [Tensor<B, 2>; 4] = parts.try_into().unwrap();

        let i = sigmoid(g_i);
        let forget = sigmoid(g_f);
        let g = g_g.tanh();
        let o = sigmoid(g_o);

        let new_cell = forget * cell + i * g;
        let new_hidden = o * new_cell.clone().tanh();
        (new_hidden, new_cell)
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
        let n = hidden.dims()[0];
        let x: Tensor<B, 3> = hidden.unsqueeze_dims::<3>(&[-1]);
        let x = relu(x);
        let x = self.decoder.forward(x);
        let x = sigmoid(x);
        // [n, 1, 1] -> [n, 1]; the head length is 1, so the mean is a reshape.
        x.reshape([n, 1])
    }

    /// Splits a packed `[2, batch, hidden]` state into `(hidden, cell)`.
    fn unpack_state(state: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = state.clone().slice(s![0, .., ..]).squeeze_dim::<2>(0);
        let cell = state.slice(s![1, .., ..]).squeeze_dim::<2>(0);
        (hidden, cell)
    }

    /// Stacks `(hidden, cell)` into a packed `[2, batch, hidden]` state.
    fn pack_state(
        hidden: Tensor<B, 2>,
        cell: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let hidden: Tensor<B, 3> = hidden.unsqueeze_dims::<3>(&[0]);
        let cell: Tensor<B, 3> = cell.unsqueeze_dims::<3>(&[0]);
        Tensor::cat([hidden, cell].into(), 0)
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
        let feature = self.frame_features(input);
        let (hidden, cell) = Self::unpack_state(state);
        let (hidden, cell) = self.lstm_step(feature, hidden, cell);
        let prob = self.output_head(hidden.clone());
        (prob, Self::pack_state(hidden, cell))
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
        let features = self.frame_features(input);
        let steps = features.dims()[0];

        let (mut hidden, mut cell) = Self::unpack_state(state);

        let mut hidden_steps = Vec::with_capacity(steps);
        for step in 0..steps {
            let feature = features.clone().slice(s![step..step + 1, ..]);
            let (new_hidden, new_cell) = self.lstm_step(feature, hidden, cell);
            hidden = new_hidden.clone();
            cell = new_cell;
            hidden_steps.push(new_hidden);
        }

        // Batch the output head over all steps at once.
        let all_hidden = Tensor::cat(hidden_steps, 0);
        let probs = self.output_head(all_hidden);
        (probs, Self::pack_state(hidden, cell))
    }
}

/// Builds the canonical 4-block `ReLU` conv encoder for `n_freq` input bins.
///
/// Channel flow: `n_freq -> 128 -> 64 -> 64 -> 128`, with the middle two blocks
/// striding by 2. Blocks default to no norm and `ReLU` activation.
fn encoder_config(n_freq: usize) -> ConvSeq1dConfig {
    let block = |in_channels: usize, out_channels: usize, stride: usize| {
        ConvBlock1dConfig::new(
            Conv1dConfig::new(in_channels, out_channels, 3)
                .with_stride(stride)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .with_bias(true),
        )
    };
    ConvSeq1dConfig::new(vec![
        block(n_freq, HIDDEN, 1),
        block(HIDDEN, 64, 2),
        block(64, 64, 2),
        block(64, HIDDEN, 1),
    ])
}

/// Builds an LSTM gate projection: `hidden -> 4 * hidden`, column layout.
///
/// The column layout matches the ONNX export so the original weights load
/// without transposition.
fn lstm_gate_config(hidden: usize) -> LinearConfig {
    LinearConfig::new(hidden, 4 * hidden)
        .with_bias(true)
        .with_layout(LinearLayout::Col)
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
    fn chunk_samples(sample_rate: usize) -> usize {
        match sample_rate {
            16000 => 512,
            8000 => 256,
            other => panic!("no test chunk for {other}"),
        }
    }

    #[test]
    fn test_config_meta() {
        let cfg16 = SileroVadConfig::standard_16khz();
        assert_eq!(cfg16.sample_rate(), 16000);
        assert_eq!(cfg16.input_pad(), 64);
        assert_eq!(cfg16.n_freq(), 129);
        assert_eq!(cfg16.hidden_size(), 128);
        assert_eq!(cfg16.gate_size(), 512);
        // The encoder consumes the magnitude bins.
        assert_eq!(cfg16.encoder.in_channels(), cfg16.n_freq());
        cfg16.validate().unwrap();

        let cfg8 = SileroVadConfig::standard_8khz();
        assert_eq!(cfg8.sample_rate(), 8000);
        assert_eq!(cfg8.input_pad(), 32);
        assert_eq!(cfg8.n_freq(), 65);
        assert_eq!(cfg8.hidden_size(), 128);
        assert_eq!(cfg8.encoder.in_channels(), cfg8.n_freq());
        cfg8.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_mismatch() {
        // An encoder whose input does not match the magnitude bins is invalid.
        let bad = SileroVadConfig {
            encoder: encoder_config(64),
            ..SileroVadConfig::standard_16khz()
        };
        assert!(matches!(bad.validate(), Err(BunsenError::Invalid(_))));
    }

    #[test]
    fn test_config_meta_matches_module() {
        let device = Default::default();

        for (cfg, n_freq) in [
            (SileroVadConfig::standard_16khz(), 129),
            (SileroVadConfig::standard_8khz(), 65),
        ] {
            let model: SileroVad<B> = cfg.init(&device);
            assert_eq!(model.sample_rate(), cfg.sample_rate());
            assert_eq!(model.hidden_size(), cfg.hidden_size());
            assert_eq!(model.n_freq(), n_freq);
            assert_eq!(model.input_pad(), cfg.input_pad());
        }
    }

    #[test]
    fn test_forward_shapes_and_range() {
        let device = Default::default();

        for cfg in [
            SileroVadConfig::standard_16khz(),
            SileroVadConfig::standard_8khz(),
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
            SileroVadConfig::standard_16khz(),
            SileroVadConfig::standard_8khz(),
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
        let model: SileroVad<B> = SileroVadConfig::standard_16khz().init(&device);

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
