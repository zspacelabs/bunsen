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
    module::{
        Module,
        Param,
        ParamId,
    },
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
        Bytes,
        Int,
        activation::{
            relu,
            sigmoid,
            tanh,
        },
        ops::PadMode,
    },
};
use burn_store::{
    BurnpackStore,
    ModuleSnapshot,
};

use crate::{
    blocks::conv::{
        ConvBlock1dConfig,
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

/// ['SileroVad'] Abstract Config.
#[derive(Config, Debug)]
pub struct SileroVadAbstractConfig {
    /// The sample rate (in Hz) this model expects, e.g. `16000`.
    pub sample_rate: usize,

    /// The reflect-padding applied to the right of the input before the STFT.
    pub input_pad: usize,

    /// STFT kernel size.
    pub stft_kernel: usize,

    /// STFT stride.
    pub stft_stride: usize,

    /// Number of frequency bins.
    pub n_freq: usize,

    /// The recurrent hidden / cell width of the LSTM.
    #[config(default = "128")]
    pub hidden: usize,
}

impl SileroVadAbstractConfig {
    /// The canonical 16 kHz model config.
    pub fn standard_16khz() -> Self {
        Self::from_signal(16000, 129)
    }

    /// The canonical 8 kHz model config.
    pub fn standard_8khz() -> Self {
        Self::from_signal(8000, 65)
    }

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
            encoder: encoder_config(self.hidden, self.n_freq),
            gate_config: lstm_gate_config(self.hidden),
            decoder: Conv1dConfig::new(self.hidden, 1, 1)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(true),
        }
    }
}

impl<B: Backend> ModuleInit<B, SileroVad<B>> for SileroVadAbstractConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<SileroVad<B>> {
        Ok(self.to_structure().try_init(device)?)
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

    /// The reflect-padding applied to the right of the input before the STFT
    /// conv.
    ///
    /// This is generally 2x the STFT stride.
    fn input_pad(&self) -> usize;

    /// The number of magnitude frequency bins feeding the encoder.
    ///
    /// This is half the STFT conv's output channels.
    fn n_freq(&self) -> usize;

    /// The kernel size of the STFT conv.
    ///
    /// This is generally 2x the STFT stride.
    fn stft_kernel(&self) -> usize;

    /// The stride of the STFT conv.
    ///
    /// This is generally n_freq - 1.
    fn stft_stride(&self) -> usize;

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

/// Builds the canonical 4-block `ReLU` conv encoder for `n_freq` input bins.
///
/// Channel flow: `n_freq -> 128 -> 64 -> 64 -> 128`, with the middle two blocks
/// striding by 2. Blocks default to no norm and `ReLU` activation.
pub fn encoder_config(
    hidden: usize,
    n_freq: usize,
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
        block(n_freq, hidden, 1),
        block(hidden, 64, 2),
        block(64, 64, 2),
        block(64, hidden, 1),
    ])
}

/// Builds an LSTM gate projection: `hidden -> 4 * hidden`, column layout.
///
/// The column layout matches the ONNX export so the original weights load
/// without transposition.
pub fn lstm_gate_config(hidden: usize) -> LinearConfig {
    LinearConfig::new(hidden, 4 * hidden)
        .with_bias(true)
        .with_layout(LinearLayout::Col)
}

/// [`SileroVad`] Structure Config.
///
/// The fully explicit structural config for a single-rate Silero VAD model.
///
/// Implements [`SileroVadMeta`]; built into a [`SileroVad`] via [`ModuleInit`].
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

    /// The LSTM Gate (hidden, feature -> gates) projection config.
    pub gate_config: LinearConfig,

    /// The `1x1` output-head conv: `hidden -> 1`.
    pub decoder: Conv1dConfig,
}

impl SileroVadMeta for SileroVadStructureConfig {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn input_pad(&self) -> usize {
        self.input_pad
    }

    fn n_freq(&self) -> usize {
        self.stft.channels_out / 2
    }

    fn stft_kernel(&self) -> usize {
        self.stft.kernel_size
    }

    fn stft_stride(&self) -> usize {
        self.stft.stride
    }

    fn hidden_size(&self) -> usize {
        self.encoder.out_channels()
    }
}

impl SileroVadStructureConfig {
    /// The canonical 16 kHz model.
    pub fn standard_16khz() -> Self {
        SileroVadAbstractConfig::standard_16khz().to_structure()
    }

    /// The canonical 8 kHz model.
    pub fn standard_8khz() -> Self {
        SileroVadAbstractConfig::standard_8khz().to_structure()
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
            input_gate: self.gate_config.init(device),
            hidden_gate: self.gate_config.init(device),
            decoder: self.decoder.init(device),
        })
    }
}

/// Silero VAD model for a single sample rate.
///
/// Implements [`SileroVadMeta`]; built by [`SileroVadStructureConfig`].
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

    /// The LSTM input (feature -> gates) projection.
    pub input_gate: Linear<B>,

    /// The LSTM recurrent (hidden -> gates) projection.
    pub hidden_gate: Linear<B>,

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

    fn stft_kernel(&self) -> usize {
        self.stft.weight.dims()[1]
    }

    fn stft_stride(&self) -> usize {
        self.stft.stride
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
            assert_shape_contract_periodically!(
                ["2", "batch", "d_hidden"],
                &state,
                &[("batch", batch), ("d_hidden", self.hidden_size())]
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

        let d_hidden = self.hidden_size();

        cfg_select! {
            any(test, debug_assertions) => {
                let [steps] = unpack_shape_contract!(
                    ["steps", "d_hidden"],
                    &seq_buf,
                    &["steps"],
                    &[("d_hidden", self.hidden_size())]
                );
                assert_shape_contract_periodically!(["2", "1", "d_hidden"], &state, &[("d_hidden", d_hidden)]);
            }
            _ => {
                let steps = features.dims()[0];
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
    fn unpack_state(state: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let cell = state.clone().slice_dim(0, 0).squeeze_dim::<2>(0);
        let hidden = state.slice_dim(0, 1).squeeze_dim::<2>(0);
        (cell, hidden)
    }

    /// Stacks `(cell, hidden)` into a packed `[2, batch, hidden]` state.
    fn pack_state(
        cell: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        Tensor::stack(vec![cell, hidden], 0)
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
        let gates = self.input_gate.forward(features) + self.hidden_gate.forward(hidden);

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
        let x = x.squeeze_dim::<2>(2);
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
    fn chunk_samples(sample_rate: usize) -> usize {
        match sample_rate {
            16000 => 512,
            8000 => 256,
            other => panic!("no test chunk for {other}"),
        }
    }

    #[test]
    fn test_config_meta() {
        let cfg16 = SileroVadAbstractConfig::standard_16khz().to_structure();
        assert_eq!(cfg16.sample_rate(), 16000);
        assert_eq!(cfg16.input_pad(), 64);
        assert_eq!(cfg16.n_freq(), 129);
        assert_eq!(cfg16.hidden_size(), 128);
        assert_eq!(cfg16.gate_size(), 512);
        // The encoder consumes the magnitude bins.
        assert_eq!(cfg16.encoder.in_channels(), cfg16.n_freq());
        cfg16.validate().unwrap();

        let cfg8 = SileroVadStructureConfig::standard_8khz();
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
        let bad = SileroVadStructureConfig {
            encoder: encoder_config(128, 64),
            ..SileroVadAbstractConfig::standard_16khz().to_structure()
        };
        assert!(matches!(bad.validate(), Err(BunsenError::Invalid(_))));
    }

    #[test]
    fn test_config_meta_matches_module() {
        let device = Default::default();

        for (cfg, n_freq) in [
            (
                SileroVadAbstractConfig::standard_16khz().to_structure(),
                129,
            ),
            (SileroVadStructureConfig::standard_8khz(), 65),
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
            SileroVadAbstractConfig::standard_16khz().to_structure(),
            SileroVadStructureConfig::standard_8khz(),
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
            SileroVadAbstractConfig::standard_16khz().to_structure(),
            SileroVadStructureConfig::standard_8khz(),
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
        let model: SileroVad<B> = SileroVadAbstractConfig::standard_16khz()
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

/// Reference model for Silero VAD.
#[derive(Module, Debug)]
pub struct ReferenceVAD<B: Backend> {
    constant32: Param<Tensor<B, 1, Int>>,
    constant41: Param<Tensor<B, 1, Int>>,
    constant42: Param<Tensor<B, 1>>,
    conv1d37: Conv1d<B>,
    conv1d38: Conv1d<B>,
    conv1d39: Conv1d<B>,
    conv1d40: Conv1d<B>,
    conv1d41: Conv1d<B>,
    linear13: Linear<B>,
    linear14: Linear<B>,
    conv1d42: Conv1d<B>,
    conv1d43: Conv1d<B>,
    conv1d44: Conv1d<B>,
    conv1d45: Conv1d<B>,
    conv1d46: Conv1d<B>,
    conv1d47: Conv1d<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    conv1d48: Conv1d<B>,
}

impl<B: Backend> ReferenceVAD<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(
        file: &str,
        device: &B::Device,
    ) -> Self {
        let mut model = Self::new(device);
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
        bytes: Bytes,
        device: &B::Device,
    ) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> ReferenceVAD<B> {
    /// Build a new reference model.
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant32: Param<Tensor<B, 1, Int>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([0i64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant41: Param<Tensor<B, 1, Int>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([1i64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant42: Param<Tensor<B, 1>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::from_data([2f64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let conv1d37 = Conv1dConfig::new(1, 258, 256)
            .with_stride(128)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d38 = Conv1dConfig::new(129, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d39 = Conv1dConfig::new(128, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d40 = Conv1dConfig::new(64, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d41 = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear13 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear14 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d42 = Conv1dConfig::new(128, 1, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d43 = Conv1dConfig::new(1, 130, 128)
            .with_stride(64)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d44 = Conv1dConfig::new(65, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d45 = Conv1dConfig::new(128, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d46 = Conv1dConfig::new(64, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d47 = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear16 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d48 = Conv1dConfig::new(128, 1, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            constant32,
            constant41,
            constant42,
            conv1d37,
            conv1d38,
            conv1d39,
            conv1d40,
            conv1d41,
            linear13,
            linear14,
            conv1d42,
            conv1d43,
            conv1d44,
            conv1d45,
            conv1d46,
            conv1d47,
            linear15,
            linear16,
            conv1d48,
        }
    }

    /// Run the module.
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        sr: i64,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let equal1_out1 = sr == 16000i64;
        let (if1_out1, if1_out2) = if equal1_out1 {
            let input = input.clone();
            let state = state.clone();
            let pad7_out1 = input.pad([(0usize, 0usize), (0usize, 64usize)], PadMode::Reflect);
            let unsqueeze31_out1: Tensor<B, 3> = pad7_out1.unsqueeze_dims::<3>(&[1]);
            let conv1d37_out1 = self.conv1d37.forward(unsqueeze31_out1);
            let slice13_out1 = conv1d37_out1.clone().slice(s![.., 0..129, ..]);
            let slice14_out1 = conv1d37_out1.slice(s![.., 129.., ..]);
            let pow13_out1 = slice13_out1.clone() * slice13_out1;
            let pow14_out1 = slice14_out1.clone() * slice14_out1;
            let add19_out1 = pow13_out1.add(pow14_out1);
            let sqrt7_out1 = add19_out1.sqrt();
            let conv1d38_out1 = self.conv1d38.forward(sqrt7_out1);
            let relu31_out1 = relu(conv1d38_out1);
            let conv1d39_out1 = self.conv1d39.forward(relu31_out1);
            let relu32_out1 = relu(conv1d39_out1);
            let conv1d40_out1 = self.conv1d40.forward(relu32_out1);
            let relu33_out1 = relu(conv1d40_out1);
            let conv1d41_out1 = self.conv1d41.forward(relu33_out1);
            let relu34_out1 = relu(conv1d41_out1);

            let feature = {
                let sliced = relu34_out1.slice(s![.., .., 0i64]);
                sliced.squeeze_dim::<2usize>(2)
            };

            let (cell, hidden) = SileroVad::unpack_state(state.clone());

            let linear13_out1 = self.linear13.forward(hidden);
            let linear14_out1 = self.linear14.forward(feature);
            let add20_out1 = linear13_out1.add(linear14_out1);

            let [g_i, g_f, g_c, g_o] = add20_out1.chunk(4, 1).try_into().unwrap();
            let i = sigmoid(g_i);
            let f = sigmoid(g_f);
            let c = g_c.tanh();
            let o = sigmoid(g_o);

            let new_cell = (f * cell) + (i * c);
            let new_hidden = o * new_cell.clone().tanh();

            let new_state = Tensor::cat(
                [
                    new_hidden.clone().unsqueeze_dims::<3>(&[0]),
                    new_cell.unsqueeze_dims::<3>(&[0]),
                ]
                .into(),
                0,
            );

            // output head
            let unsqueeze32_out1: Tensor<B, 3> = new_hidden.clone().unsqueeze_dims::<3>(&[-1]);
            let relu35_out1 = relu(unsqueeze32_out1);
            let conv1d42_out1 = self.conv1d42.forward(relu35_out1);
            let sigmoid28_out1 = sigmoid(conv1d42_out1);
            let squeeze7_out1 = sigmoid28_out1.squeeze_dims::<2>(&[1]);
            let reducemean7_out1 = { squeeze7_out1.mean_dim(1usize).squeeze_dims::<1usize>(&[1]) };
            let probs: Tensor<B, 2> = reducemean7_out1.unsqueeze_dims::<2>(&[1]);
            (probs, new_state)
        } else {
            let input = input.clone();
            let state = state.clone();
            let pad8_out1 = input.pad([(0usize, 0usize), (0usize, 32usize)], PadMode::Reflect);
            let unsqueeze36_out1: Tensor<B, 3> = pad8_out1.unsqueeze_dims::<3>(&[1]);
            let conv1d43_out1 = self.conv1d43.forward(unsqueeze36_out1);
            let slice15_out1 = conv1d43_out1.clone().slice(s![.., 0..65, ..]);
            let slice16_out1 = conv1d43_out1.slice(s![.., 65.., ..]);
            let pow15_out1 = slice15_out1.clone() * slice15_out1;
            let pow16_out1 = slice16_out1.clone() * slice16_out1;
            let add22_out1 = pow15_out1.add(pow16_out1);
            let sqrt8_out1 = add22_out1.sqrt();
            let conv1d44_out1 = self.conv1d44.forward(sqrt8_out1);
            let relu36_out1 = relu(conv1d44_out1);
            let conv1d45_out1 = self.conv1d45.forward(relu36_out1);
            let relu37_out1 = relu(conv1d45_out1);
            let conv1d46_out1 = self.conv1d46.forward(relu37_out1);
            let relu38_out1 = relu(conv1d46_out1);
            let conv1d47_out1 = self.conv1d47.forward(relu38_out1);
            let relu39_out1 = relu(conv1d47_out1);
            let gather24_out1 = {
                let sliced = relu39_out1.slice(s![.., .., 0i64]);
                sliced.squeeze_dim::<2usize>(2)
            };
            let gather25_out1 = {
                let sliced = state.clone().slice(s![0i64, .., ..]);
                sliced.squeeze_dim::<2usize>(0)
            };
            let gather26_out1 = {
                let sliced = state.slice(s![1i64, .., ..]);
                sliced.squeeze_dim::<2usize>(0)
            };
            let linear15_out1 = self.linear15.forward(gather25_out1);
            let linear16_out1 = self.linear16.forward(gather24_out1);
            let add23_out1 = linear15_out1.add(linear16_out1);
            let split_tensors = add23_out1.split_with_sizes([128, 128, 128, 128].into(), 1);
            let [split8_out1, split8_out2, split8_out3, split8_out4] =
                split_tensors.try_into().unwrap();
            let sigmoid29_out1 = sigmoid(split8_out1);
            let sigmoid30_out1 = sigmoid(split8_out2);
            let tanh15_out1 = split8_out3.tanh();
            let sigmoid31_out1 = sigmoid(split8_out4);
            let mul22_out1 = sigmoid30_out1.mul(gather26_out1);
            let mul23_out1 = sigmoid29_out1.mul(tanh15_out1);
            let add24_out1 = mul22_out1.add(mul23_out1);
            let tanh16_out1 = add24_out1.clone().tanh();
            let mul24_out1 = sigmoid31_out1.mul(tanh16_out1);
            let unsqueeze37_out1: Tensor<B, 3> = mul24_out1.clone().unsqueeze_dims::<3>(&[-1]);
            let unsqueeze38_out1: Tensor<B, 3> = mul24_out1.unsqueeze_dims::<3>(&[0]);
            let unsqueeze39_out1: Tensor<B, 3> = add24_out1.unsqueeze_dims::<3>(&[0]);
            let concat8_out1 = Tensor::cat([unsqueeze38_out1, unsqueeze39_out1].into(), 0);
            let relu40_out1 = relu(unsqueeze37_out1);
            let conv1d48_out1 = self.conv1d48.forward(relu40_out1);
            let sigmoid32_out1 = sigmoid(conv1d48_out1);
            let squeeze8_out1 = sigmoid32_out1.squeeze_dims::<2>(&[1]);
            let reducemean8_out1 = { squeeze8_out1.mean_dim(1usize).squeeze_dims::<1usize>(&[1]) };
            let unsqueeze40_out1: Tensor<B, 2> = reducemean8_out1.unsqueeze_dims::<2>(&[1]);
            (unsqueeze40_out1, concat8_out1)
        };
        (if1_out1, if1_out2)
    }
}
