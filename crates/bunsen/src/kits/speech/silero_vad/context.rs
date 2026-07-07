use burn::{
    Tensor,
    config::Config,
    prelude::{
        Backend,
        s,
    },
};

use crate::{
    contracts::{
        assert_shape_contract,
        unpack_shape_contract,
    },
    kits::speech::silero_vad::{
        SileroVad,
        SileroVadMeta,
    },
};

/// Common meta for [`VadRunningContext`] and [`VadRunningContextConfig`].
pub trait VadRunningContextMeta {
    /// The sample rate (in Hz) this context expects, e.g. `16000`.
    fn sample_rate(&self) -> usize;

    /// The batch size.
    fn batch_size(&self) -> usize;

    /// The size of the previous sequence window to preserve.
    fn context_size(&self) -> usize;
}

/// Config for [`VadRunningContext`].
#[derive(Config, Debug)]
pub struct VadRunningContextConfig {
    /// The sample rate (in Hz) this context expects, e.g. `16000`.
    pub sample_rate: usize,

    /// The batch size.
    #[config(default = "1")]
    pub batch_size: usize,

    /// The size of the previous sequence window to preserve.
    #[config(default = "64")]
    pub context_size: usize,
}

impl VadRunningContextMeta for VadRunningContextConfig {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn context_size(&self) -> usize {
        self.context_size
    }
}

impl VadRunningContextConfig {
    /// Initializes a new running context.
    pub fn init<B: Backend>(
        &self,
        vad: &SileroVad<B>,
        device: &B::Device,
    ) -> VadRunningContext<B> {
        VadRunningContext {
            sample_rate: self.sample_rate,
            context: Tensor::zeros([self.batch_size, self.context_size], device),
            state: vad.init_state(self.batch_size, device),
        }
    }
}

/// Running context state for sequentially chunked VAD inference.
///
/// See: [`VadRunningContextConfig`].
pub struct VadRunningContext<B: Backend> {
    /// The sample rate (in Hz) this context expects, e.g. `16000`.
    pub sample_rate: usize,

    /// The preceding input context.
    pub context: Tensor<B, 2>,

    /// The current input state.
    pub state: Tensor<B, 3>,
}

impl<B: Backend> VadRunningContextMeta for VadRunningContext<B> {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn batch_size(&self) -> usize {
        self.context.dims()[0]
    }

    fn context_size(&self) -> usize {
        self.context.dims()[1]
    }
}

impl<B: Backend> VadRunningContext<B> {
    /// Predict the VAD output for a chunk of input.
    ///
    /// Advances the context.
    ///
    /// # Arguments
    /// * `chunk`: A tensor of shape `[batch, samples = vad.chunk_size()]`.
    /// * `vad`: the VAD model.
    ///
    /// # Returns
    /// * Predictions of shape `[batch]`.
    pub fn predict_chunk(
        &mut self,
        chunk: Tensor<B, 2>,
        vad: &SileroVad<B>,
    ) -> Tensor<B, 1> {
        assert_eq!(self.sample_rate, vad.sample_rate());
        assert_shape_contract!(
            ["batch", "samples"],
            &chunk,
            &[("batch", self.batch_size()), ("samples", vad.chunk_size())],
        );

        let ext_input = Tensor::cat(vec![self.context.clone(), chunk], 1);
        self.context = ext_input
            .clone()
            .slice(s![.., -(self.context_size() as isize)..]);

        let (out, state) = vad.forward(ext_input, self.state.clone());
        self.state = state;

        // [batch]
        out
    }

    /// Process a series of chunks.
    ///
    /// Advances the context.
    ///
    /// # Arguments
    /// * `chunk_seq`: A tensor of shape `[steps, batch, samples =
    ///   vad.chunk_size()]`.
    /// * `vad`: The VAD model.
    ///
    /// # Returns
    /// * Predictions of shape `[steps, batch]`.
    pub fn predict_chunk_sequence(
        &mut self,
        chunk_seq: Tensor<B, 3>,
        vad: &SileroVad<B>,
    ) -> Tensor<B, 2> {
        assert_eq!(self.sample_rate, vad.sample_rate());
        let [steps] = unpack_shape_contract!(
            ["steps", "batch", "samples"],
            &chunk_seq,
            &["steps"],
            &[("batch", self.batch_size()), ("samples", vad.chunk_size())],
        );

        // [1, batch, context_size]
        let context: Tensor<B, 3> = self.context.clone().unsqueeze_dim(0);

        // [steps, batch, context_size]
        let context: Tensor<B, 3> = if steps <= 1 {
            context
        } else {
            let tails = chunk_seq
                .clone()
                .slice(s![0..-1, -(self.context_size() as i32)..]);
            Tensor::cat(vec![context, tails], 0)
        };

        // [steps, batch, context_size + samples]
        let ext_chunk_seq: Tensor<B, 3> = Tensor::cat(vec![context, chunk_seq.clone()], 2);
        self.context = ext_chunk_seq
            .clone()
            .slice(s![-1, .., -(self.context_size() as i32)..])
            .unsqueeze_dim(0);

        let (out_seq, state) = vad.forward_sequence(ext_chunk_seq, self.state.clone());
        self.state = state;

        // [steps, batch]
        out_seq
    }
}
