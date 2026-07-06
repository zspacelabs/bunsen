use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
};

use crate::kits::speech::silero_vad::SileroVad;

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
            step_count: 0,
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

    /// The number of samples processed so far.
    pub step_count: usize,

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
    /// The elapsed time in seconds.
    pub fn elapsed_seconds(&self) -> f32 {
        self.step_count as f32 / self.sample_rate as f32
    }

    /*
    /// Advance the context.
    pub fn advance(
        self,
        input: Tensor<B, 2>,
        vad: &SileroVad<B>,
    ) -> Self {
        assert_eq!(
            self.sample_rate,
            vad.sample_rate(),
            "Sample rate mismatch: {} vs. {}",
            self.sample_rate,
            vad.sample_rate()
        );

        let steps = input.dims()[0];

        let context: Tensor<B, 2> = self.context;

        let buf: Tensor<B, 2> = Tensor::cat(vec![self.context, input], 0);

        let (out, state) = vad.forward_sequence(buf, self.state);

        let out = out.slice(s![])
    }
     */
}
