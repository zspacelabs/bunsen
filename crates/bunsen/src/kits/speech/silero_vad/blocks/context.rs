use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::Backend,
};

use crate::kits::speech::silero_vad::SileroVad;

/// Common methods for [`SileroVad`] and [`SileroVadContext`].
pub trait SileroVadContextMeta {
    /// The sample rate (in Hz) this context expects, e.g. `16000`.
    fn sample_rate(&self) -> usize;

    /// The batch size.
    fn batch_size(&self) -> usize;

    /// The size of the previous sequence window to preserve.
    fn context_size(&self) -> usize;
}

/// Config for [`SileroVadContext`].
#[derive(Config, Debug)]
pub struct SileroVadContextConfig {
    /// The sample rate (in Hz) this context expects, e.g. `16000`.
    pub sample_rate: usize,

    /// The batch size.
    #[config(default = "1")]
    pub batch_size: usize,

    /// The size of the previous sequence window to preserve.
    #[config(default = "64")]
    pub context_size: usize,
}

impl SileroVadContextMeta for SileroVadContextConfig {
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

/// Context and state for sequential mode for [`SileroVad`].
///
/// Built by [`SileroVadContextConfig`].
/// Implements [`SileroVadContextMeta`].
#[derive(Module, Debug)]
pub struct SileroVadContext<B: Backend> {
    /// The sample rate of the context.
    pub sample_rate: usize,

    /// The preceding input context.
    pub context: Tensor<B, 2>,

    /// The current input state.
    pub state: Tensor<B, 3>,
}

impl<B: Backend> SileroVadContextMeta for SileroVadContext<B> {
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

impl SileroVadContextConfig {
    /// Initializes a new context.
    pub fn init<B: Backend>(
        &self,
        vad: &SileroVad<B>,
        device: &B::Device,
    ) -> SileroVadContext<B> {
        vad.init_context(self.batch_size, self.context_size, device)
    }
}
