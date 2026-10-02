use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::Backend,
};

use crate::kits::speech::silero_vad::SileroVad;

/// Common methods for [`SileroVadContextConfig`] and [`SileroVadContext`].
///
/// Implemented by:
/// * [`SileroVadContextConfig`]
/// * [`SileroVadContext`]
pub trait SileroVadContextMeta {
    /// The sample rate (in Hz) this context expects, e.g. `16000`.
    fn sample_rate(&self) -> usize;

    /// The batch size.
    fn batch_size(&self) -> usize;

    /// The size of the previous sequence window to preserve.
    fn context_size(&self) -> usize;
}

/// Config for [`SileroVadContext`]: one stream's continuation state.
///
/// [`init`](Self::init) opens a context against a [`SileroVad`] at this
/// config's rate; the model's
/// [`context_forward`](SileroVad::context_forward) consumes it and returns
/// the next one.
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
/// What a stream carries between chunks, held by the caller rather than
/// the model: the tail of the last chunk, which the next is prefixed with,
/// and the recurrent state. So one loaded model serves any number of
/// streams, each with a context of its own; the Whisper stream driver
/// keeps one per stream it gates.
///
/// Built by [`SileroVadContextConfig`], or by
/// [`SileroVad::init_context`]; threaded through
/// [`SileroVad::context_forward`] and
/// [`SileroVad::context_forward_sequence`], each of which takes one and
/// returns the next. Implements [`SileroVadContextMeta`].
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
