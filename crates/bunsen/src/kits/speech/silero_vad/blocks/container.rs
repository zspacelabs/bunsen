use burn::{
    Tensor,
    module::Module,
    prelude::Backend,
};

use crate::kits::speech::silero_vad::{
    SileroVad,
    SileroVadMeta,
};

/// Common 16/8 khz Silero VAD Container.
#[derive(Module, Debug)]
pub struct SileroVad16x8<B: Backend> {
    /// 16 kHz model.
    pub vad16: SileroVad<B>,

    /// 8 kHz model.
    pub vad8: SileroVad<B>,
}

impl<B: Backend> SileroVad16x8<B> {
    fn branch(
        &self,
        sr: usize,
    ) -> &SileroVad<B> {
        match sr {
            16000 => &self.vad16,
            8000 => &self.vad8,
            _ => panic!("unsupported sample rate: {sr}"),
        }
    }

    /// Forward.
    pub fn forward(
        &self,
        sr: usize,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>) {
        self.branch(sr).forward(input, state)
    }

    /// Initialize the state.
    pub fn init_state(
        &self,
        sr: usize,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        self.branch(sr).init_state(batch_size, device)
    }

    /// Chunk size for the given sample rate.
    pub fn chunk_size(
        &self,
        sr: usize,
    ) -> usize {
        self.branch(sr).chunk_size()
    }
}
