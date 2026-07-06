use burn::{
    Tensor,
    module::Module,
    prelude::Backend,
};

use crate::kits::speech::silero_vad::{
    SileroVad,
    SileroVadMeta,
};

/// Collection of sample-rate Silero VAD models.
#[derive(Module, Debug)]
pub struct SileroVadCollection<B: Backend> {
    /// Per-sample-rate models.
    pub branches: Vec<(usize, SileroVad<B>)>,
}

impl<B: Backend> SileroVadCollection<B> {
    fn branch(
        &self,
        sr: usize,
    ) -> &SileroVad<B> {
        self.branches
            .iter()
            .find(|(rate, _)| *rate == sr)
            .map(|(_, vad)| vad)
            .expect("unsupported sample rate")
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
