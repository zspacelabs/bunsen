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
    /// Select the branch for the given signal rate.
    ///
    /// Panics if the sample rate is not supported.
    pub fn expect_branch(
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
    ///
    /// See: [`SileroVad::forward`].
    pub fn forward(
        &self,
        sr: usize,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>) {
        self.expect_branch(sr).forward(input, state)
    }

    /// Forward Sequence.
    ///
    /// See: [`SileroVad::forward_sequence`].
    pub fn forward_sequence(
        &self,
        sr: usize,
        input: Tensor<B, 3>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        self.expect_branch(sr).forward_sequence(input, state)
    }

    /// Initialize the state.
    ///
    /// See: [`SileroVad::init_state`].
    pub fn init_state(
        &self,
        sr: usize,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        self.expect_branch(sr).init_state(batch_size, device)
    }

    /// Chunk size for the given sample rate.
    ///
    /// See: [`SileroVad::chunk_size`].
    pub fn chunk_size(
        &self,
        sr: usize,
    ) -> usize {
        self.expect_branch(sr).chunk_size()
    }
}
