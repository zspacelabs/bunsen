use burn::{
    Tensor,
    module::Module,
    prelude::Backend,
};

use crate::kits::speech::silero_vad::SileroVad;

/// Common 16/8 khz Silero VAD Container.
#[derive(Module, Debug)]
pub struct SileroVad16x8<B: Backend> {
    /// 16 kHz model.
    pub vad16: SileroVad<B>,

    /// 8 kHz model.
    pub vad8: SileroVad<B>,
}

impl<B: Backend> SileroVad16x8<B> {
    /// Forward.
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        sr: usize,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        match sr {
            16000 => self.vad16.forward(input, state),
            8000 => self.vad8.forward(input, state),
            _ => panic!("unsupported sample rate: {sr}"),
        }
    }
}
