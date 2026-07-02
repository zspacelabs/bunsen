use burn::{
    module::Module,
    prelude::Backend,
};

use crate::kits::speech::silero_vad::SileroVadBranch;

/// Common 16/8 khz Silero VAD Container.
#[derive(Module, Debug)]
pub struct SileroVad<B: Backend> {
    /// 16 kHz model.
    pub vad16: SileroVadBranch<B>,

    /// 8 kHz model.
    pub vad8: SileroVadBranch<B>,
}
