use burn::{
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
