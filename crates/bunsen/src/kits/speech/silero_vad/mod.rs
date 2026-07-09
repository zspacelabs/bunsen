//! Silero VAD voice-activity-detection model.

/// The reference model.
#[cfg(feature = "store")]
pub mod reference {
    pub use bunsen_onnx_gen::silero_vad_op18_ifless::*;

    /// Reference ONNX Model.
    pub type ReferenceModel<B> = Model<B>;
}

#[cfg(feature = "store")]
mod cross_test;
#[cfg(feature = "store")]
pub mod pretrained;

pub mod blocks;
mod context;

#[doc(inline)]
pub use blocks::*;
#[doc(inline)]
pub use context::*;
