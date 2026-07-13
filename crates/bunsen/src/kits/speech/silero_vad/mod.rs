//! Silero VAD voice-activity-detection model.
//!
//! There is *some* bug in the burn CUDA backend, which we see
//! by model divergence from the golden tests on that backend alone.

/// The reference model.
#[cfg(feature = "store")]
pub mod reference {
    pub use bunsen_onnx_gen::silero::*;

    /// Reference ONNX Model.
    pub type ReferenceModel<B> = Model<B>;
}

#[cfg(feature = "store")]
mod cross_test;
#[cfg(feature = "store")]
pub mod pretrained;

pub mod blocks;

#[doc(inline)]
pub use blocks::*;
