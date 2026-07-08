//! Silero VAD voice-activity-detection model.

/// The reference model.
#[cfg(feature = "store")]
pub use bunsen_onnx_gen::silero_vad_op18_ifless as reference;
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
