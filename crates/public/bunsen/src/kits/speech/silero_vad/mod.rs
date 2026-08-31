//! Silero VAD voice-activity-detection model.
//!
//! There is *some* bug in the burn CUDA backend, which we see
//! by model divergence from the golden tests on that backend alone.
//!
//! The ONNX reference this was transliterated from, and the cross-checks
//! against it, live in the `silero-model-validation` crate.

#[cfg(feature = "store")]
pub mod pretrained;

pub mod blocks;

#[doc(inline)]
pub use blocks::*;
