//! Silero VAD voice-activity-detection model.

pub mod blocks;
mod context;

#[cfg(feature = "store")]
pub mod pretrained;

#[doc(inline)]
pub use blocks::*;
#[doc(inline)]
pub use context::*;
