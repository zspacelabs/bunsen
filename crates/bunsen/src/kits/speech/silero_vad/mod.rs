//! Silero VAD voice-activity-detection model.

pub mod blocks;

#[cfg(feature = "cache")]
pub mod pretrained;

#[doc(inline)]
pub use blocks::module::{
    SileroVad,
    SileroVadAbstractConfig,
    SileroVadMeta,
    SileroVadStructureConfig,
};
