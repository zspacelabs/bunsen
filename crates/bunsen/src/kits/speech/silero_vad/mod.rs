//! Silero VAD voice-activity-detection model.

pub mod blocks;

#[doc(inline)]
pub use blocks::module::{
    SileroVad,
    SileroVadConfig,
    SileroVadMeta,
};
