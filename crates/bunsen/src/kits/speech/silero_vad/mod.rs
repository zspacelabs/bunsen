//! Silero VAD voice-activity-detection model.

pub mod blocks;

#[cfg(feature = "cache")]
pub mod pretrained;

#[doc(inline)]
pub use blocks::{
    SileroVad,
    SileroVadBranch,
    SileroVadBranchAbstractConfig,
    SileroVadBranchMeta,
    SileroVadBranchStructureConfig,
};
