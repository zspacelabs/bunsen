//! Whisper Model.

pub mod blocks;
pub mod decode;
pub mod driver;
pub mod logit_filters;
pub mod pretrained;

#[doc(inline)]
pub use blocks::{
    Whisper,
    WhisperApiConfig,
    WhisperMeta,
    WhisperStructuralConfig,
};
#[doc(inline)]
pub use decode::*;
