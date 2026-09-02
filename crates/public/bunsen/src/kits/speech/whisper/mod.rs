//! Whisper Model.

pub mod blocks;
pub mod decode;
pub mod driver;
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
#[doc(inline)]
pub use driver::*;
