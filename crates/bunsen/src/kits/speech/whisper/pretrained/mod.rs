//! Pretrained Whisper Models.

#[cfg(feature = "store")]
mod pytorch_utils;

#[cfg(feature = "store")]
#[doc(inline)]
pub use pytorch_utils::*;
