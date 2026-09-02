//! Pretrained Whisper Models.

#[cfg(feature = "store_pytorch")]
mod pytorch_utils;

#[cfg(feature = "store_pytorch")]
#[doc(inline)]
pub use pytorch_utils::*;

#[cfg(all(feature = "whisper-weights", feature = "store_pytorch"))]
mod load;
#[cfg(all(feature = "whisper-weights", feature = "store_pytorch"))]
pub use bunsen_bundled_whisper as bundled;
#[cfg(all(feature = "whisper-weights", feature = "store_pytorch"))]
pub use load::bundled_vocabulary;
