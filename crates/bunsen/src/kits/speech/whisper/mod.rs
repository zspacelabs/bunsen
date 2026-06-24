//! Whisper Model.
//!
//! [Whisper][w] is a large-scale, general-purpose speech recognition model.
//!
//! [w]: https://github.com/openai/whisper

pub mod blocks;
pub mod pretrained;

#[doc(inline)]
pub use blocks::{
    Whisper,
    WhisperApiConfig,
    WhisperMeta,
    WhisperStructuralConfig,
};
