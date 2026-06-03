//! Whisper Component Blocks

/// Default Whisper Head Dimensionality.
pub const WHISPER_DEFAULT_D_MODEL: usize = 64;

mod audio_encoder;
mod decoder_block;
mod encoder_block;
mod text_decoder;
mod whisper_model;

#[doc(inline)]
pub use audio_encoder::*;
#[doc(inline)]
pub use decoder_block::*;
#[doc(inline)]
pub use encoder_block::*;
#[doc(inline)]
pub use text_decoder::*;
#[doc(inline)]
pub use whisper_model::*;
