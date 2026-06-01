//! Whisper

mod audio_encoder;
mod residual_decoder;
mod residual_encoder;

#[doc(inline)]
pub use audio_encoder::*;
#[doc(inline)]
pub use residual_decoder::*;
#[doc(inline)]
pub use residual_encoder::*;
