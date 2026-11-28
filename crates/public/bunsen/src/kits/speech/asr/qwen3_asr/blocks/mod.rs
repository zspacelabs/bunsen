//! Qwen3-ASR component blocks: the audio encoder and the model.
//!
//! [`Qwen3AsrConfig`] builds a [`Qwen3Asr`]: a [`Qwen3AsrAudioEncoder`] (the
//! shared encoder
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack),
//! windowed, over chunked convolutions), a projector (a
//! [`BunsenMlp`](crate::blocks::mlp::BunsenMlp): linear, GELU, linear), and a
//! [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel) text model.

mod audio_encoder;
mod front_end;
mod model;

pub use audio_encoder::*;
pub use front_end::*;
pub use model::*;
