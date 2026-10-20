//! Qwen3-ASR component blocks: the audio encoder, its projector, and the
//! model.
//!
//! [`Qwen3AsrConfig`] builds a [`Qwen3Asr`]: a [`Qwen3AsrAudioEncoder`] (the
//! shared
//! [`ResidualEncoderAttentionBlock`](crate::blocks::transformers::encoders::residual::ResidualEncoderAttentionBlock)
//! over chunked convolutions), a [`Qwen3AsrProjector`], and a
//! [`Qwen3Model`](crate::kits::gpts::qwen3::Qwen3Model) text model.

mod audio_encoder;
mod model;

pub use audio_encoder::*;
pub use model::*;
