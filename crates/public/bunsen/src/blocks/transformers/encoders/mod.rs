//! # Encoder blocks
//!
//! Bidirectional transformer encoder layers, and the encoders built from
//! them:
//!
//! - [`residual`]: self-attention and an MLP, normalized before or after each
//!   residual, as
//!   [`ResidualEncoderAttentionBlock`](residual::ResidualEncoderAttentionBlock)
//!   ([`NormPlacement`](residual::NormPlacement)): pre-norm, the encoder layer
//!   of Whisper's and Qwen3-ASR's audio encoders; post-norm, of Sortformer v2's
//!   transformer.
//! - [`rope`]: a pre-norm encoder with rotary attention,
//!   [`RopeTransformerEncoder`](rope::RopeTransformerEncoder), over
//!   [`RopeEncoderBlock`](rope::RopeEncoderBlock)s: `Nemotron-3-Diarization`'s.

pub mod residual;
pub mod rope;
