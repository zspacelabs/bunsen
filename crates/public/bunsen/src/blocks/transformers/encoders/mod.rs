//! # Encoder blocks
//!
//! Bidirectional transformer encoder layers, and the encoders built from
//! them:
//!
//! - [`residual`]: the pre-norm encoder layer of Whisper's and Qwen3-ASR's
//!   audio encoders,
//!   [`ResidualEncoderAttentionBlock`](residual::ResidualEncoderAttentionBlock).
//! - [`rope`]: a pre-norm encoder with rotary attention,
//!   [`RopeTransformerEncoder`](rope::RopeTransformerEncoder), over
//!   [`RopeEncoderBlock`](rope::RopeEncoderBlock)s: `Nemotron-3-Diarization`'s.

pub mod residual;
pub mod rope;
