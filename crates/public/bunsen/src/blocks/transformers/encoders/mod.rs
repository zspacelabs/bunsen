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
//!   transformer. And the stack of them,
//!   [`ResidualEncoderTransformer`](residual::ResidualEncoderTransformer),
//!   which masks padding as keys: Sortformer v2's transformer.
//! - [`rope`]: a pre-norm encoder with rotary attention,
//!   [`RopeTransformerEncoder`](rope::RopeTransformerEncoder), over
//!   [`RopeEncoderBlock`](rope::RopeEncoderBlock)s: `Nemotron-3-Diarization`'s.
//! - [`conformer`]: the `FastConformer` encoder,
//!   [`ConformerEncoder`](conformer::ConformerEncoder): depthwise-striding
//!   subsampling, then [`ConformerBlock`](conformer::ConformerBlock)s, each two
//!   half-step feed-forwards around relative-position attention and a
//!   convolution module. Parakeet's and Sortformer v2's.

pub mod conformer;
pub mod residual;
pub mod rope;
