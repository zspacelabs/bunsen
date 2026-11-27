//! # Encoder blocks
//!
//! Bidirectional transformer encoder layers, and the encoders built from
//! them:
//!
//! - a [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//!   whose attention sees the whole input (or its own window of it), padding
//!   masked as keys: a pre-norm encoder with rotary attention,
//!   `Nemotron-3-Diarization`'s; pre-norm layers without rotary positions,
//!   Whisper's and Qwen3-ASR's audio encoders and Laya's head; post-norm
//!   ([`NormPlacement`](super::sublayer::NormPlacement)), Sortformer v2's
//!   transformer.
//! - [`conformer`]: the `FastConformer` encoder,
//!   [`ConformerEncoderConfig`](conformer::ConformerEncoderConfig), a policy
//!   for a [`SubsampledEncoder`]: depthwise-striding subsampling, then
//!   [`ConformerBlock`](conformer::ConformerBlock)s, each two half-step
//!   feed-forwards around relative-position attention and a convolution module.
//!   Parakeet's and Sortformer v2's. Its block stack, past the subsampling, is
//!   [`ConformerTransformer`](conformer::ConformerTransformer).
//!
//! [`EncoderTransformer`] is any of these stacks, chosen by config
//! ([`EncoderTransformerConfig`]): each keeps the frame rate and runs
//! `[batch, frames, d_model]` to the same, padding masked by each item's
//! length, so a model whose stack is a choice holds one field of this type.
//!
//! [`SubsampledEncoder`] is a speech encoder's shape: a front end that
//! changes the frame rate
//! ([`EncoderSubsampling`](crate::blocks::signals::EncoderSubsampling): feature
//! stacking, or the `FastConformer`'s depthwise striding), then an
//! `EncoderTransformer`,
//! with an [`BoundaryPermutes`](crate::blocks::layout::BoundaryPermutes)
//! bridging the caller's layout to the front end's.

pub mod conformer;

mod encoder_transformer;
mod subsampled;
/// Encoder stacks to test with: [`rope_stack`](testing::rope_stack), a
/// toy-width pre-norm rotary stack.
///
/// This module exists with the `testing` feature, and in bunsen's own unit
/// tests.
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use encoder_transformer::*;
pub use subsampled::*;
