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
//!   [`ConformerEncoderConfig`](conformer::ConformerEncoderConfig), a policy
//!   for a [`SubsampledEncoder`]: depthwise-striding subsampling, then
//!   [`ConformerBlock`](conformer::ConformerBlock)s, each two half-step
//!   feed-forwards around relative-position attention and a convolution module.
//!   Parakeet's and Sortformer v2's. Its block stack, past the subsampling, is
//!   [`ConformerTransformer`](conformer::ConformerTransformer).
//!
//! [`EncoderTransformer`] is any of the three stacks, chosen by config
//! ([`EncoderTransformerConfig`]): each keeps the frame rate and runs
//! `[batch, frames, d_model]` to the same, padding masked by each item's
//! length, so a model whose stack is a choice holds one field of this type.
//!
//! [`SubsampledEncoder`] is a speech encoder's shape: a front end that
//! changes the frame rate ([`Subsampling`]: feature stacking, or the
//! `FastConformer`'s depthwise striding), then an `EncoderTransformer`, with
//! an [`AxisOrder`](crate::blocks::layout::AxisOrder) bridging the caller's
//! layout to the front end's.

pub mod conformer;
pub mod residual;
pub mod rope;

mod encoder_transformer;
mod subsampled;

pub use encoder_transformer::*;
pub use subsampled::*;
