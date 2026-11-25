//! # The generic staged vision encoder
//!
//! [`StagedVisionEncoderModel`]: a hierarchical vision transformer, timm's
//! Swin family's shape. A
//! [`PatchEmbed`](crate::blocks::images::patching::patch_embed::PatchEmbed)
//! (with its norm) cuts the image into a grid of tokens; [`VisionStage`]s
//! follow, each an optional
//! [`PatchMerge`](crate::blocks::images::patching::patch_merge::PatchMerge)
//! that halves the grid and widens the tokens, then a
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! of blocks over the stage's grid; then a final norm, a
//! [`TokenPool`](crate::blocks::transformers::models::vision_encoder::TokenPool)
//! over the grid, and the image models'
//! [`ClassifierHead`](crate::blocks::images::models::classifier_head::ClassifierHead).
//!
//! A Swin block is a
//! [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)
//! whose token mixer is
//! [`WindowAttention`](crate::blocks::transformers::attention::window::WindowAttention)
//! (`TokenMixer::Window`): Swin V1's pre-norm and V2's res-post-norm are the
//! block's norm slots. Each family's kit holds no model type of its own: a
//! Swin is a `StagedVisionEncoderModel`, built from its policy.
//!
//! ## Parameter paths
//!
//! | part | path |
//! |---|---|
//! | the patch embedding | `patch_embed.projection`, `patch_embed.norm` |
//! | stage `i`'s merge | `stages.{i}.downsample.reduction`, `….norm` |
//! | stage `i`'s blocks | `stages.{i}.stack.*`, [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)'s |
//! | the final norm | `norm` |
//! | the head | `head.fc`, `head.fc_norm` |

mod model;

pub use model::*;

#[cfg(test)]
mod tests;
