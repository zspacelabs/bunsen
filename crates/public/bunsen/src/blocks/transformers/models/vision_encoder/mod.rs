//! # The generic vision encoder
//!
//! [`VisionEncoderModel`]: a [`PatchEmbed`](crate::blocks::images::patching::patch_embed::PatchEmbed),
//! [`PrefixTokens`](crate::blocks::images::patching::prefix_tokens::PrefixTokens)
//! (the class and register tokens, and the learnt position embedding), a
//! bidirectional [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack),
//! and a [`VisionHead`]. It is the one model type of the `ViT` families,
//! timm's `VisionTransformer`: a `ViT` is a `VisionEncoderModel`, built from
//! its family's policy.
//!
//! ## The forward
//!
//! As timm's `forward_features` and `forward_head`:
//!
//! 1. the patch embedding cuts the image into patches, one token each;
//! 2. the prefix tokens go before them, and the position embedding is added
//!    (over the prefix too, or the patches alone);
//! 3. the transformer's input norm (timm's `norm_pre`, CLIP's), its pre-norm
//!    layers (attention over every token, then an MLP; layer scale on each),
//!    and its final norm;
//! 4. the head's pool (the class token, or a reduction of the patch tokens),
//!    its norm (timm's `fc_norm`, which replaces the final norm), and the
//!    classifier, none on a headless model.
//!
//! ## Parameter paths
//!
//! The paths are the model's own; each family's interop rekeys its
//! checkpoints onto them.
//!
//! | part | path |
//! |---|---|
//! | the patch embedding's conv | `patch_embed.projection` |
//! | the class and register tokens, the position embedding | `tokens.cls_token`, `tokens.reg_token`, `tokens.pos_embed` |
//! | the transformer | `stack.*`, [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)'s |
//! | the head's norm and classifier | `head.fc_norm`, `head.fc` |

mod head;
mod model;

pub use head::*;
pub use model::*;

#[cfg(test)]
mod tests;
