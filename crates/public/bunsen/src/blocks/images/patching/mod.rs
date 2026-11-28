//! # Patch embedding
//!
//! [`patch_embed::PatchEmbed`] cuts a `[batch, channels, height, width]`
//! image into square patches and embeds each one as a token: the first layer
//! of the vision transformers, plain and staged.
//! [`prefix_tokens::PrefixTokens`] puts a vision transformer's learnt tokens
//! before the patch tokens (class, register) and adds the learnt position
//! embedding. [`patch_merge::PatchMerge`] is a staged vision transformer's
//! downsample between stages.
//!
//! ## Discussion
//!
//! The patch embedding is the first layer of the `ViT`s and of the Swin
//! Transformer, the prefix tokens are a `ViT`'s, and the patch merge is Swin's
//! `PatchMerging`.

pub mod patch_embed;
pub mod patch_merge;
pub mod prefix_tokens;
