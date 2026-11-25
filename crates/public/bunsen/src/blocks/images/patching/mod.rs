//! # Patch embedding
//!
//! [`patch_embed::PatchEmbed`] cuts a `[batch, channels, height, width]`
//! image into square patches and embeds each one as a token: the first layer
//! of the vision transformers (the `ViT`s' and the Swin Transformer's).
//! [`prefix_tokens::PrefixTokens`] puts a `ViT`'s learnt tokens before the
//! patch tokens (class, register) and adds the learnt position embedding.

pub mod patch_embed;
pub mod prefix_tokens;
