//! # Patch embedding
//!
//! [`patch_embed::PatchEmbed`] cuts a `[batch, channels, height, width]`
//! image into square patches and embeds each one as a token, as the Swin
//! Transformer V2 kit's first layer.

pub mod patch_embed;
