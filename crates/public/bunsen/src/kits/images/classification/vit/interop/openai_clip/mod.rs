//! # `ViT` as `OpenAI` CLIP saves it
//!
//! `OpenAI`'s CLIP checkpoints hold the whole model: the image tower under
//! `visual.` in CLIP's own names, the text tower beside it. timm's `openai`
//! hub repos (`timm/vit_base_patch32_clip_224.openai`) ship it as their
//! `pytorch_model.bin`, and timm's `_convert_openai_clip` reads it. The
//! image tower is a pre-norm `ViT` whose head is CLIP's image projection.
//!
//! - **Names:** [`is_openai_clip`] tells the layout apart;
//!   [`openai_clip_row_splits`] (the fused `in_proj`, split) and
//!   [`openai_clip_rekey`] map it onto timm's names, which
//!   [`vit_rekey`](super::timm::vit_rekey) maps on.
//! - **Loading** (feature `store`): `load_openai_clip_tensors`, which
//!   [`load_vit_tensors`](super::timm::load_vit_tensors) takes this layout to.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod rekey;

pub use rekey::*;

#[cfg(test)]
mod tests;
