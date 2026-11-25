//! # `ViT` interop
//!
//! - [`timm`]: timm's hub repos, the `ViT`s' checkpoints: the contract from a
//!   `config.json`, and the state dict's names onto the model's.
//! - [`openai_clip`]: `OpenAI`'s CLIP checkpoints, whose image tower some of
//!   timm's `openai` repos ship in CLIP's own layout.

pub mod openai_clip;
pub mod timm;
