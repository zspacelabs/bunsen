//! # `DeiT` interop
//!
//! [`timm`]: timm's hub repos, the `DeiT`s' checkpoints: the contract from a
//! `config.json`. Their state dicts are a `ViT`'s, the distilled ones' with
//! a `dist_token` and a `head_dist`, which
//! [`load_vit_tensors`](crate::kits::images::classification::vit::interop::timm::load_vit_tensors)
//! reads.

pub mod timm;
