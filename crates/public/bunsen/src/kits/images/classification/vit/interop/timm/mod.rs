//! # `ViT` as timm publishes it
//!
//! A timm hub repo's `ViT` (`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`):
//! its `config.json` names the architecture, which is one of the kit's
//! prefabs, and its `model.safetensors` holds timm's `VisionTransformer`
//! state dict.
//!
//! - **Conversion:** [`vit_contract_config`], a timm `config.json` as the
//!   [`ViTContractConfig`](crate::kits::images::classification::vit::ViTContractConfig)
//!   it builds.
//! - **Names:** [`vit_row_splits`] (the fused `qkv`, split) and [`vit_rekey`],
//!   onto the model's canonical names; for a gated MLP's model,
//!   [`vit_gated_row_splits`] and [`vit_gated_rekey`].
//! - **Loading** (feature `store`): `load_vit_tensors`; with `cache`,
//!   `load_vit`, a resource set's checkpoint into a model it builds, and
//!   `load_vit_into`, into one a family built.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;
mod rekey;

pub use config::*;
pub use rekey::*;

#[cfg(test)]
mod tests;
