//! # `DeiT` as timm publishes it
//!
//! A timm hub repo's `DeiT` (`timm/deit_base_distilled_patch16_224.fb_in1k`):
//! its `config.json` names the architecture, which is one of the kit's
//! prefabs, and its `model.safetensors` holds timm's `VisionTransformer`
//! state dict, or its `VisionTransformerDistilled`'s.
//!
//! - **Conversion:** [`deit_contract_config`], a timm `config.json` as the
//!   [`DeiTContractConfig`](crate::kits::images::classification::deit::DeiTContractConfig)
//!   it builds.
//! - **Loading** (feature `store`): `load_deit`, a resource set's checkpoint,
//!   through the vit kit's
//!   [`load_vit_tensors`](crate::kits::images::classification::vit::interop::timm::load_vit_tensors).

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;

pub use config::*;

#[cfg(test)]
mod tests;
