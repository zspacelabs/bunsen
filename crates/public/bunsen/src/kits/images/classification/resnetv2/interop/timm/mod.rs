//! # `ResNetV2` as timm publishes it
//!
//! A timm hub repo's `ResNetV2` (`timm/resnetv2_50x1_bit.goog_in21k_ft_in1k`):
//! its `config.json` names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `ResNetV2` state dict.
//!
//! - **Conversion:** [`resnetv2_contract_config`], a timm `config.json` as the
//!   [`ResNetV2ContractConfig`](crate::kits::images::classification::resnetv2::ResNetV2ContractConfig)
//!   it builds.
//! - **Names:** [`resnetv2_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_resnetv2_tensors`; with `cache`,
//!   `load_resnetv2`, a resource set's checkpoint.

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
