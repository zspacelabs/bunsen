//! # `ResNet` as timm publishes it
//!
//! A timm hub repo's `ResNet` (`timm/resnet50.a1_in1k`): its `config.json`
//! names the architecture, which is one of the kit's prefabs, and its
//! `model.safetensors` uses torchvision's names
//! ([`resnet_rekey`](super::torchvision::resnet_rekey)).
//!
//! - **Conversion:** [`resnet_contract_config`], a timm `config.json` as the
//!   [`ResNetContractConfig`](crate::kits::images::classification::resnet::ResNetContractConfig)
//!   it builds.
//! - **Loading** (feature `store`): `load_resnet`, a resource set's safetensors
//!   checkpoint.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;

pub use config::*;

#[cfg(test)]
mod tests;
