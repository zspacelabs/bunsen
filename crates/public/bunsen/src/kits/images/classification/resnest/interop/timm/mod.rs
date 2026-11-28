//! # `ResNeSt` as timm publishes it
//!
//! A timm hub repo's `ResNeSt` (`timm/resnest50d.in1k`): its `config.json`
//! names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `ResNet` state dict, its blocks
//! `ResNestBottleneck`s.
//!
//! - **Conversion:** [`resnest_contract_config`], a timm `config.json` as the
//!   [`ResNeStContractConfig`](crate::kits::images::classification::resnest::ResNeStContractConfig)
//!   it builds.
//! - **Names:** [`resnest_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_resnest_tensors`; with `cache`,
//!   `load_resnest`, a resource set's checkpoint.

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
