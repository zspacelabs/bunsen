//! # `SKNet` as timm publishes it
//!
//! A timm hub repo's `SKNet` (`timm/skresnet18.ra_in1k`): its `config.json`
//! names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `ResNet` state dict, its blocks
//! `SelectiveKernelBasic`s or `SelectiveKernelBottleneck`s.
//!
//! - **Conversion:** [`sknet_contract_config`], a timm `config.json` as the
//!   [`SKNetContractConfig`](crate::kits::images::classification::sknet::SKNetContractConfig)
//!   it builds.
//! - **Names:** [`sknet_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_sknet_tensors`; with `cache`,
//!   `load_sknet`, a resource set's checkpoint.

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
