//! # `RegNet` as timm publishes it
//!
//! A timm hub repo's `RegNet` (`timm/regnety_032.ra_in1k`): its `config.json`
//! names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `RegNet` state dict.
//!
//! - **Conversion:** [`regnet_contract_config`], a timm `config.json` as the
//!   [`RegNetContractConfig`](crate::kits::images::classification::regnet::RegNetContractConfig)
//!   it builds.
//! - **Names:** [`regnet_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_regnet_tensors`; with `cache`,
//!   `load_regnet`, a resource set's checkpoint.

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
