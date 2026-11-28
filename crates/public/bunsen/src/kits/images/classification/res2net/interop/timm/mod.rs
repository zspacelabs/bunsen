//! # `Res2Net` as timm publishes it
//!
//! A timm hub repo's `Res2Net` (`timm/res2net50_26w_4s.in1k`): its
//! `config.json` names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `ResNet` state dict, its blocks
//! `Bottle2neck`s.
//!
//! - **Conversion:** [`res2net_contract_config`], a timm `config.json` as the
//!   [`Res2NetContractConfig`](crate::kits::images::classification::res2net::Res2NetContractConfig)
//!   it builds.
//! - **Names:** [`res2net_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_res2net_tensors`; with `cache`,
//!   `load_res2net`, a resource set's checkpoint.

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
