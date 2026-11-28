//! # Swin V2 as timm publishes it
//!
//! A timm hub repo's Swin V2 (`timm/swinv2_tiny_window8_256.ms_in1k`): its
//! `config.json` names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `SwinTransformerV2` state dict.
//!
//! - **Conversion:** [`swin_v2_contract_config`], a timm `config.json` as the
//!   [`SwinV2ContractConfig`](crate::kits::images::classification::swin::v2::SwinV2ContractConfig)
//!   it builds.
//! - **Names and loading:** the family's
//!   [`swin::interop::timm`](crate::kits::images::classification::swin::interop::timm),
//!   V1's and V2's alike.

mod config;

pub use config::*;

#[cfg(test)]
mod tests;
