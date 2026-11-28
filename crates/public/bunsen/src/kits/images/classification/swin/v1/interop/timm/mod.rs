//! # Swin V1 as timm publishes it
//!
//! A timm hub repo's Swin V1 (`timm/swin_tiny_patch4_window7_224.ms_in1k`): its
//! `config.json` names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `SwinTransformer` state dict.
//!
//! - **Conversion:** [`swin_v1_contract_config`], a timm `config.json` as the
//!   [`SwinV1ContractConfig`](crate::kits::images::classification::swin::v1::SwinV1ContractConfig)
//!   it builds.
//! - **Names and loading:** the family's
//!   [`swin::interop::timm`](crate::kits::images::classification::swin::interop::timm),
//!   V1's and V2's alike.

mod config;

pub use config::*;

#[cfg(test)]
mod tests;
