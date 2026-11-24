//! # `ResNet`'s interop: its checkpoints' ecosystems
//!
//! - [`torchvision`]: torchvision's `ResNet` state dict, whose names `timm`'s
//!   checkpoints share.
//! - [`timm`]: a timm hub repo: its `config.json`'s architecture as a prefab,
//!   and its safetensors checkpoint.

pub mod timm;
pub mod torchvision;
