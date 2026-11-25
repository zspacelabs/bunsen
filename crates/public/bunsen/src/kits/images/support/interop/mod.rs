//! # Interop: what an external ecosystem's image-model files mean, for any image kit
//!
//! The image kits' shared tier of interop, below each kit's own
//! `interop::{ecosystem}` and above the crate-wide
//! [`kits::support::interop`](crate::kits::support::interop):
//!
//! - [`timm`]: a timm hub repo's `config.json`, its architecture and its
//!   `pretrained_cfg` ([`TimmModelConfigMirror`](timm::TimmModelConfigMirror)),
//!   and the eval transform it states.

pub mod timm;
