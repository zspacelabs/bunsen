//! # `timm`: the timm ecosystem, for any image kit
//!
//! The image kits' shared tier for timm, below each kit's own
//! `interop::timm`:
//!
//! - [`interop`]: a timm hub repo's `config.json`, its architecture and its
//!   `pretrained_cfg`
//!   ([`TimmModelConfigMirror`](interop::TimmModelConfigMirror)), and the eval
//!   transform it states.

pub mod interop;
