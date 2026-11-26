//! # `ResNetV2`
//!
//! The pre-activation `ResNet` (`ResNetV2`,
//! [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)) and Big Transfer
//! (`BiT`, [arXiv:1912.11370](https://arxiv.org/abs/1912.11370)), with the
//! checkpoints `timm` publishes: Google's `BiT` towers (`ImageNet`-21k, and
//! their 1k fine-tunes at 448-480) and timm's own `ResNetV2`s. A `ResNetV2`
//! normalizes and activates each unit's input before its convs, not after;
//! `BiT` adds weight standardization to every conv and group norms.
//!
//! [`pretrained`] turns names into loaded models.
//! [`ResNetV2ContractConfig`]'s prefabs name the geometries, by timm's
//! architecture names, generated from timm's own entrypoints
//! (`tools/gen_resnetv2_prefabs.py`), and
//! [`default_resnetv2_factory`](pretrained::default_resnetv2_factory) is the
//! index of timm's hub repos (`timm/resnetv2_50x1_bit.goog_in21k_ft_in1k`),
//! and loads any other of the prefabs' architectures as `hf:timm/<repo>`.
//!
//! # The model
//!
//! A `ResNetV2` is the generic conv net,
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel):
//! the kit has no model type of its own. [`ResNetV2ContractConfig`] is the
//! policy in timm's terms (the depths and widths, the width factor, the stem,
//! the conv's weight standardization, the norm, basic or bottleneck units, the
//! shortcut's pool) and lowers to the conv net's.
//!
//! The model's parameter paths are the conv net's own; [`interop`] maps
//! timm's names onto them
//! ([`resnetv2_rekey`](interop::timm::resnetv2_rekey)).
//!
//! # Compatibility
//!
//! The kit aims for parity with `timm`'s `resnetv2.py`. Missing today:
//!
//! * the non-pre-activation `ResNetV2` (`preact=False`, `stem_type='same'`),
//!   the hybrid `ViT`s' backbone, which has no hub weights of its own

pub mod pretrained;

pub mod interop;

mod contract;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

pub use contract::*;
