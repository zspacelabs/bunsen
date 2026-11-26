//! # `SKNet`
//!
//! Selective Kernel Networks
//! ([arXiv:1903.06586](https://arxiv.org/abs/1903.06586)), as timm builds
//! them (`skresnet18`, `skresnet34`, `skresnext50_32x4d`), with timm's own
//! `ImageNet`-1k weights. An `SKNet` is a `ResNet` whose spatial mixer is two
//! paths of two receptive fields, weighed per channel by an attention over
//! their sum, so each unit selects its kernel size.
//!
//! [`pretrained`] turns names into loaded models.
//! [`SKNetContractConfig`]'s prefabs name the geometries, by timm's
//! architecture names, generated from timm's own entrypoints
//! (`tools/gen_resnet_kin_prefabs.py`), and
//! [`default_sknet_factory`](pretrained::default_sknet_factory) is the index
//! of timm's hub repos (`timm/skresnet18.ra_in1k`), and loads any other of the
//! prefabs' architectures as `hf:timm/<repo>`.
//!
//! # The model
//!
//! An `SKNet` is the generic conv net,
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel):
//! the kit has no model type of its own. [`SKNetContractConfig`] is the
//! policy in timm's terms, a
//! [`ResNetContractConfig`](crate::kits::images::classification::resnet::ResNetContractConfig)
//! and the selective kernel's knobs, and lowers to the conv net's, its mixer
//! the selective kernel
//! ([`SelectiveKernelConfig`](crate::blocks::images::mixers::SelectiveKernelConfig)).
//!
//! The model's parameter paths are the conv net's own; [`interop`] maps
//! timm's names onto them ([`sknet_rekey`](interop::timm::sknet_rekey)).

pub mod pretrained;

pub mod interop;

mod contract;

pub use contract::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
