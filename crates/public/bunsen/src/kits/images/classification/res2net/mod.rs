//! # `Res2Net`
//!
//! `Res2Net` ([arXiv:1904.01169](https://arxiv.org/abs/1904.01169)) and
//! `Res2NeXt`, with the checkpoints `timm` publishes: the authors'
//! `ImageNet`-1k weights. A `Res2Net` is a `ResNet` whose bottlenecks split
//! their middle channels into `scale` chunks and convolve them in a
//! hierarchy, each branch seeing the last one's output, so one block mixes
//! at several receptive fields.
//!
//! [`pretrained`] turns names into loaded models.
//! [`Res2NetContractConfig`]'s prefabs name the geometries, by timm's
//! architecture names, generated from timm's own entrypoints
//! (`tools/gen_resnet_kin_prefabs.py`), and
//! [`default_res2net_factory`](pretrained::default_res2net_factory) is the
//! index of timm's hub repos (`timm/res2net50_26w_4s.in1k`), and loads any
//! other of the prefabs' architectures as `hf:timm/<repo>`.
//!
//! # The model
//!
//! A `Res2Net` is the generic conv net,
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel):
//! the kit has no model type of its own. [`Res2NetContractConfig`] is the
//! policy in timm's terms, a
//! [`ResNetContractConfig`](crate::kits::images::classification::resnet::ResNetContractConfig)
//! and the `scale`, and lowers to the conv net's, its mixer `Res2Net`'s scale
//! split ([`Res2ConvConfig`](crate::blocks::images::mixers::Res2ConvConfig)).
//!
//! The model's parameter paths are the conv net's own; [`interop`] maps
//! timm's names onto them ([`res2net_rekey`](interop::timm::res2net_rekey)).

pub mod pretrained;

pub mod interop;

mod contract;

pub use contract::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
