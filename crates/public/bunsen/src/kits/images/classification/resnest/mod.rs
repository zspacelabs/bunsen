//! # `ResNeSt`
//!
//! `ResNeSt` ([arXiv:2004.08955](https://arxiv.org/abs/2004.08955)), with
//! the checkpoints `timm` publishes: the authors' `ImageNet`-1k weights, and
//! `GluonCV`'s for the 14- and 26-layer nets. A `ResNeSt` is a `ResNet`
//! whose bottlenecks split their middle conv into `radix` attended splits,
//! weighing them per channel by an attention over their sum, and whose
//! striding units downsample by an average pool.
//!
//! [`pretrained`] turns names into loaded models.
//! [`ResNeStContractConfig`]'s prefabs name the geometries, by timm's
//! architecture names, generated from timm's own entrypoints
//! (`tools/gen_resnet_kin_prefabs.py`), and
//! [`default_resnest_factory`](pretrained::default_resnest_factory) is the
//! index of timm's hub repos (`timm/resnest50d.in1k`), and loads any other of
//! the prefabs' architectures as `hf:timm/<repo>`.
//!
//! # The model
//!
//! A `ResNeSt` is the generic conv net,
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel):
//! the kit has no model type of its own. [`ResNeStContractConfig`] is the
//! policy in timm's terms, a
//! [`ResNetContractConfig`](crate::kits::images::classification::resnet::ResNetContractConfig)
//! and the block's `radix`, `avd` and `avd_first`, and lowers to the conv
//! net's, its mixer `ResNeSt`'s split attention
//! ([`SplitAttnConfig`](crate::blocks::images::mixers::SplitAttnConfig)).
//!
//! The model's parameter paths are the conv net's own; [`interop`] maps
//! timm's names onto them ([`resnest_rekey`](interop::timm::resnest_rekey)).

pub mod pretrained;

pub mod interop;

mod contract;

pub use contract::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
