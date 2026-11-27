//! # Pretrained `ResNetV2` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The
//! providers, the hook and the factory need the `store` and `cache`
//! features, both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are
//! [`ResNetV2ContractConfig`](super::ResNetV2ContractConfig)'s prefabs, by
//! timm's architecture names, generated from timm's own entrypoints, each with
//! 1000 classes. The checkpoints are the built-in catalog's builds of
//! [`RESNETV2_CATALOG_CONTRACT`] in its `timm` namespace, written by the
//! catalog builder's `timm-rows`: every timm hub repo of a prefab's
//! architecture (`timm/resnetv2_50x1_bit.goog_in21k_ft_in1k`), each pinned at
//! a revision: Google's `BiT` towers and their fine-tunes, and timm's own. A
//! row whose head has an `ImageNet` label set's class count binds those
//! labels: the `BiT` `goog_in21k` heads Google's 21k.
//!
//! Any timm hub repo of a prefab's architecture loads as its
//! `model.safetensors` and its `config.json`, whose architecture is the
//! prefab of its name, with the repo's classes and pool
//! ([`resnetv2_contract_config`](super::interop::timm::resnetv2_contract_config)).
//!
//! [`default_resnetv2_factory`] is the index, and [`ResNetV2ConstructConfig`]
//! its hook: it builds the model, reads the checkpoint into it with
//! [`load_resnetv2`](super::interop::timm::load_resnetv2), and reads the
//! build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard).
//! The loaded handle is an
//! [`ImageModelConstruct`](crate::kits::images::support::card::ImageModelConstruct)
//! of a
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel).

#[cfg(feature = "store")]
mod construct;
mod prefabs;
#[cfg(feature = "store")]
mod providers;

#[cfg(feature = "store")]
pub use construct::*;
#[cfg(feature = "store")]
pub use providers::*;
