//! # Pretrained `ResNeSt` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The
//! providers, the hook and the factory need the `store` and `cache`
//! features, both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are
//! [`ResNeStContractConfig`](super::ResNeStContractConfig)'s prefabs, by
//! timm's architecture names, generated from timm's own entrypoints, each with
//! 1000 classes. The checkpoints are the built-in catalog's builds of
//! [`RESNEST_CATALOG_CONTRACT`] in its `timm` namespace, written by the
//! catalog builder's `timm-rows`: every timm hub repo of a prefab's
//! architecture (`timm/resnest50d.in1k`), each pinned at a revision: the
//! `ResNeSt` authors' and `GluonCV`'s `ImageNet`-1k weights, binding its
//! labels.
//!
//! Any timm hub repo of a prefab's architecture loads as its
//! `model.safetensors` and its `config.json`, whose architecture is the
//! prefab of its name, with the repo's classes and pool
//! ([`resnest_contract_config`](super::interop::timm::resnest_contract_config)).
//!
//! [`default_resnest_factory`] is the index, and [`ResNeStConstructConfig`]
//! its hook: it builds the model, reads the checkpoint into it with
//! [`load_resnest`](super::interop::timm::load_resnest), and reads the
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
