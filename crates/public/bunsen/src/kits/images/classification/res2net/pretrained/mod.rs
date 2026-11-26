//! # Pretrained `Res2Net` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The
//! providers, the hook and the factory need the `store` and `cache`
//! features, both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are
//! [`Res2NetContractConfig`](super::Res2NetContractConfig)'s prefabs, by
//! timm's architecture names, generated from timm's own entrypoints, each with
//! 1000 classes. The checkpoints are the built-in catalog's builds of
//! [`RES2NET_CATALOG_CONTRACT`] in its `timm` namespace, written by the
//! catalog builder's `timm-rows`: every timm hub repo of a prefab's
//! architecture (`timm/res2net50_26w_4s.in1k`), each pinned at a revision:
//! the `Res2Net` authors' `ImageNet`-1k weights, binding its labels. Their
//! cards state the license as `unknown`, and the rows say so.
//!
//! Any timm hub repo of a prefab's architecture loads as its
//! `model.safetensors` and its `config.json`, whose architecture is the
//! prefab of its name, with the repo's classes and pool
//! ([`res2net_contract_config`](super::interop::timm::res2net_contract_config)).
//!
//! [`default_res2net_factory`] is the index, and [`Res2NetConstructConfig`]
//! its hook: it builds the model, reads the checkpoint into it with
//! [`load_res2net`](super::interop::timm::load_res2net), and reads the
//! build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard).
//! The loaded handle is an
//! [`ImageModelConstruct`](crate::kits::images::support::card::ImageModelConstruct)
//! of a
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel).

#[cfg(feature = "store")]
mod construct;
#[cfg(feature = "store")]
mod factory;
mod prefabs;
#[cfg(feature = "store")]
mod providers;

#[cfg(feature = "store")]
pub use construct::*;
#[cfg(feature = "store")]
pub use factory::*;
#[cfg(feature = "store")]
pub use providers::*;
