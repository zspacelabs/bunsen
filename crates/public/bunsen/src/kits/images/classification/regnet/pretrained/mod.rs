//! # Pretrained `RegNet` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The
//! providers, the hook and the factory need the `store` and `cache`
//! features, both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are [`RegNetContractConfig`](super::RegNetContractConfig)'s
//! prefabs, by timm's architecture names, generated from timm's own configs,
//! each with 1000 classes. The checkpoints are the built-in catalog's builds
//! of [`REGNET_CATALOG_CONTRACT`] in its `timm` namespace, written by the
//! catalog builder's `timm-rows`: every timm hub repo of a prefab's
//! architecture (`timm/regnety_032.ra_in1k`), each pinned at a revision:
//! pycls's, torchvision's v2, timm's own (`RegNetV` and `RegNetZ` among
//! them), `SEER`'s and `SWAG`'s. A row whose
//! head has an `ImageNet` label set's class count binds those labels; the
//! headless `SEER` trunks bind none. A row's license is its card's: `SWAG`'s
//! are non-commercial, `SEER`'s under its own license.
//!
//! Any timm hub repo of a prefab's architecture
//! (`hf:timm/regnety_032.ra_in1k`) loads as its `model.safetensors` and its
//! `config.json`, whose architecture is the prefab of its name, with the
//! repo's classes and pool
//! ([`regnet_contract_config`](super::interop::timm::regnet_contract_config)).
//!
//! [`default_regnet_factory`] is the index, and [`RegNetConstructConfig`] its
//! hook: it builds the model, reads the checkpoint into it with
//! [`load_regnet`](super::interop::timm::load_regnet), and reads the build's
//! [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard).
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
