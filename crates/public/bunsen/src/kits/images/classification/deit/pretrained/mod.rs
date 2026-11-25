//! # Pretrained `DeiT` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The
//! providers, the hook and the factory need the `store` and `cache`
//! features, both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are [`DeiTContractConfig`](super::DeiTContractConfig)'s
//! prefabs: 16 by timm's architecture names, each with 1000 classes at its
//! name's size. The checkpoints are the built-in catalog's builds of
//! [`DEIT_CATALOG_CONTRACT`] in its `timm` namespace, written by the catalog
//! builder's `timm-rows`: every timm hub repo of a prefab's architecture
//! (`timm/deit_base_distilled_patch16_224.fb_in1k`), each pinned at a
//! revision and binding `ImageNet`-1k's labels.
//!
//! Any timm hub repo of a prefab's architecture
//! (`hf:timm/deit_base_patch16_224.fb_in1k`) loads as its
//! `model.safetensors` and its `config.json`, whose architecture is the
//! prefab of its name, with the repo's classes, pool and size
//! ([`deit_contract_config`](super::interop::timm::deit_contract_config)).
//!
//! [`default_deit_factory`] is the index, and [`DeiTConstructConfig`] its
//! hook: it builds the model, reads the checkpoint into it with
//! [`load_deit`](super::interop::timm::load_deit), and reads the build's
//! [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard).
//! The loaded handle is an
//! [`ImageModelConstruct`](crate::kits::images::support::card::ImageModelConstruct)
//! of a
//! [`VisionEncoderModel`](crate::blocks::transformers::models::vision_encoder::VisionEncoderModel).

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
