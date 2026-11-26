//! # Pretrained Swin V2 models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The
//! providers, the hook and the factory need the `store` and `cache`
//! features, both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are [`SwinV2ContractConfig`](super::SwinV2ContractConfig)'s
//! prefabs: 12 by timm's architecture names, each with 1000 classes at its
//! name's size. The checkpoints are the built-in catalog's builds of
//! [`SWIN_V2_CATALOG_CONTRACT`] in its `timm` namespace, written by the
//! catalog builder's `timm-rows`: every timm hub repo of a prefab's
//! architecture (`timm/swinv2_tiny_window8_256.ms_in1k`), each pinned at a
//! revision: 12, the `ImageNet`-1k fine-tunes binding its labels, and the two
//! `ImageNet`-22k pretrainings (`swinv2_base_window12_192.ms_in22k`) none,
//! their 21841 classes no label set of the catalog's.
//!
//! Any timm hub repo of a prefab's architecture
//! (`hf:timm/swinv2_tiny_window8_256.ms_in1k`) loads as its
//! `model.safetensors` and its `config.json`, whose architecture is the
//! prefab of its name, with the repo's classes, pool and size
//! ([`swin_v2_contract_config`](super::interop::timm::swin_v2_contract_config)).
//!
//! [`default_swin_v2_factory`] is the index, and [`SwinV2ConstructConfig`]
//! its hook: it builds the model, reads the checkpoint into it with
//! [`load_swin`](super::super::interop::timm::load_swin), and reads the
//! build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard).
//! The loaded handle is an
//! [`ImageModelConstruct`](crate::kits::images::support::card::ImageModelConstruct)
//! of a
//! [`StagedVisionEncoderModel`](crate::blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel).

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
