//! # Pretrained `ViT` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The
//! providers, the hook and the factory need the `store` and `cache`
//! features, both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are [`ViTContractConfig`](super::ViTContractConfig)'s
//! prefabs: 25 by timm's architecture names, each with 1000 classes at its
//! name's size. A timm hub repo
//! (`hf:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`) loads as its
//! `model.safetensors` and its `config.json`, whose architecture is the
//! prefab of its name, with the repo's classes (0 for a headless tower),
//! pool and size ([`vit_contract_config`](super::interop::timm::vit_contract_config)),
//! and whose `pretrained_cfg` is the card's eval transform.
//!
//! [`default_vit_factory`] is the index, and [`ViTConstructConfig`] its hook:
//! it builds the model, reads the checkpoint into it with
//! [`load_vit`](super::interop::timm::load_vit), and reads the build's
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
