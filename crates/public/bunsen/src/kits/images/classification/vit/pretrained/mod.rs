//! # Pretrained `ViT` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The
//! providers, the hook and the factory need the `store` and `cache`
//! features, both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are [`ViTContractConfig`](super::ViTContractConfig)'s
//! prefabs: 132 by timm's architecture names, generated from timm's own
//! entrypoints, each with 1000 classes at its architecture's size. The
//! checkpoints are 286 rows, the built-in catalog's builds
//! of [`VIT_CATALOG_CONTRACT`] in its `timm` namespace, written by the
//! catalog builder's `timm-rows`: every timm hub repo of a prefab's
//! architecture (`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`): its
//! `model.safetensors`, or an older repo's `pytorch_model.bin` (three
//! `OpenAI` CLIP towers, in CLIP's own layout), each pinned at a
//! revision: `AugReg`'s, the original `ViT` paper's, SAM's, DINO's, MAE's,
//! MIIL's, the CLIP towers of LAION, `DataComp`, `MetaCLIP`, DFN, `TinyCLIP`
//! and `OpenAI`, with timm's fine-tunes of them, `FlexiViT`'s, EVA's,
//! `DINOv2`'s and its registers', `TIPSv2`'s, timm's sbb sets, the towers of
//! `SigLIP` and `SigLIP` 2 (attention-pooled, and `gap`), C-RADIO's and
//! `AIMv2`'s. A row
//! whose head has an `ImageNet` label set's class count binds those labels (1k,
//! 12k, 21k Google's, 21k MIIL's); the headless towers, the CLIP projections
//! and the iNaturalist heads bind none. A row's license is its card's; a card
//! naming its own license (DFN's Apple license, C-RADIO's NVIDIA one) says
//! `other`, its repo linked as the row's origin.
//!
//! Any timm hub repo of a prefab's architecture
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
