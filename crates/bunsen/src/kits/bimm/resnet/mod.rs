//! # `ResNet`
//!
//! Implementation of the *`ResNet`* family of models for image recognition.
//! See: [arXiv:1512.03385v1 [cs.CV]](<https://arxiv.org/abs/1512.03385>)
//!
//! # Examples
//!
//! Examples of loading pretrained model:
//!
//! ```rust,no_run
//! use bunsen::{
//!     burner::module::ModuleInit,
//!     data::cache::BunsenDiskCache,
//!     kits::bimm::resnet::{
//!         PREFAB_RESNET_MAP,
//!         ResNet,
//!     },
//! };
//! use burn::backend::Flex;
//!
//! let device = Default::default();
//!
//! let prefab = PREFAB_RESNET_MAP.expect_lookup_prefab("resnet18");
//!
//! let weights = prefab
//!     .expect_lookup_pretrained_weights("tv_in1k")
//!     .fetch_weights(&mut BunsenDiskCache::default())
//!     .expect("Failed to fetch weights");
//!
//! let model: ResNet<Flex> = prefab
//!     .to_config()
//!     .init(&device)
//!     .load_pytorch_weights(weights)
//!     .expect("Failed to load weights")
//!     // re-head the model to 10 classes:
//!     .with_classes(10)
//!     // Enable (drop_block_prob) stochastic block drops for training:
//!     .with_stochastic_drop_block(0.2)
//!     // Enable (drop_path_prob) stochastic depth for training:
//!     .with_stochastic_path_depth(0.1);
//! ```
//!
//! ## Configuration
//!
//! This module uses 2-layer configuration.
//!
//! * [`ResNetContractConfig`]
//! * [`ResNetStructureConfig`]
//!
//! The high-level abstract config describes `ResNet` modules semantically,
//! while the low-level config describes the structural config.
//!
//! As the compatibility matrix grows; we may differentiate the abstract configs
//! for different variants of `ResNet`; but the goal is that for any given
//! family, there should be a simple to configure abstract config.
//!
//! The structural config is focused on the actual structure of the model,
//! and the intended user base is people working on variants of `ResNet`.
//!
//! It should continue to get cheaper to build `ResNet` variants.
//!
//! ## Compatibility
//!
//! `ResNet` has evolved into a large family of models, and this crate aims
//! to evolve towards equity with the ``timm`` library of models.
//!
//! Unfortunately, the equivalence matrix itself represents a large
//! amount of work and is not yet complete.
//!
//! An incomplete list of missing features includes:
//! * all the fancy-stem options
//! * injectable norm layers
//! * injectable activation layers
//! * conv / avg downsample switching
//! * anti-aliasing
//! * block attention

#[cfg(feature = "cache")]
mod pretrained;
#[cfg(feature = "cache")]
pub use pretrained::*;

pub mod blocks;

#[doc(inline)]
pub use blocks::resnet_model::*;

/// ResNet-18 block depths.
pub const RESNET18_BLOCKS: [usize; 4] = [2, 2, 2, 2];
/// ResNet-34 block depths.
pub const RESNET34_BLOCKS: [usize; 4] = [3, 4, 6, 3];
/// ResNet-50 block depths.
pub const RESNET50_BLOCKS: [usize; 4] = [3, 4, 6, 3];
/// ResNet-101 block depths.
pub const RESNET101_BLOCKS: [usize; 4] = [3, 4, 23, 3];
/// ResNet-152 block depths.
pub const RESNET152_BLOCKS: [usize; 4] = [3, 8, 36, 3];
