//! # `ResNet`
//!
//! The `ResNet` family of image classifiers
//! ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)), with the
//! checkpoints torchvision and `timm` publish. A residual network is a
//! stem, a run of stages of residual blocks, and a classifier head; the
//! depth of each stage, and whether its blocks are basic or bottleneck,
//! name the variant, `resnet18` to `resnet152`.
//!
//! [`pretrained`] (feature `cache`) turns names into loaded models.
//! [`RESNET_PREFABS`](pretrained::RESNET_PREFABS) names the six
//! geometries, `resnet18` to `resnet152`, and
//! [`default_resnet_factory`](pretrained::default_resnet_factory) is the
//! index of 14 torchvision and `timm` checkpoints: `torchvision/resnet50`,
//! `timm/resnet18_a1`, or a bare `resnet50`.
//!
//! # Example
//!
//! Loading a pretrained model by name, and preparing it for fine-tuning
//! (fetching needs the `fetch` feature):
//!
//! ```rust,no_run
//! # #[cfg(feature = "fetch")] {
//! use std::sync::Arc;
//!
//! use bunsen::{
//!     data::pretrained::{
//!         PretrainedCache,
//!         PretrainedCacheOptions,
//!     },
//!     kits::images::resnet::{
//!         ResNet,
//!         pretrained::default_resnet_factory,
//!     },
//!     support::testing::default_device,
//! };
//! use burn::backend::Flex;
//!
//! let device = default_device();
//! let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
//!
//! // `torchvision/resnet18` names its prefab, which the factory's hook
//! // builds from.
//! let loaded = default_resnet_factory()?.load::<Flex>(
//!     "torchvision/resnet18",
//!     &cache,
//!     &device,
//! )?;
//!
//! let model: ResNet<Flex> = Arc::unwrap_or_clone(loaded.handle)
//!     // re-head the model to 10 classes:
//!     .with_classes(10)
//!     // Enable (drop_block_prob) stochastic block drops for training:
//!     .with_stochastic_drop_block(0.2)
//!     // Enable (drop_path_prob) stochastic depth for training:
//!     .with_stochastic_path_depth(0.1);
//! # }
//! # Ok::<(), bunsen::errors::BunsenError>(())
//! ```
//!
//! # Lifecycle
//!
//! `ResNet` is a [Stacked
//! config](crate::burner::module::ModuleInit#stacked-config):
//!
//! 1. [`ResNetContractConfig`] is the policy: the stage depths, the class
//!    count, the stem width, the output stride, the bottleneck policy, the norm
//!    and the activation.
//! 2. [`to_structure`](crate::burner::module::ToStructureConfig::to_structure)
//!    unrolls it into a [`ResNetStructureConfig`]: the stem's conv block, one
//!    [`LayerBlockStructureConfig`](blocks::LayerBlockStructureConfig) per
//!    stage, each an explicit list of residual blocks, and the class count.
//! 3. `init` builds the [`ResNet`]: from the structure, or from the policy in
//!    one step through the blanket
//!    [`ModuleInit`](crate::burner::module::ModuleInit).
//! 4. [`forward`](ResNet::forward) maps a `[batch, 3, height, width]` image
//!    batch to `[batch, num_classes]` logits.
//!
//! The policy is the simple way to name a variant, and is meant to stay
//! that way as variants are added. The structure is for building one the
//! policy cannot express: edit the tree, then `init` it. [`ResNetMeta`]
//! answers the same questions of the structure and the module, and
//! [`blocks`] has the parts.
//!
//! A built model can still change: [`with_classes`](ResNet::with_classes)
//! re-heads it, and
//! [`with_stochastic_drop_block`](ResNet::with_stochastic_drop_block) and
//! [`with_stochastic_path_depth`](ResNet::with_stochastic_path_depth) add
//! regularization for training.
//!
//! # Compatibility
//!
//! `ResNet` has evolved into a large family of models, and this kit aims
//! for parity with the `timm` library's. The equivalence matrix is a large
//! amount of work and is not complete. Missing today:
//!
//! * all the fancy-stem options
//! * injectable norm layers
//! * injectable activation layers
//! * conv / avg downsample switching
//! * anti-aliasing
//! * block attention

#[cfg(feature = "cache")]
pub mod pretrained;

pub mod blocks;

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
