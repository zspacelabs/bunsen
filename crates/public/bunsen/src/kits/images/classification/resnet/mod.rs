//! # `ResNet`
//!
//! The `ResNet` family of image classifiers
//! ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)), with the
//! checkpoints torchvision and `timm` publish. A residual network is a
//! stem, a run of stages of residual blocks, and a classifier head; the
//! depth of each stage, its width, whether its blocks are basic or
//! bottleneck, and the norm name the variant, `resnet18` to `resnet152` and
//! `resnet50_gn`.
//!
//! [`pretrained`] turns names into loaded models.
//! [`ResNetContractConfig`]'s prefabs name the seven geometries, by timm's
//! architecture names, and
//! [`default_resnet_factory`](pretrained::default_resnet_factory) is the
//! index of 48 torchvision and `timm` checkpoints: `torchvision/resnet50`,
//! `timm/resnet50.a1_in1k`, or a bare `resnet50`.
//!
//! # Example
//!
//! Loading a pretrained model by name, and preparing it for fine-tuning
//! (fetching needs the `fetch` feature):
//!
//! ```rust,no_run
//! # #[cfg(all(feature = "store", feature = "fetch"))] {
//! use std::sync::Arc;
//!
//! use bunsen::{
//!     blocks::images::models::conv_net::ConvNetModel,
//!     data::resources::{
//!         ResourceCache,
//!         ResourceCacheOptions,
//!     },
//!     kits::images::classification::resnet::pretrained::default_resnet_factory,
//!     support::testing::cpu_device,
//! };
//!
//! let device = cpu_device();
//! let cache = ResourceCache::new(ResourceCacheOptions::default())?;
//!
//! // `torchvision/resnet18` names its prefab, which the factory's hook
//! // builds from.
//! let loaded = default_resnet_factory()?.load(
//!     "torchvision/resnet18",
//!     &cache,
//!     &device,
//!     bunsen::data::formats::LoadPolicy::performance(),
//! )?;
//!
//! let model: ConvNetModel = Arc::unwrap_or_clone(loaded.handle)
//!     .model
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
//! # The model
//!
//! A `ResNet` is the generic conv net,
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel):
//! the kit has no model type of its own. [`ResNetContractConfig`] is the
//! policy in `ResNet`'s terms, its knobs timm's by their meaning (the stage
//! depths and widths, the classes, the input's channels, the stem's width,
//! the output stride, the bottleneck policy, the norm, a group norm
//! included, and the activation), and lowers to the generic one
//! ([`try_to_conv_net`](ResNetContractConfig::try_to_conv_net)); `init` builds the
//! model through it. A geometry the policy cannot express is built from the
//! generic policy or structure directly.
//!
//! The model's parameter paths are the conv net's own; [`interop`] maps
//! torchvision's and timm's names onto them
//! ([`resnet_rekey`](interop::torchvision::resnet_rekey)).
//!
//! # Compatibility
//!
//! `ResNet` has evolved into a large family of models, and this kit aims
//! for parity with the `timm` library's. The equivalence matrix is a large
//! amount of work and is not complete. Missing today:
//!
//! * the deep and tiered stems
//! * the average-pool shortcut (`ResNet-D`)
//! * grouped and widened units (`ResNeXt`, wide `ResNet`s) on the policy
//! * anti-aliasing
//! * channel attention

pub mod pretrained;

pub mod interop;

mod contract;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

pub use contract::*;
