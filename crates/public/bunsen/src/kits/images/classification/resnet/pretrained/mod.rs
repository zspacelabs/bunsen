//! # Pretrained `ResNet` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The tables,
//! the hook and the factory need the `store` and `cache` features,
//! both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are
//! [`ResNetContractConfig`](crate::kits::images::classification::resnet::ResNetContractConfig)'
//! s prefabs: 76
//! [`ResNetContractConfig`](super::ResNetContractConfig)s by timm's
//! architecture names (`resnet18` to `resnet152`, `resnet50_gn`, the deep-
//! and tiered-stem C, D, S and T variants, the `ResNeXt`s, the wide
//! `ResNet`s, the squeeze-excite and ECA nets, the anti-aliased nets, and the
//! `ResNet-RS`s),
//! each with 1000 classes.
//! The checkpoints are 140 rows, the
//! built-in catalog's builds of [`RESNET_CATALOG_CONTRACT`], in two namespaces,
//! each row binding `ImageNet`-1k's labels:
//!
//! - `torchvision`: the reference `ImageNet` weights, five rows. Each states
//!   the prefab geometry it instantiates and its eval transform, and is one
//!   `.pth` file under torchvision's base URL, pinned to its digest.
//! - `timm`: 135 timm hub repos of the prefabs' architectures
//!   (`timm/resnet50.a1_in1k`), written by the catalog builder's `timm-rows`:
//!   the `ResNet` Strikes Back recipes, timm's own, torchvision's (v1 for the
//!   architectures with no `torchvision` row, and v2), Gluon's, and Facebook's
//!   semi- and weakly supervised weights. Each is the repo's
//!   `model.safetensors` and `config.json`, which gives the geometry and the
//!   eval transform, pinned at a revision. The names of earlier rows
//!   (`timm/resnet18_a1`) are aliases of the repos with their weights.
//!
//! A spec is `torchvision/resnet50`, `timm/resnet50.a1_in1k`,
//! `timm/resnet18_a1`, or a bare `resnet50`, which the torchvision group
//! answers first.
//!
//! A timm hub repo the catalog does not name loads as `hf:timm/<repo>`
//! (`hf:timm/resnet50.a1_in1k`): its `model.safetensors`, and its
//! `config.json`, whose architecture is the prefab of its name
//! ([`resnet_contract_config`](super::interop::timm::resnet_contract_config))
//! and whose `pretrained_cfg` is the card's eval transform.
//!
//! [`default_resnet_factory`] is the index, and [`ResNetConstructConfig`] its
//! hook: it builds the model from the row's prefab, reads the checkpoint
//! into it with [`load_resnet`](super::interop::torchvision::load_resnet),
//! and reads the row's
//! [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard):
//! the eval transform it states and the labels it binds (`ImageNet`-1k's,
//! for every row). The loaded handle is an
//! [`ImageModelConstruct`](crate::kits::images::support::card::ImageModelConstruct). A
//! checkpoint does not describe its own geometry, so a model built from
//! any other config, or a checkpoint given as a path, needs
//! [`ResNetConstructConfig::with_config`]: resolve the name, set the config on
//! the deferred model's construct config, then load it. `examples/resnet_tiny`
//! swaps the activation this way before the weights land:
//!
//! ```rust,no_run
//! # #[cfg(feature = "store")] {
//! use std::sync::Arc;
//!
//! use bunsen::{
//!     blocks::{
//!         activation::BunsenActivationConfig,
//!         images::models::conv_net::ConvNetModel,
//!     },
//!     data::{
//!         pretrained::Prefabs,
//!         resources::{
//!             ResourceCache,
//!             ResourceCacheOptions,
//!         },
//!     },
//!     kits::images::classification::resnet::{
//!         ResNetContractConfig,
//!         pretrained::default_resnet_factory,
//!     },
//!     support::testing::cpu_device,
//! };
//!
//! let device = cpu_device();
//! let cache = ResourceCache::new(ResourceCacheOptions::default())?;
//!
//! let config = ResNetContractConfig::expect_prefab("resnet18")
//!     .with_activation(BunsenActivationConfig::Gelu);
//!
//! let mut model =
//!     default_resnet_factory()?.resolve("timm/resnet18_a1", &cache)?;
//! model.config = model.config.with_config(config);
//! let resnet: ConvNetModel = Arc::unwrap_or_clone(
//!     model
//!         .load(
//!             &cache,
//!             &device,
//!             bunsen::data::formats::LoadPolicy::performance(),
//!         )?
//!         .handle,
//! )
//! .model;
//! # }
//! # Ok::<(), bunsen::errors::BunsenError>(())
//! ```

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
