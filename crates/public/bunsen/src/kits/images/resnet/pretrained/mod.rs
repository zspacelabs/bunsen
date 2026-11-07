//! # Pretrained `ResNet` models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The tables,
//! the hook and the factory need the `store` and `cache` features,
//! both default. Fetching a checkpoint needs `fetch`.
//!
//! The geometries are [`RESNET_PREFABS`]: six
//! [`ResNetContractConfig`](super::ResNetContractConfig)s by name,
//! `resnet18` to `resnet152`, each with 1000 classes. The checkpoints are
//! 14 rows in two groups of [`RESNET_CATALOG`]: `torchvision`, the
//! reference `ImageNet` weights, and `timm`, the `ResNet` Strikes Back
//! recipes (`a1`, `a2`, `a3`) and earlier `pytorch-image-models` releases.
//! Each row names the prefab it instantiates, and is one `.pth` file under
//! its group's base URL, pinned to its digest. A spec is
//! `torchvision/resnet50`, `timm/resnet18_a1`, or a bare `resnet50`, which
//! the torchvision group answers first.
//!
//! [`default_resnet_factory`] is the index, and [`ResNetConstructConfig`] its
//! hook: it builds the model from the row's prefab and reads the
//! checkpoint into it with
//! [`ResNet::load_pytorch_weights`](super::ResNet::load_pytorch_weights). A
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
//!     blocks::activation::BunsenActivationConfig,
//!     data::pretrained::{
//!         PretrainedCache,
//!         PretrainedCacheOptions,
//!     },
//!     kits::images::resnet::{
//!         ResNet,
//!         pretrained::{
//!             RESNET_PREFABS,
//!             default_resnet_factory,
//!         },
//!     },
//!     support::testing::cpu_device,
//! };
//!
//! let device = cpu_device();
//! let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
//!
//! let config = RESNET_PREFABS
//!     .expect_lookup_prefab("resnet18")
//!     .to_config()
//!     .with_activation(BunsenActivationConfig::Gelu);
//!
//! let mut model =
//!     default_resnet_factory()?.resolve("timm/resnet18_a1", &cache)?;
//! model.config = model.config.with_config(config);
//! let resnet: ResNet = Arc::unwrap_or_clone(
//!     model
//!         .load(
//!             &cache,
//!             &device,
//!             bunsen::kits::support::construct::LoadPolicy::performance(),
//!         )?
//!         .handle,
//! );
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
pub use prefabs::*;
#[cfg(feature = "store")]
pub use providers::*;
