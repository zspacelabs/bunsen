//! # `ResNet` as torchvision saves it
//!
//! torchvision's `ResNet` state dict (a `.pt`; `timm`'s checkpoints use the
//! same names) as bunsen's
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel).
//! A checkpoint does not describe its own geometry: the catalog row's prefab
//! does.
//!
//! - **Names:** [`resnet_rekey`], for a model's [`ResNetStateLayout`].
//! - **Loading** (feature `store`): `load_resnet_tensors`; with `cache`,
//!   `load_resnet`, a resource set's checkpoint.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod rekey;

pub use rekey::*;

#[cfg(test)]
mod tests;
