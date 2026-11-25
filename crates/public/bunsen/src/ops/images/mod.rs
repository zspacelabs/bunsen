//! # Image tensor operations
//!
//! The tensor half of an image model's preprocessing, on a
//! `[batch, channels, height, width]` batch of pixel values: cropping and
//! padding to a size ([`center_crop`], [`center_crop_or_pad`]) and
//! per-channel normalisation ([`normalize`]), as torchvision computes them.
//! Decoding and resizing an image file need the `image` crate, and live in
//! `support::images` (feature `images`); the size arithmetic of a model's
//! eval transform is
//! [`ImageEvalTransformConfig`](crate::kits::images::support::transform::ImageEvalTransformConfig)'s.

mod crop;
mod normalize;

pub use crop::*;
pub use normalize::*;
