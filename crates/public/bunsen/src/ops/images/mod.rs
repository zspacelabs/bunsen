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
//!
//! [`windows`] holds what a windowed vision transformer (Swin) does around
//! its attention: partitioning a token grid into windows and back, the
//! stage's window and shift, its cyclic roll, the shifted-window mask, the
//! relative position index and Swin V2's log-spaced coordinate table, and the
//! `2x2` patch merge's collation.

mod crop;
mod normalize;
pub mod windows;

pub use crop::*;
pub use normalize::*;
