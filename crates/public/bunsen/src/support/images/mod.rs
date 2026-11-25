//! # Image Support
//!
//! Image files, through the [`image`] crate (feature `images`): decoding
//! ([`try_open_image`]), resizing with one of `image`'s filters
//! ([`resize_image_exact`]), and pixels as an `f32` tensor
//! ([`image_to_f32_tensor`]).
//!
//! An image model's eval transform builds on these in the image kits, in
//! `kits::images::support::eval` (features `kits` and `images`).

mod decode;
mod resize;

pub use decode::*;
pub use resize::*;
