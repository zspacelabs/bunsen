//! # Image augmentation operators
//!
//! Augmentation is expressed as a sequence of [`AugmentationStage`]s — flips
//! ([`orientation`]), [`noise`], and control-flow combinators ([`control`],
//! e.g. apply-with-probability or choose-one). Stages register themselves with
//! [`define_image_aug_plugin!`](crate::define_image_aug_plugin) so they can be
//! reconstructed from a serialized [`AugmentationStageConfig`].
//!
//! At pipeline scale, an [`AugmentImageOperation`] (the `AUGMENT_IMAGE`
//! operator) runs a stage list against a `source` image column using a per-row
//! `seed` column, so augmentation is reproducible. A single stage can also be
//! applied directly:
//!
//! ```
//! use bunsen::errors::BunsenResult;
//! use bunsen_firehose_image::augmentation::{
//!     AugmentationStage,
//!     ImageAugContext,
//!     orientation::flip::HorizontalFlipStage,
//! };
//! use image::{
//!     DynamicImage,
//!     RgbImage,
//! };
//! use rand::{
//!     SeedableRng,
//!     rngs::StdRng,
//! };
//!
//! fn main() -> BunsenResult<()> {
//!     let stage = HorizontalFlipStage::new();
//!
//!     // The context carries a seeded RNG; stages draw from it for randomness.
//!     let mut ctx = ImageAugContext::new(StdRng::seed_from_u64(0));
//!
//!     let source: DynamicImage = RgbImage::new(8, 4).into();
//!     let flipped = stage.augment_image(source, &mut ctx)?;
//!
//!     // A horizontal flip preserves the dimensions.
//!     assert_eq!((flipped.width(), flipped.height()), (8, 4));
//!     Ok(())
//! }
//! ```
//!
//! See the crate root for an end-to-end `LOAD_IMAGE` → augment → tensor schema.

pub use image::{
    ColorType,
    DynamicImage,
    imageops::FilterType,
};

pub mod control;

pub mod noise;
pub mod orientation;

use bunsen_firehose::define_firehose_operator;

mod operator;
mod plugin;
mod stage;

pub use operator::*;
pub use plugin::*;
pub use stage::*;

#[cfg(test)]
mod tests;

// The operator's id is this module's path (`define_firehose_operator!`), and
// schemas store it, so its definition stays here.
define_firehose_operator!(AUGMENT_IMAGE, AugmentImageOperatorFactory::new());
