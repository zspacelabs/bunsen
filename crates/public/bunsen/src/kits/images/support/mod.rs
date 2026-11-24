//! # Shared parts of the image kits
//!
//! What the image kits share across their tasks, at the area's lowest
//! covering level (the [kits layout](crate::kits)):
//!
//! - [`transform`]: an image model's eval transform,
//!   [`ImageEvalTransformConfig`](transform::ImageEvalTransformConfig), with
//!   timm's settings and size arithmetic, and the tensor half that crops and
//!   normalises.
//! - [`labels`]: a classifier's class labels,
//!   [`ImageLabelSet`](labels::ImageLabelSet) (`ImageNet`'s `WordNet` synsets
//!   and their lemmas, as timm ships them), and the `ImageNet` subset a class
//!   count means.
//! - `eval` (feature `images`): the half of the transform that needs the
//!   `image` crate, from an image file to a model's input.

#[cfg(feature = "images")]
pub mod eval;
pub mod labels;
pub mod transform;
