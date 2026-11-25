//! # Shared parts of the image kits
//!
//! What the image kits share across their tasks, at the area's lowest
//! covering level (the [kits layout](crate::kits)):
//!
//! - [`transform`]: an image model's eval transform,
//!   [`ImageEvalTransformConfig`](transform::ImageEvalTransformConfig), with
//!   timm's settings and size arithmetic, and the tensor half that crops and
//!   normalises.
//! - [`encoder`]: the seam every image model implements,
//!   [`ImageEncoder`](encoder::ImageEncoder): features, pooled features and
//!   stage outputs, whatever the model's tree.
//! - [`card`]: what a pretrained image model comes with,
//!   [`ImageModelCard`](card::ImageModelCard) (its eval transform and labels),
//!   and the construct every image kit loads,
//!   [`ImageModelConstruct`](card::ImageModelConstruct).
//! - [`imagenet`]: `ImageNet`'s data by its catalog names: the label sets, and
//!   the validation images, pulled from a gated Hub mirror and never shipped.
//! - [`interop`]: the image kits' shared interop: a timm hub repo's
//!   `config.json` and the eval transform it states.
//! - [`labels`]: a classifier's class labels,
//!   [`ImageLabelSet`](labels::ImageLabelSet) (`ImageNet`'s `WordNet` synsets
//!   and their lemmas, as timm ships them), and the `ImageNet` subset a class
//!   count means.
//! - `eval` (feature `images`): the half of the transform that needs the
//!   `image` crate, from an image file to a model's input.

pub mod card;
pub mod encoder;
#[cfg(feature = "images")]
pub mod eval;
pub mod imagenet;
pub mod interop;
pub mod labels;
pub mod transform;
