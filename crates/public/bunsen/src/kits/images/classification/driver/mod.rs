//! # The image-classification driver
//!
//! What every image classifier shares, whatever its kit:
//!
//! - [`ImageClassifier`], the seam: an
//!   [`ImageEncoder`](crate::kits::images::support::encoder::ImageEncoder) with
//!   a classifier head. It is object-safe, so a tool or a validation holds any
//!   kit's model as a `Box<dyn ImageClassifier>`.
//! - [`try_top_k`] / [`top_k`]: the `k` most probable classes of each row of
//!   logits, named by a card's labels ([`ClassPrediction`]).
//! - `classify_files` (feature `images`): image files through a model's eval
//!   transform and head, to their top classes.

mod classifier;
#[cfg(feature = "images")]
mod classify;

pub use classifier::*;
#[cfg(feature = "images")]
pub use classify::*;
