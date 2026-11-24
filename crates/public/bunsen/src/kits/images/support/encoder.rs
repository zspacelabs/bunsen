//! The image-encoder seam: what every image model answers, whatever its
//! tree.

use alloc::vec::Vec;
use core::fmt::Debug;

use burn::Tensor;

/// An image encoder's unpooled output: a conv net's feature map, or a
/// transformer's tokens.
#[derive(Debug, Clone)]
pub enum ImageFeatures {
    /// A `[batch, channels, height, width]` feature map.
    Spatial(Tensor<4>),

    /// A transformer's tokens, `[batch, prefix + grid_h*grid_w, d]`: the
    /// `prefix` tokens (class, distillation, register) first, then the patch
    /// grid, row by row.
    Tokens {
        /// The `[batch, prefix + grid_h*grid_w, d]` tokens.
        tokens: Tensor<3>,
        /// How many of the tokens precede the patch grid.
        prefix: usize,
        /// The patch grid's `[grid_h, grid_w]`.
        grid: [usize; 2],
    },
}

/// An image model's encoder: images in, features out, before any classifier
/// head. Every image model is one; a classifier is an
/// [`ImageClassifier`](crate::kits::images::classification::driver::ImageClassifier)
/// too.
///
/// The trait is object-safe, so a tool holds a `Box<dyn ImageEncoder>` from
/// any kit. Its methods take the eval transform's output
/// ([`ImageEvalTransformConfig`](super::transform::ImageEvalTransformConfig)):
/// normalised `[batch, channels, height, width]` images.
pub trait ImageEncoder: Debug + Send + Sync {
    /// The channel count the model takes.
    fn input_channels(&self) -> usize;

    /// The width of [`forward_pooled`](Self::forward_pooled)'s features:
    /// timm's `num_features`.
    fn d_features(&self) -> usize;

    /// The unpooled features of a `[batch, channels, height, width]` batch:
    /// timm's `forward_features`.
    fn forward_features(
        &self,
        images: Tensor<4>,
    ) -> ImageFeatures;

    /// The pooled `[batch, d_features]` features, the classifier's input
    /// (timm's `forward_head(..., pre_logits=True)`).
    fn forward_pooled(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2>;

    /// Every stage's output, as `[batch, channels_i, height_i, width_i]`
    /// maps, in timm's `feature_info` order (timm's `forward_intermediates`):
    /// what a validation compares stage by stage, and a backbone's taps.
    fn forward_intermediates(
        &self,
        images: Tensor<4>,
    ) -> Vec<Tensor<4>>;
}
