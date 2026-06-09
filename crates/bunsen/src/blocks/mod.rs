//! # Neural-Network / Module Building Blocks
//!
//! `bunsen::blocks` is a library of reusable `burn::module::Module`
//! components &mdash; the *parts* you compose into larger models. Each
//! block ships as a triple:
//!
//! - a `Config` struct (`#[derive(Config)]`),
//! - a `Module` struct (`#[derive(Module)]`),
//! - an `init` constructor connecting the two.
//!
//! Where [`crate::ops`] supplies pure functional tensor operations,
//! `blocks` supplies the *stateful* layers that own parameters and can
//! be trained.
//!
//! Where [`crate::kits`] supplies whole end-to-end models (`ResNet`,
//! `NanoChatGpt`, ...), `blocks` supplies the sub-modules those kits
//! are assembled from. A typical user picks an existing kit; an author
//! building a new model reaches into `blocks`.
//!
//! ## Map of the module
//!
//! - [`transformers`] &mdash; building blocks for transformer models.
//!   - [`transformers::attention`] &mdash; causal self-attention
//!     (`CausalSelfAttention`), scaled-dot-product attention helpers
//!     (`scaled_dot_product_attention` and friends), and a `KVCache` for
//!     autoregressive decoding.
//!   - [`transformers::embedding`] &mdash; positional embeddings, currently
//!     `RotaryEmbedding` (`RoPE`) with a clip-by-range helper for
//!     KV-cache-friendly slicing.
//!
//! - [`images`] &mdash; building blocks for vision models.
//!   - [`conv`] &mdash; conv composites: `ConvNorm2d` (conv + batchnorm) and
//!     `ConvBlock2d` (conv / norm / activation).
//!   - [`images::patching`] &mdash; patch tokenization, currently `PatchEmbed`
//!     (ViT-style patch &rarr; embedding projection).
//!   - [`images::pool`] &mdash; pooling that doesn't fit `burn`'s defaults,
//!     currently `AvgPool2dSame` (TF-style same-padding average pool).
//!   - [`images::drop`] &mdash; stochastic regularization layers: `DropBlock`
//!     (Ghiasi et al., 2018), `DropPath` (stochastic depth, Huang et al.,
//!     2016), and supporting types (`progressive_dpr` rate tables,
//!     `SizeConfig`).
//!
//! See the per-item rustdoc for shape contracts, defaults, and the
//! papers each block implements.

pub mod conv;
pub mod images;
pub mod transformers;
