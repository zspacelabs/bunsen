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
//!     (`scaled_dot_product_attention` and friends), and a `KVCache`
//!     for autoregressive decoding.
//!   - [`transformers::embedding`] &mdash; positional embeddings,
//!     currently `RotaryEmbedding` (RoPE) with a clip-by-range
//!     helper for KV-cache-friendly slicing.
//!
//! - [`images`] &mdash; building blocks for vision models.
//!   - [`images::conv`] &mdash; conv composites: `ConvNorm2d`
//!     (conv + batchnorm) and `CNA2d` (conv / norm / activation).
//!   - [`images::patching`] &mdash; patch tokenization, currently
//!     `PatchEmbed` (ViT-style patch &rarr; embedding projection).
//!   - [`images::pool`] &mdash; pooling that doesn't fit `burn`'s
//!     defaults, currently `AvgPool2dSame` (TF-style same-padding
//!     average pool).
//!   - [`images::drop`] &mdash; stochastic regularization layers:
//!     `DropBlock` (Ghiasi et al., 2018), `DropPath` (stochastic
//!     depth, Huang et al., 2016), and supporting types
//!     (`progressive_dpr` rate tables, `SizeConfig`).
//!
//! ## Conventions
//!
//! Every block follows the patterns documented in the *Building
//! Reusable Modules* guide:
//!
//! - **`{Block}Meta` traits** when sub-modules want to introspect the
//!   block. For example, [`transformers::attention::CausalSelfAttentionMeta`]
//!   is implemented on the config and the live module, so a parent can
//!   read `n_head`, `n_kv_head`, `head_dim` without caching its own
//!   copy.
//! - **Contract &rarr; Structure splits** when the user-facing config is
//!   smaller than the implementation parameter list (see
//!   [`crate::kits::bimm::resnet::ResidualBlockContractConfig`] for a
//!   block-level example).
//! - **Tensor contracts** at module boundaries via [`crate::contracts`]
//!   for both runtime safety and as machine-checked documentation.
//!
//! See the per-item rustdoc for shape contracts, defaults, and the
//! papers each block implements.

pub mod images;
pub mod transformers;
