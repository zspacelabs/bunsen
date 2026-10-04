//! # Neural-network building blocks
//!
//! `bunsen::blocks` is a library of reusable `burn::module::Module`
//! components: the *parts* you compose into larger models.
//!
//! ## Blocks and ops
//!
//! A block is a component meant to be used as a Module root or as a part of
//! a module tree. [Ops and blocks](crate::ops#ops-and-blocks) defines the
//! line between blocks and [`crate::ops`], and the rule that cache and stream
//! state is injected.
//!
//! A block is typically "these parameters, plus these `ops` calls in order";
//! blocks import ops, never the reverse. A block need not own parameters:
//! `DropPath`, `DropBlock2d` and `AvgPool2dSame` own none, and are blocks
//! because they sit in a module tree as layers. A block that decodes takes
//! its cache as an argument rather than holding it:
//! [`CausalSelfAttention::forward`] takes a [`KVCache`].
//!
//! Where [`crate::kits`] supplies whole models (`ResNet`, `NanoChatGpt`,
//! Whisper, ...), `blocks` supplies the sub-modules those kits are assembled
//! from. A typical user picks an existing kit; an author building a new model
//! reaches into `blocks`.
//!
//! ## Lifecycle
//!
//! Each block ships as:
//!
//! - a `Config` struct (`#[derive(Config)]`);
//! - a `Module` struct (`#[derive(Module)]`), built by the config;
//! - usually a `{Name}Meta` trait implemented by both, so the block's geometry
//!   reads the same before and after init.
//!
//! Most configs build their module through [`ModuleInit`]
//! (`init(&device)` / `try_init(&device)`). A few keep an inherent `init`
//! instead: the parameter-free `DropPathConfig`, `DropBlock2dConfig` and
//! `AvgPool2dSameConfig` take no device, and `CausalSelfAttentionConfig` also
//! takes the layer's index in its stack (see `ModuleInit`'s
//! [hand-written `init`](crate::burner::module::ModuleInit#hand-written-init)).
//!
//! ## Map of the module
//!
//! - [`conv`]: conv / norm / activation blocks, in 1D and 2D (`ConvBlock1d`,
//!   `ConvBlock2d`), and sequences of them (`ConvSeq1d`, `ConvSeq2d`).
//! - [`images`]: vision blocks.
//!   - [`images::drop`]: structured drop layers, `DropBlock2d` and `DropPath`,
//!     and the drop-path rate tables.
//!   - [`images::patching`]: `PatchEmbed`, the Swin Transformer V2 patch
//!     embedding.
//!   - [`images::pool`]: `AvgPool2dSame`, average pooling with TensorFlow-style
//!     "SAME" padding.
//! - [`rnn`]: recurrent blocks: `FusedLstm`, a single-step LSTM with fused
//!   gates, and its state type.
//! - [`transformers`]: transformer blocks.
//!   - [`transformers::attention`]: `CausalSelfAttention`, with QK-norm, rotary
//!     embeddings, grouped-query attention and an optional KV cache.
//!   - [`transformers::embedding`]: `RotaryEmbedding`, and fixture embeddings
//!     for tests.
//!   - [`transformers::mlp`]: `Mlp`, the transformer feed-forward block.
//!
//! See the per-item rustdoc for shape contracts, defaults, and the papers each
//! block implements.
//!
//! [`CausalSelfAttention::forward`]: transformers::attention::csa::CausalSelfAttention::forward
//! [`KVCache`]: crate::ops::transformers::attention::KVCache
//! [`ModuleInit`]: crate::burner::module::ModuleInit

pub mod conv;
pub mod images;
pub mod rnn;
pub mod transformers;
