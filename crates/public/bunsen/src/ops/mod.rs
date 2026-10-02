//! # Tensor operations
//!
//! `bunsen::ops` collects tensor operations that extend burn's built-in
//! surface: the work you call inside a `Module::forward`, or around one, that
//! has no home on `Tensor` upstream.
//!
//! ## Ops and blocks
//!
//! The line between `ops` and [`crate::blocks`] is about what a thing is
//! *for*:
//!
//! - **blocks** are `bunsen.nn` / `torch.nn`-alike components, meant to be used
//!   as Module roots or as parts of a module tree;
//! - **ops** are operation-focused. They may still use the Module or Config
//!   machinery, but to hold cached tables or state.
//!
//! Cache and stream state is *injected*: the caller builds it and passes it
//! in, and the model never owns it, so one model can have several caches in
//! one process. [`KVCache`](transformers::attention::KVCache) and
//! [`SlidingStftContext`](signal::SlidingStftContext) are both examples.
//!
//! In practice `ops` is mostly free functions plus small configuration
//! values, and a few Modules and Configs that hold cached tables or stream
//! state ([`SlidingStft`](signal::SlidingStft) holds an
//! analysis window as a bare tensor, not a trainable `Param`). A block is
//! typically "these parameters, plus these `ops` calls in order":
//! [`DropBlock2d`] holds a [`DropBlockOptions`](drop::DropBlockOptions) and
//! calls [`drop_block_2d`](drop::drop_block_2d) while training.
//!
//! The dependency runs one way: blocks import ops, and ops never import
//! blocks. Both build on [`crate::burner`], and on [`crate::contracts`] for
//! the shape checks at their boundaries. Method-style extensions to `Tensor`
//! itself live in [`crate::burner::tensor`] instead.
//!
//! ## Value objects as configuration
//!
//! Many ops take their settings as one small value object rather than a list
//! of arguments. A value object is `Clone`, comparable and serializable, so it
//! can be a field of a `#[derive(Config)]` struct and is saved with the
//! config. With a `ModuleDisplay` impl it can also be a field of a
//! `#[derive(Module)]` struct, and prints with the model. Either way the same
//! value is passed straight to the op. The in-tree chain nests four deep,
//! `ClampOp` ⊂ `NoiseConfig` ⊂ `DropBlockOptions` ⊂ `DropBlock2dConfig`:
//!
//! - [`ClampOp`](clamp::ClampOp): an optional min and max;
//! - [`NoiseConfig`](noise::NoiseConfig): a `Distribution` plus an optional
//!   `ClampOp`;
//! - [`DropBlockOptions`](drop::DropBlockOptions): the `DropBlock` settings,
//!   plus an optional `NoiseConfig` to refill dropped regions;
//! - [`DropBlock2dConfig`]: the block's `Config`, whose one field is a
//!   `DropBlockOptions` (`#[config(default = "DropBlockOptions::default()")]`),
//!   and which [`DropBlock2d`] keeps as a module field.
//!
//! Two styles coexist. `ClampOp`, `NoiseConfig` and `DropBlockOptions` derive
//! `serde` and implement `ModuleDisplay` by hand
//! ([`RmsNormOptions`](norm::RmsNormOptions) derives `serde` only); others
//! are `#[derive(Config)]` types, such as [`LogBase`](math::LogBase),
//! [`StftWindowConfig`](signal::StftWindowConfig) and
//! [`ScaledDotProductAttentionConfig`](transformers::attention::ScaledDotProductAttentionConfig).
//! The names are mixed too (`*Op`, `*Config`, `*Options`).
//!
//! ## Map of the module
//!
//! ### Tensor generation
//! - [`arange`]: float ranges and `linspace`, on the host (`Vec<f64>`) and on
//!   the device. burn's `Tensor::arange` is integer-only.
//! - [`noise`]: [`NoiseConfig`](noise::NoiseConfig), a distribution plus an
//!   optional clamp, sampled with `noise()` / `noise_like()`.
//!
//! ### Element-wise
//! - [`clamp`]: [`ClampOp`](clamp::ClampOp), clamping as a setting rather than
//!   a one-shot call.
//! - [`math`]: [`LogBase`](math::LogBase), a logarithm in a configurable base.
//!
//! ### Regularization
//! - [`mod@drop`]: [`dropout`](drop::dropout), [`drop_path`](drop::drop_path)
//!   (stochastic depth), and [`drop_block_2d`](drop::drop_block_2d) with its
//!   [`DropBlockOptions`](drop::DropBlockOptions) and
//!   [`SizeConfig`](drop::SizeConfig). The layers that apply them while
//!   training are in [`crate::blocks::images::drop`].
//!
//! ### Normalization
//! - [`norm`]: [`rms_norm`](norm::rms_norm), RMS normalization without
//!   trainable parameters. For the parametric layer use burn's `RmsNorm`.
//!
//! ### Shape transforms
//! - [`repeat`]: [`repeat_interleave`](repeat::repeat_interleave) along a
//!   (negatively indexable) dimension, with `NumPy` / `PyTorch` semantics.
//! - [`split`]: [`split_padded`](split::split_padded) cuts a dimension into
//!   equal chunks, zero-padding the last one;
//!   [`window_padded`](split::window_padded) does the same cut but keeps the
//!   chunks in one tensor, as an extra window axis.
//!
//! ### Convolution support
//! - [`conv`]: convolution shape arithmetic, symmetric and TensorFlow-style
//!   "SAME" padding, a conv initializer, functional 2D convolution, and
//!   kernel-midpoint masks. The conv layers are burn's and
//!   [`crate::blocks::conv`]'s.
//!
//! ### Embeddings
//! - [`embedding`]: [`unembed`](embedding::unembed) for a tied output head, and
//!   the rotary-embedding frequency tables.
//!
//! ### Transformers
//! - [`transformers`]: [`attention`](transformers::attention), scaled
//!   dot-product attention, attention over burn's `MultiHeadAttention` weights,
//!   and the two key/value caches (`KVCache`, `AttnKvPair`).
//!
//! ### Signal processing
//! - [`signal`]: sliding-window STFT, analysis windows, and waveform-to-log-mel
//!   conversion ([`signal::perceptive_audio`]), with per-stream contexts.
//!
//! [`DropBlock2d`]: crate::blocks::images::drop::drop_block::DropBlock2d
//! [`DropBlock2dConfig`]: crate::blocks::images::drop::drop_block::DropBlock2dConfig

pub mod arange;
pub mod clamp;
pub mod conv;
pub mod drop;
pub mod embedding;
pub mod math;
pub mod noise;
pub mod norm;
pub mod repeat;
pub mod signal;
pub mod split;
pub mod transformers;
