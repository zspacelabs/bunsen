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
//! Cache and stream state is *injected*, never owned by the model: the module
//! is immutable, and the caller creates a per-stream context or a per-decode
//! cache and passes it in on every call. One model can then serve several
//! caches or streams in one process.
//! [`KVCache`](transformers::attention::KVCache),
//! [`SlidingStftContext`](signal::SlidingStftContext) and
//! [`WhisperStreamContext`](crate::kits::speech::asr::whisper::driver::WhisperStreamContext)
//! are examples.
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
//! config. As a `#[module(skip)]` field it can also be held by a
//! `#[derive(Module)]` struct, and prints with the model through its `Debug`.
//! Either way the same value is passed straight to the op. The in-tree chain
//! nests four deep, `ClampOptions` ⊂ `NoiseOptions` ⊂ `DropBlockOptions` ⊂
//! `DropBlock2dConfig`:
//!
//! - [`ClampOptions`](clamp::ClampOptions): an optional min and max;
//! - [`NoiseOptions`](noise::NoiseOptions): a `Distribution` plus an optional
//!   `ClampOptions`;
//! - [`DropBlockOptions`](drop::DropBlockOptions): the `DropBlock` settings,
//!   plus an optional `NoiseOptions` to refill dropped regions;
//! - [`DropBlock2dConfig`]: the block's `Config`, whose one field is a
//!   `DropBlockOptions` (`#[config(default = "DropBlockOptions::default()")]`),
//!   and which [`DropBlock2d`] keeps as a skipped module field.
//!
//! Two styles coexist. `ClampOptions`, `NoiseOptions` and `DropBlockOptions`
//! derive `serde` and implement `ModuleDisplay` by hand
//! ([`RmsNormOptions`](norm::RmsNormOptions) derives `serde` only); others
//! are `#[derive(Config)]` types, such as [`LogBase`](math::LogBase),
//! [`StftWindowConfig`](signal::kernels::window::StftWindowConfig) and
//! [`ScaledDotProductAttentionOptions`](transformers::attention::ScaledDotProductAttentionOptions).
//! A bundle of settings handed to an op is `*Options`; `*Config` builds
//! something (a module, a state, a table); a single value is named for what
//! it is ([`LogBase`](math::LogBase), [`SizeSpec`](drop::SizeSpec)).
//!
//! ## Design Rules
//!
//! Also binding here: the crate root's [layering](crate#design-rules), and the
//! dtype rules of [`burner::module`'s Design
//! Rules](crate::burner::module#design-rules).
//!
//! - **An op is an operation:** a free function, a small value object, or a
//!   `Module` / `Config` used only to hold a cached table or state. A tensor
//!   operation or table that blocks share lives here, and `ops` never imports
//!   [`crate::blocks`] ([Ops and blocks](#ops-and-blocks)). Operations stay
//!   reusable by any block or kit, without pulling in module trees.
//! - **Cache and stream state is injected, never owned by the model** ([Ops and
//!   blocks](#ops-and-blocks) is the rule's one definition): the module is
//!   immutable, and the caller passes a per-stream context or a per-decode
//!   cache in on every call.
//! - **Settings worth naming form one value object**, `Clone`, comparable and
//!   serializable, passed straight to the op. A config embeds the value object
//!   rather than copying its fields, and a module holds it as `#[module(skip)]`
//!   ([Value objects as configuration](#value-objects-as-configuration)). A
//!   bundle of settings handed to an op is `*Options`; `*Config` builds
//!   something (a module, a state, a table); a single value is named for what
//!   it is.
//! - **An op checks the shapes at its boundary with [`crate::contracts`]**, in
//!   the notation of its rustdoc, as blocks do
//!   ([`drop_block_2d`](drop::drop_block_2d)): an always-run unpack where it
//!   reads dims, cfg-gated checks on what it returns. An op is the last place a
//!   wrong shape can be named before it turns into a wrong value. The crate
//!   root's [contracts rule](crate#design-rules) gives the reasons.
//! - **A table or constant that an op builds is made in a dtype the caller
//!   names**, through a dtype policy or a
//!   [`Widening`](crate::burner::tensor::Widening), never in the device default
//!   by implication
//!   ([`RotaryTableDTypePolicy`](embedding::RotaryTableDTypePolicy)). A table
//!   built in the device default turns into a silent narrowing as soon as that
//!   default changes.
//!
//! ## Map of the module
//!
//! ### Tensor generation
//! - [`arange`]: float ranges and `linspace`, on the host (`Vec<f64>`) and on
//!   the device. burn's `Tensor::arange` is integer-only.
//! - [`noise`]: [`NoiseOptions`](noise::NoiseOptions), a distribution plus an
//!   optional clamp, sampled with `noise()` / `noise_like()`.
//!
//! ### Element-wise
//! - [`clamp`]: [`ClampOptions`](clamp::ClampOptions), clamping as a setting
//!   rather than a one-shot call.
//! - [`math`]: [`LogBase`](math::LogBase), a logarithm in a configurable base.
//!
//! ### Regularization
//! - [`mod@drop`]: [`dropout`](drop::dropout), [`drop_path`](drop::drop_path)
//!   (stochastic depth), and [`drop_block_2d`](drop::drop_block_2d) with its
//!   [`DropBlockOptions`](drop::DropBlockOptions) and
//!   [`SizeSpec`](drop::SizeSpec). The layers that apply them while training
//!   are in [`crate::blocks::images::drop`].
//!
//! ### Normalization
//! - [`norm`]: [`RmsNormOptions::norm`](norm::RmsNormOptions::norm), RMS
//!   normalization without trainable parameters. For the parametric layer use
//!   burn's `RmsNorm`, or bunsen's norm seam.
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
//!   and the key/value cache every decoder uses (`KVCache`).
//!
//! ### Signal processing
//! - [`signal`]: sliding-window STFT, analysis windows, and waveform-to-log-mel
//!   conversion ([`signal::perceptive_audio`]), with per-stream contexts.
//!
//! ### Images
//! - [`images`]: the tensor half of image preprocessing: center crops (with
//!   padding where an image is smaller) at torchvision's offsets, and
//!   per-channel normalisation.
//!
//! [`DropBlock2d`]: crate::blocks::images::drop::drop_block::DropBlock2d
//! [`DropBlock2dConfig`]: crate::blocks::images::drop::drop_block::DropBlock2dConfig

pub mod arange;
pub mod clamp;
pub mod conv;
pub mod drop;
pub mod embedding;
pub mod images;
pub mod math;
pub mod noise;
pub mod norm;
pub mod reduce;
pub mod repeat;
pub mod signal;
pub mod split;
pub mod transformers;
