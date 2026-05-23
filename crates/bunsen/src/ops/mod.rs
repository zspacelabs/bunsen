//! # Functional Tensor API
//!
//! `bunsen::ops` collects standalone tensor functions that extend
//! `burn`'s built-in surface &mdash; the kind of utilities you'd reach
//! for inside a `Module::forward` implementation but that don't fit
//! the `Tensor` method namespace upstream.
//!
//! Everything here is *functional* in the sense relevant to deep
//! learning: pure functions over `Tensor`s (and optional small
//! configuration value-objects). Nothing in `ops` owns trainable
//! parameters &mdash; that's the job of [`crate::blocks`]. The two
//! work together: a `Module` config bundles a `Tensor`-shaped layer
//! plus an `ops::*` helper; the `forward` calls the layer and then
//! the helper.
//!
//! ## Map of the module
//!
//! ### Tensor generation
//! - [`arange`] &mdash; floating-point range generators on both `Vec<f64>`
//!   (`float_vec_arange`, `float_vec_linspace`) and `Tensor` (`float_arange`,
//!   `float_linspace`).
//! - [`noise`] &mdash; `NoiseConfig` bundles a `Distribution` with an optional
//!   [`clamp::ClampOp`], and supplies `noise()` / `noise_like()` to materialize
//!   tensors from it.
//!
//! ### Element-wise transforms
//! - [`clamp`] &mdash; `ClampOp` is a small value-object that captures an
//!   optional min and an optional max as a single reusable configuration, with
//!   `.clamp(t)` to apply it to a tensor. Useful anywhere clamping shows up as
//!   a *setting* rather than a one-shot call (config files, noise generators,
//!   etc.).
//! - [`mod@drop`] &mdash; functional `dropout()`. The matching trainable layer
//!   lives in [`crate::blocks::images::drop`].
//!
//! ### Normalization
//! - [`norm`] &mdash; `rms_norm()` and `RmsNormOptions` (`RMSNorm` without
//!   trainable parameters). For the parametric layer use
//!   `burn::nn::norm::RmsNorm`.
//!
//! ### Shape transforms
//! - [`repeat`] &mdash; `repeat_interleave()` along a (negatively indexable)
//!   dimension, mirroring `NumPy` / `PyTorch` semantics.
//!
//! ### Convolution support
//! - [`conv`] &mdash; everything around convolution that isn't a trainable
//!   layer:
//!   - **Shape arithmetic.** `maybe_conv_output_shape` /
//!     `expect_conv_output_shape` (and their `_dyn`, `_1d` variants) compute
//!     output shapes from kernel/stride/ padding/dilation;
//!     `stride_div_output_resolution` is the downsample helper;
//!     `get_square_conv2d_padding` / `build_square_conv2d_padding_config` are
//!     the most-common-case wrappers.
//!   - **Functional convolution.** `convolve_func_2d` folds a user-supplied
//!     closure over 2-D conv windows, which is the starting point for kernels
//!     that aren't expressible as a linear `Conv2d`.
//!   - **Filters.** `conv2d_kernel_midpoint_filter` &mdash; a kernel that picks
//!     the spatial mid-point sample.
//!
//! ## Relationship to [`crate::blocks`] and [`crate::contracts`]
//!
//! `ops` is the lower layer. A typical block is "this set of
//! parameters, plus these `ops` calls in order"; the
//! [`crate::contracts`] machinery sits between them, validating the
//! shapes that flow across the boundary. Most blocks document their
//! shape contracts in those terms.

pub mod arange;
pub mod clamp;
pub mod conv;
pub mod drop;
pub mod noise;
pub mod norm;
pub mod repeat;
