//! # Dropout-family operations
//!
//! Regularizers that zero part of an activation at random, as functions.
//! Each one takes its settings as arguments and runs whenever it is called;
//! deciding to run only while training is the caller's job. The matching
//! modules in [`crate::blocks::images::drop`] (and burn's [`Dropout`]) make
//! that decision: they apply the op only when autodiff is enabled.
//!
//! - [`dropout`] drops independent elements at random and rescales the rest by
//!   `1 / (1 - prob)`. It is the operation inside burn's [`Dropout`] module,
//!   without the training check;
//!   [`scaled_dot_product_attention`](crate::ops::transformers::attention::scaled_dot_product_attention)
//!   uses it on attention weights.
//! - [`drop_path`] is stochastic depth: it drops whole batch rows, so a
//!   residual branch is skipped for that sample. The [`DropPath`] module wraps
//!   it.
//! - [`drop_block_2d`] is `DropBlock`: it drops contiguous spatial blocks of a
//!   `[batch, channels, height, width]` map. Its settings are one
//!   [`DropBlockOptions`] value, whose block size is a [`SizeConfig`] per axis
//!   and whose dropped regions can be refilled with
//!   [`NoiseConfig`](crate::ops::noise::NoiseConfig) noise. The [`DropBlock2d`]
//!   module holds a `DropBlockOptions` and calls it.
//!
//! Why the structured drops regularize, and the rate tables that configure
//! a stack of `DropPath` layers, are in [`crate::blocks::images::drop`].
//!
//! [`Dropout`]: burn::nn::Dropout
//! [`DropPath`]: crate::blocks::images::drop::drop_path::DropPath
//! [`DropBlock2d`]: crate::blocks::images::drop::drop_block::DropBlock2d

mod drop_block;
mod drop_path_func;
mod dropout_func;
mod size_config;

pub use drop_block::*;
pub use drop_path_func::*;
pub use dropout_func::*;
pub use size_config::*;
