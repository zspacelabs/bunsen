//! # Structured drop layers
//!
//! Regularization layers that drop *structured* pieces of an activation,
//! rather than independent elements as burn's [`Dropout`] does. Each is a
//! parameter-free `Module` that holds its settings and applies the matching
//! [`crate::ops::drop`] function only when autodiff is enabled (training);
//! otherwise it passes its input through.
//!
//! - [`DropBlock2d`](drop_block::DropBlock2d), built by
//!   [`DropBlock2dConfig`](drop_block::DropBlock2dConfig), wraps
//!   [`drop_block_2d`](crate::ops::drop::drop_block_2d). The config's one field
//!   is a [`DropBlockOptions`](crate::ops::drop::DropBlockOptions) value
//!   (re-exported in [`drop_block`]), whose block size is a
//!   [`SizeConfig`](crate::ops::drop::SizeConfig) per axis.
//!
//!   `DropBlock` ([Ghiasi et al., 2018](https://arxiv.org/pdf/1810.12890.pdf))
//!   drops contiguous blocks of activations instead of independent pixels.
//!   For convnets this is a much stronger regularizer than plain dropout:
//!   adjacent pixels are highly correlated, so dropping independent ones
//!   barely removes information.
//! - [`DropPath`](drop_path::DropPath), built by
//!   [`DropPathConfig`](drop_path::DropPathConfig), wraps
//!   [`ops::drop::drop_path`](crate::ops::drop::drop_path);
//!   [`with_skip`](drop_path::DropPath::with_skip) runs a residual branch under
//!   it.
//!
//!   Stochastic depth ([Huang et al., 2016](https://arxiv.org/abs/1603.09382))
//!   zeroes the entire residual branch for a sample, with some probability, so
//!   the network sees a shorter effective depth on each training step.
//! - [`rate_table`] configures a stack of `DropPath` layers:
//!   [`progressive_dpr`](rate_table::progressive_dpr) ramps the drop rate
//!   linearly from 0 over the depth, as Swin Transformer V2 and `timm` do, and
//!   [`DropPathRateDepthTable`](rate_table::DropPathRateDepthTable) splits that
//!   ramp across stages of given depths.
//!
//! [`Dropout`]: burn::nn::Dropout

pub mod drop_block;
pub mod drop_path;
pub mod rate_table;
