//! # Convolution support
//!
//! Everything around convolution that is not itself a trainable layer. The
//! layers are burn's ([`Conv1d`], [`Conv2d`]); the composites built from them
//! are in [`crate::blocks::conv`].
//!
//! ## Why pure shape functions
//!
//! Given a convolution's kernel size, stride, padding, and dilation, what
//! output shape does it produce? burn answers this implicitly when you call
//! `forward`. This module gives the same answer as a pure function, which is
//! what a config or a `{Name}Meta` trait needs to advertise its output size
//! before any tensor exists. [`ConvBlock2dMeta::try_output_resolution`] and
//! the conv net's padding are built on it.
//!
//! ## Map
//!
//! - **Shape arithmetic.** [`try_conv1d_output_size`] for one axis;
//!   [`try_conv_output_shape`] over a `[usize; D]` shape, and
//!   [`try_conv_output_shape_dyn`] over slices, for a rank known only at run
//!   time. Each refuses a zero size, kernel, stride or dilation, and a kernel
//!   that does not fit the padded input, with a
//!   [`ConstraintError`](crate::errors::ConstraintError) naming the geometry;
//!   its twin without `try_` panics instead. [`stride_div_output_resolution`]
//!   is the downsample-by-stride shortcut.
//! - **Padding.** [`get_square_conv2d_padding`] and
//!   [`build_square_conv2d_padding_config`] give the symmetric padding for a
//!   square kernel. [`get_same_padding`] and [`pad_same`] give TensorFlow-style
//!   "SAME" padding, which may be asymmetric and is applied to the input rather
//!   than configured on the layer.
//! - **Conv variants.** [`Conv2dVariantOptions`] is timm's conv variant as one
//!   value, held by the conv blocks: its `forward` runs a [`Conv2d`]
//!   SAME-padded (`Conv2dSame`), weight-standardized
//!   ([`weight_standardized_conv2d`], `StdConv2d`), or both, and it states the
//!   rule they share (the conv's own padding is `Valid`).
//! - **Initialization.** [`CONV_INTO_RELU_INITIALIZER`] for a conv layer
//!   feeding a `ReLU`.
//! - **Functional convolution.** [`convolve_func_2d`] unfolds every window of a
//!   `[batch, c_in, height, width]` input and hands them to one closure in a
//!   single call, for window functions that are not a linear `Conv2d`.
//! - **Streams.** [`CausalConvStreamContext`], one stream through a causal
//!   convolution: the inputs a later call still needs, windowed as the whole
//!   forward is, so the chunking is not observable; [`CausalPadding`], what its
//!   edges are padded with. The convolution itself is
//!   [`CausalConv1d`](crate::blocks::conv::CausalConv1d).
//! - **Filters.** [`conv2d_kernel_midpoint_filter`] builds a 0/1 mask of the
//!   positions where a kernel's midpoint can land;
//!   [`drop_block_2d`](crate::ops::drop::drop_block_2d) uses it.
//!
//! [`Conv1d`]: burn::nn::conv::Conv1d
//! [`Conv2d`]: burn::nn::conv::Conv2d
//! [`ConvBlock2dMeta::try_output_resolution`]: crate::blocks::conv::ConvBlock2dMeta::try_output_resolution

mod causal_stream;
mod conv_func_2d;
mod conv_shape;
mod conv_variants;
mod midpoint_filter;
mod same_padding;
mod weight_std;

pub use causal_stream::*;
pub use conv_func_2d::*;
pub use conv_shape::*;
pub use conv_variants::*;
pub use midpoint_filter::*;
pub use same_padding::*;
pub use weight_std::*;
