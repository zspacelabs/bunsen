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
//! the `ResNet` downsample are built on it.
//!
//! ## Map
//!
//! - **Shape arithmetic.** [`maybe_conv1d_output_size`] for one axis;
//!   [`maybe_conv_output_shape`] over a `[usize; D]` shape, and
//!   [`maybe_conv_output_shape_dyn`] over slices, for a rank known only at run
//!   time. Each `maybe_*` returns `None` when no output is legal (the kernel
//!   does not fit the padded input); its `expect_*` twin panics instead.
//!   [`stride_div_output_resolution`] is the downsample-by-stride shortcut.
//! - **Padding.** [`get_square_conv2d_padding`] and
//!   [`build_square_conv2d_padding_config`] give the symmetric padding for a
//!   square kernel. [`get_same_padding`] and [`pad_same`] give TensorFlow-style
//!   "SAME" padding, which may be asymmetric and is applied to the input rather
//!   than configured on the layer; [`AvgPool2dSame`] uses it.
//! - **Initialization.** [`CONV_INTO_RELU_INITIALIZER`] for a conv layer
//!   feeding a `ReLU`.
//! - **Functional convolution.** [`convolve_func_2d`] unfolds every window of a
//!   `[batch, c_in, height, width]` input and hands them to one closure in a
//!   single call, for window functions that are not a linear `Conv2d`.
//! - **Filters.** [`conv2d_kernel_midpoint_filter`] builds a 0/1 mask of the
//!   positions where a kernel's midpoint can land;
//!   [`drop_block_2d`](crate::ops::drop::drop_block_2d) uses it.
//!
//! [`Conv1d`]: burn::nn::conv::Conv1d
//! [`Conv2d`]: burn::nn::conv::Conv2d
//! [`ConvBlock2dMeta::try_output_resolution`]: crate::blocks::conv::ConvBlock2dMeta::try_output_resolution
//! [`AvgPool2dSame`]: crate::blocks::images::pool::AvgPool2dSame

mod conv_func_2d;
mod conv_shape;
mod midpoint_filter;
mod same_padding;

pub use conv_func_2d::*;
pub use conv_shape::*;
pub use midpoint_filter::*;
pub use same_padding::*;
