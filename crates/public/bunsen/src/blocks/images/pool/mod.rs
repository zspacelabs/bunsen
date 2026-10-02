//! # Pooling layers
//!
//! [`AvgPool2dSame`] is average pooling with TensorFlow-style "SAME" padding:
//! it pads the input with [`pad_same`](crate::ops::conv::pad_same), then runs
//! burn's [`AvgPool2d`](burn::nn::pool::AvgPool2d).

mod avg_pool_2d_same;

pub use avg_pool_2d_same::*;
