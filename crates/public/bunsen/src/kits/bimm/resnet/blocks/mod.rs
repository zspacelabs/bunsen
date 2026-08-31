//! # `ResNet` Blocks

pub(crate) mod resnet_model;

mod basic_block;
mod bottleneck_block;
mod downsample;
mod layer_block;
mod residual_block;

#[doc(inline)]
pub use basic_block::*;
#[doc(inline)]
pub use bottleneck_block::*;
#[doc(inline)]
pub use downsample::*;
#[doc(inline)]
pub use layer_block::*;
#[doc(inline)]
pub use residual_block::*;
#[doc(inline)]
pub use resnet_model::*;
