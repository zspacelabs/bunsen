//! # `DynTensor` - Dynamically Typed burn Tensors

extern crate core;

mod dispatch_rank;
mod dyn_tensor;
mod kind;

#[doc(inline)]
pub use dispatch_rank::*;
#[doc(inline)]
pub use dyn_tensor::*;
#[doc(inline)]
pub use kind::*;
