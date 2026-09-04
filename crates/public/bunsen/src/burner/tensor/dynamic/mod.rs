//! Support for wrapping, unwrapping, and operating on dynamically typed and
//! ranked tensors.

mod dispatch_rank;
mod dyn_tensor;

#[doc(inline)]
pub use dispatch_rank::*;
#[doc(inline)]
pub use dyn_tensor::*;
