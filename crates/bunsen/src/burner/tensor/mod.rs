//! Tensor Operations.

mod data_view;
mod tensor_inplace_result_ext;
mod tensor_release_ext;

#[doc(inline)]
pub use data_view::*;
#[doc(inline)]
pub use tensor_inplace_result_ext::*;
#[doc(inline)]
pub use tensor_release_ext::*;
