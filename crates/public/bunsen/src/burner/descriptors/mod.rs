//! Serializable / Reference-Free descriptors of `burner` types.

pub mod shims;

mod param_desc;
mod parse_dtype;
mod tensor_desc;
mod tensor_kinds;
mod tolerance_desc;

#[doc(inline)]
pub use param_desc::*;
#[doc(inline)]
pub use parse_dtype::*;
#[doc(inline)]
pub use tensor_desc::*;
#[doc(inline)]
pub use tensor_kinds::*;
#[doc(inline)]
pub use tolerance_desc::*;
