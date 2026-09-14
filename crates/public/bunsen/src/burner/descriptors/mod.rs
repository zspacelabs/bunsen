//! Serializable / Reference-Free descriptors of `burner` types.

mod param_desc;
pub mod shims;
mod tensor_desc;
mod tensor_kinds;
mod tolerance_desc;

#[doc(inline)]
pub use param_desc::*;
#[doc(inline)]
pub use tensor_desc::*;
#[doc(inline)]
pub use tensor_kinds::*;
#[doc(inline)]
pub use tolerance_desc::*;
