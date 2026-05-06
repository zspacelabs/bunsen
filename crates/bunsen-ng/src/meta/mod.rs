//! Meta descriptors of external types.

pub mod param_desc;
pub mod tensor_desc;
pub mod tensor_kinds;

#[doc(inline)]
pub use param_desc::*;
#[doc(inline)]
pub use tensor_desc::*;
#[doc(inline)]
pub use tensor_kinds::*;
