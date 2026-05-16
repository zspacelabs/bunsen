//! XML / `XPath` support.

pub mod names;
mod xee_util;
mod xml_param_desc;
mod xot_util;

#[doc(inline)]
pub use xee_util::*;
#[doc(inline)]
pub use xml_param_desc::*;
#[doc(inline)]
pub use xot_util::*;
