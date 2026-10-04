//! XML / `XPath` support.
//!
//! The module tree's element and attribute [`names`]; the `<Param/>` node
//! codec for a [`TensorParamDesc`]; the codec for a parameter's `shape`
//! attribute, [`shape_to_xml_attr`] / [`shape_from_xml_attr`], which writes a
//! [`Shape`](burn::prelude::Shape)'s dimensions space-separated (`"2 3 4"`);
//! and `xot` / `xee` printing and error helpers.
//!
//! [`TensorParamDesc`]: crate::burner::descriptors::TensorParamDesc

pub mod names;
mod shape_attr;
mod xee_util;
mod xml_param_desc;
mod xot_util;

pub use shape_attr::*;
pub use xee_util::*;
pub use xml_param_desc::*;
pub use xot_util::*;
