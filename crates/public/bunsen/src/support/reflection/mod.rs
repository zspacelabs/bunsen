//! Serializable source locations.
//!
//! This holds [`LocationDesc`], a serializable copy of a
//! [`std::panic::Location`]; audit event headers record where each
//! checkpoint was called with it.
//!
//! Despite the name, this is **not** module reflection. That is
//! [`burner::module::reflection`](crate::burner::module::reflection): the XML
//! tree and `XPath` queries over a module's structure. The two modules are
//! unrelated.

mod location_desc;

pub use location_desc::*;
