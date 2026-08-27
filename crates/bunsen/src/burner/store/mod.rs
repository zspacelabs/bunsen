//! Store-boundary helpers.
//!
//! Utilities for what happens as parameters cross a module-store boundary —
//! the read and write paths, rather than the module's own structure or its
//! compute.

mod param_mappers;

#[doc(inline)]
pub use param_mappers::*;
