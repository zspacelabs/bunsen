//!# bunsen burn(er)
#![warn(missing_docs)]

extern crate alloc;

extern crate core;

/// Test-only macro import.
#[cfg(test)]
#[allow(unused_imports)]
#[macro_use]
extern crate hamcrest;

#[allow(dead_code)]
pub mod compat;

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod testing;

pub mod burn_ext;
pub mod cache;
pub mod nn;
pub mod utility;
