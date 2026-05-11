#![cfg_attr(feature = "wgpu", recursion_limit = "512")]
//!# bunsen kit(er)
#![warn(missing_docs)]

extern crate alloc;
extern crate core;

#[cfg(feature = "cache")]
pub use bunsen_cache as cache;

pub(crate) mod support;

pub mod blocks;
pub mod errors;
pub mod kit;
pub mod ops;
pub mod zspace;

#[cfg(feature = "testing")]
pub mod testing;

#[doc(inline)]
pub use bunsen_contracts as contracts;
