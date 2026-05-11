#![cfg_attr(feature = "wgpu", recursion_limit = "512")]
#![warn(missing_docs)]
//!# bunsen (burn-er)
//!
//! `bunsen` is an extension library for `burn`.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

extern crate alloc;
extern crate core;

#[cfg(feature = "cache")]
pub use bunsen_cache as cache;
#[doc(inline)]
pub use bunsen_contracts as contracts;

pub mod blocks;
pub mod kit;
pub mod ops;
pub mod support;
pub mod zspace;

mod errors;
#[doc(inline)]
pub use errors::*;
