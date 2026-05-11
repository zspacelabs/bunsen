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

/// Re-export public dependencies.
#[allow(unused_imports)]
#[allow(missing_docs)]
pub mod public {
    pub use burn;
    pub use hashbrown;
}

#[cfg(feature = "cache")]
pub use bunsen_cache as cache;
// Make the macro targets public.
// TODO: re-examine contracts publication.

pub mod blocks;
pub mod kit;
pub mod models;
pub mod ops;
pub mod support;
pub mod zspace;

pub use bunsen_contracts as contracts;
#[doc(inline)]
pub use bunsen_contracts_macros::shape_contract;
mod macros;

mod errors;

#[doc(inline)]
pub use errors::*;
