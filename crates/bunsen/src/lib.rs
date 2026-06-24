#![cfg_attr(feature = "wgpu", recursion_limit = "512")]
#![warn(missing_docs)]
//!# bunsen
//!
//! `bunsen` aims to be a "batteries included" complementary
//! community standard library for extending the [burn](https://burn.dev) tensor library.
//!
//! ## Organization
//!
//! ### Burn Extensions
//!
//! * [`burner`] - this is a library of [`burn::module::Module`] lifecycle
//!   components that extend the current functionality of burn.
//!   * [`module::reflection`](`burner::module::reflection`) has powerful tools
//!     for dynamic [`burn::module::Module`] reflection.
//!   * [`optim`](`burner::optim`) has parameter-group optimizer extensions.
//! * [`contracts`] - this is a library of runtime tensor-shape contracts.
//!
//! ### Component Libraries
//!
//! * [`blocks`] - this is a library of [`burn::module::Module`] components.
//!   This includes simple inner layers, recurrent utility blocks, and entire
//!   model families.
//! * [`ops`] - this is a library [`burn::tensor::Tensor`] operations.
//! * [`kits`] - this is a library of full models and simulation kits.
//!
//! ### App and Testing Support Libs
//!
//! * [`errors`] - this is a library of error types and tooling.
//! * [`support`] - this is a library of support functions for bunsen, including
//!   testing tooling which may be useful for clients.
//! * [`zspace`] - this is a library of z-space / index utilities.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

extern crate alloc;
extern crate core;

// Make the macro targets public.
// TODO: re-examine contracts publication.
#[doc(hidden)]
pub use bunsen_contracts_macros::shape_contract as __proc_shape_contract;

/// Re-exports public dependencies.
#[allow(unused_imports)]
#[allow(missing_docs)]
pub mod public {
    pub use burn;
    #[cfg(feature = "train")]
    pub use hashbrown;
}

pub mod blocks;
pub mod burner;
pub mod contracts;
pub mod data;
pub mod errors;
pub mod kits;
pub mod ops;
pub mod support;
pub mod zspace;
