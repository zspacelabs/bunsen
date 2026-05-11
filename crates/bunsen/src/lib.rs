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
// Make the macro targets public.
// TODO: re-examine contracts publication.
#[doc(hidden)]
pub use bunsen_contracts;
#[doc(inline)]
pub use bunsen_contracts as contracts;
/// A macro which defines a static [`crate::ShapeContract`].
///
/// See [`crate::shape_contract`] for documentation on the contract syntax.
///
/// ```rust,no_run
/// use bunsen_contracts::define_shape_contract;
///
/// define_shape_contract!(
///   CONTRACT,
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"]);
/// ```
#[macro_export]
macro_rules! define_shape_contract {
    ($name:ident, [ $($contract_expr:tt)* ] $(,)?) => {
        static $name: $crate::ShapeContract<'static> = $crate::bunsen_contracts::shape_contract![$($contract_expr)*];
    };
}

pub mod blocks;
pub mod kit;
pub mod ops;
pub mod support;
pub mod zspace;

mod errors;
pub mod models;

#[doc(inline)]
pub use errors::*;
