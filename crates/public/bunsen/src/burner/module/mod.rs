//! # Module Support
//!
//! Helpers for building `burn` modules from configs, and for working with
//! them once built.
//!
//! - [`ModuleInit`] &mdash; the Config → Module step. A bunsen config that
//!   builds its module from a device alone implements it, with a fallible
//!   `try_init` and a panicking `init`. Its docs are the reference for the
//!   config lifecycle: the Simple Config and Stacked Config shapes, each with a
//!   compiled example.
//! - [`ToStructureConfig`] &mdash; the config-side twin of `ModuleInit`: it
//!   lowers an upper policy config of a Stacked Config family to its structure
//!   config, and a blanket impl gives every implementor `ModuleInit` through
//!   that structure.
//! - [`HasDType`] &mdash; reports the [`DType`](burn::tensor::DType) a value
//!   works in, such as the precision a model's parameters were loaded at.
//! - [`DTypeMapper`] &mdash; a [`ModuleMapper`](burn::module::ModuleMapper)
//!   that casts every float parameter of a built module to one `DType`
//!   (`model.map(&mut DTypeMapper::new(DType::F16))`).
//! - [`reflection`] &mdash; (feature `reflection`) a queryable XML view of a
//!   built module tree, for selecting parameters with `XPath`.
//!
//! `ModuleInit` and `ToStructureConfig` are also in [`crate::prelude`].

#[cfg(feature = "reflection")]
pub mod reflection;

mod has_dtype;
mod module_init;
mod to_structure_config;
mod type_mapper;

pub use has_dtype::*;
pub use module_init::*;
pub use to_structure_config::*;
pub use type_mapper::*;
