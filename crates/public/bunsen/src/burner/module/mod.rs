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
//! - [`InputDTypes`], [`CoercesInput`], [`ResultDType`] and [`DTypeWrapped`]
//!   &mdash; a block's float dtypes: the one it computes in and casts its
//!   inputs to, and the one its output is in (see [`InputDTypes`]).
//! - [`HasDType`] &mdash; reports the [`DType`](burn::tensor::DType) a value
//!   that is not a block works in: a simulation's state, a dynamic tensor.
//! - [`DTypeMapper`] &mdash; a [`ModuleMapper`](burn::module::ModuleMapper)
//!   that casts every float parameter of a built module to one `DType`
//!   (`model.map(&mut DTypeMapper::new(DType::F16))`).
//! - [`reflection`] &mdash; (feature `reflection`) a queryable XML view of a
//!   built module tree, for selecting parameters with `XPath`.
//! - [`Rekey`] &mdash; a checkpoint's tensor names to a module's paths, built
//!   from rules: a kit routes a checkpoint's prefixes onto its model's fields
//!   and mounts each block's own map for the checkpoint's format there.
//! - [`CanonicalKeys`] &mdash; a module's canonical weight names, and the map
//!   from them onto its record paths: a kit's map lands a checkpoint on the
//!   canonical names, and [`Rekey::then`] the module's map takes them on.
//!
//! - [`Forward1Module`] … [`Forward4Module`] (and [`ForwardModule`], the
//!   first's common name) &mdash; a module's forward by its number of
//!   arguments, for code generic over the modules it calls ([Forward
//!   modules](#forward-modules)).
//! - [`Gate`] &mdash; a view of two modules as a gate, `a(x) ⊙ b(gate)`, its
//!   product taken in a [`Widening`](crate::burner::tensor::Widening).
//!
//! `ModuleInit` and `ToStructureConfig` are also in [`crate::prelude`].
//!
//! ## Forward modules
//!
//! burn gives a module's forward no trait: each module has an inherent
//! `forward`, of its own arguments. [`Forward1Module`] … [`Forward4Module`]
//! name one by its number of arguments, `ForwardN<A1, …, AN>`, its result
//! the associated `Output`; `forward` takes `&self`. They stand in for the
//! `Fn` traits, which a type cannot implement on stable Rust; they are
//! restricted to modules (a [`Module`](burn::module::Module) supertrait),
//! and give way to the `Fn` traits when a module can implement those.
//!
//! A module implements them by delegating to its inherent `forward`, which
//! method calls on the concrete type still resolve to first. Code bounded
//! on two of them for one type names the trait it calls,
//! `Forward1Module::forward(module, x)`: the method's name is the same.
//!
//! The norm and activation seams implement [`Forward1Module`] at every rank.
//! A view such as [`Gate`] borrows modules, and is not one.

#[cfg(feature = "reflection")]
pub mod reflection;

mod canonical_keys;
mod dtype_traits;
mod forward_module;
mod gate;
mod has_dtype;
mod module_init;
mod rekey;
mod tensor_load_transforms;
mod to_structure_config;
mod type_mapper;

pub use canonical_keys::*;
pub use dtype_traits::*;
pub use forward_module::*;
pub use gate::*;
pub use has_dtype::*;
pub use module_init::*;
pub use rekey::*;
pub use tensor_load_transforms::*;
pub use to_structure_config::*;
pub use type_mapper::*;
