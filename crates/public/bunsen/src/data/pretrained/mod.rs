//! # Pretrained models
//!
//! A pretrained is a row over a [`ResourceMap`]: keyed files, each with
//! its digest and the places it can be had from. Rows sit in
//! [`PretrainedGroup`]s behind a [`PretrainedTable`], the compiled-in
//! [`PretrainedProvider`]; a [`PretrainedFactory`] holds a kit's providers
//! in search order and resolves a spec, `provider:ref` or a bare name or a
//! path, to a [`PretrainedRef`]. With the `cache` feature, the
//! [`PretrainedCache`] brings a map's files local and a kit's
//! [`Construct`] hook builds from them, behind an `Arc`. [`StaticPreFabMap`]
//! is the other half: the geometries a kit knows by name.

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
mod construct;
mod factory;
#[cfg(feature = "cache")]
mod loaded;
mod prefabs;
mod pretrained_ref;
mod providers;
mod resource;
mod resource_map;
mod rows;

#[cfg(feature = "cache")]
#[doc(inline)]
pub use cache::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use construct::*;
#[doc(inline)]
pub use factory::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use loaded::*;
#[doc(inline)]
pub use prefabs::*;
#[doc(inline)]
pub use pretrained_ref::*;
#[doc(inline)]
pub use providers::*;
#[doc(inline)]
pub use resource::*;
#[doc(inline)]
pub use resource_map::*;
#[doc(inline)]
pub use rows::*;

/// [`ResourceNotFound`](crate::errors::BunsenError::ResourceNotFound) for a
/// name a table does not have, naming what it has: `"<table>: no <kind>
/// "x"; there are: a, b"`.
pub(crate) fn not_found(
    table: Option<&str>,
    kind: &str,
    name: &str,
    names: &[&str],
) -> crate::errors::BunsenError {
    let table = table.map(|t| alloc::format!("{t}: ")).unwrap_or_default();
    let names = if names.is_empty() {
        alloc::string::String::from("(none)")
    } else {
        names.join(", ")
    };
    crate::errors::BunsenError::ResourceNotFound(alloc::format!(
        "{table}no {kind} {name:?}; there are: {names}"
    ))
}
