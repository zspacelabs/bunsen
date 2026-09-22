//! # Pretrained models
//!
//! A pretrained is a row over a [`ResourceMap`]: keyed files, each with
//! its digest and the places it can be had from. Rows sit in
//! [`PretrainedGroup`]s behind a [`PretrainedTable`], the compiled-in
//! [`PretrainedProvider`]. With the `cache` feature, a [`PretrainedFactory`]
//! holds a kit's providers in search order and resolves a name,
//! `provider:ref` or bare, to a [`Deferred`] model: the row's map
//! ([`PretrainedRef`]) with the kit's [`Construct`] hook chosen for it by
//! what the map says about its resources. Loading it, the
//! [`PretrainedCache`] brings the map's files local and the hook builds
//! from them, behind an `Arc`.
//! [`StaticPreFabMap`] is the other half: the geometries a kit knows by name.

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
mod construct;
#[cfg(feature = "cache")]
mod deferred;
#[cfg(feature = "cache")]
mod factory;
mod hf;
#[cfg(feature = "cache")]
mod loaded;
mod prefabs;
mod pretrained_ref;
mod providers;
mod resource;
mod resource_map;
mod rows;
#[cfg(feature = "store_safetensors")]
mod safetensors;

#[cfg(feature = "cache")]
#[doc(inline)]
pub use cache::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use construct::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use deferred::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use factory::*;
#[doc(inline)]
pub use hf::*;
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
#[cfg(feature = "store_safetensors")]
#[doc(inline)]
pub use safetensors::*;

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
