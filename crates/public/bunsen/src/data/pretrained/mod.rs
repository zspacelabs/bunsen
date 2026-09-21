//! # Module / Pretrained Weights

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
mod construct;
#[cfg(feature = "cache")]
mod loaded;
mod prefabs;
#[cfg(feature = "cache")]
mod pretrained_ref;
mod providers;
mod resource;
mod resource_map;
mod rows;
mod weights;

#[cfg(feature = "cache")]
#[doc(inline)]
pub use cache::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use construct::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use loaded::*;
#[doc(inline)]
pub use prefabs::*;
#[cfg(feature = "cache")]
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
#[doc(inline)]
pub use weights::*;

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
