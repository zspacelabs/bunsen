//! # Local disk cache
//!
//! Where files land ([`BunsenDiskCache`], resolved from options and the
//! environment), the fetch that puts them there ([`fetch_file`] and its
//! pinned form [`fetch_verified`], with [`sha256_of`] and
//! [`link_or_copy`]), a policy-driven batch of them
//! ([`BunsenDiskCache::fetch_many`], reported as a [`FetchReport`]), and the
//! [`TransferObserver`] stack every transfer is reported to. The fetch and the
//! batch need the `fetch` feature; the rest is local.
mod digest;
mod disk_cache;
#[cfg(feature = "fetch")]
mod fetch;
#[cfg(feature = "fetch")]
mod fetch_many;
mod files;
#[cfg(feature = "indicatif")]
mod indicatif_observer;
mod path_resolver;
mod path_utils;
mod transfer;

#[doc(inline)]
pub use digest::*;
#[doc(inline)]
pub use disk_cache::*;
#[cfg(feature = "fetch")]
#[doc(inline)]
pub use fetch::*;
#[cfg(feature = "fetch")]
#[doc(inline)]
pub use fetch_many::*;
#[doc(inline)]
pub use files::*;
#[cfg(feature = "indicatif")]
#[doc(inline)]
pub use indicatif_observer::*;
#[doc(inline)]
pub use path_resolver::*;
#[doc(inline)]
pub use path_utils::*;
#[doc(inline)]
pub use transfer::*;
