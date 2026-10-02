//! # Local disk cache
//!
//! Where files land ([`BunsenDiskCache`], resolved from options and the
//! environment), digests and links ([`sha256_of`], [`link_or_copy`]), and the
//! [`TransferObserver`] stack every transfer is reported to. All of that is
//! local; reaching the network is the `fetch` feature.
#![cfg_attr(
    feature = "fetch",
    doc = "",
    doc = "With `fetch`: the fetch that puts a file in place ([`fetch_file`], and its",
    doc = "pinned form [`fetch_verified`]) and a policy-driven batch of them",
    doc = "([`BunsenDiskCache::fetch_many`], reported as a [`FetchReport`])."
)]
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

pub use digest::*;
pub use disk_cache::*;
#[cfg(feature = "fetch")]
pub use fetch::*;
#[cfg(feature = "fetch")]
pub use fetch_many::*;
pub use files::*;
#[cfg(feature = "indicatif")]
pub use indicatif_observer::*;
pub use path_resolver::*;
pub use path_utils::*;
pub use transfer::*;
