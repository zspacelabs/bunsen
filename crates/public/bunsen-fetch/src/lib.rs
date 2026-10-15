//! # bunsen-fetch
//!
//! Bunsen's file fetching, without the rest of bunsen: bunsen's disk cache
//! fetches through it at run time, and the `bunsen-bundled-*` crates' build
//! scripts at build time, so both share one implementation and one cache
//! layout.
//!
//! - `fetch_file` / `fetch_verified` (feature `fetch`): a URL streamed to disk,
//!   hashed as it lands ([`HashingReader`]), renamed into place only once it
//!   checks out, and reported to [`TransferObserver`]s.
//! - [`sha256_of`]: a file's digest.
//! - [`link_or_copy`], [`place_file`] and [`replace_with_link`]: putting a
//!   verified file where it belongs.
//! - [`pretrained_cache_path`]: where bunsen's pretrained cache keeps a pinned
//!   file.
//! - `Asset` (feature `fetch`): a pinned file a build script needs, fetched
//!   into `OUT_DIR` or taken from a local override.
//!
//! Errors are plain: an [`IoAt`] names the local operation and path, and a
//! `FetchError` the network side. bunsen sorts both into its own error
//! kinds.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

mod digest;
mod files;
mod layout;
mod transfer;

#[cfg(feature = "fetch")]
mod asset;
#[cfg(feature = "fetch")]
mod fetch;

#[cfg(feature = "fetch")]
pub use asset::*;
pub use digest::*;
#[cfg(feature = "fetch")]
pub use fetch::*;
pub use files::*;
pub use layout::*;
pub use transfer::*;
