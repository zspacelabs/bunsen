//! # Local disk cache
//!
//! Where files land ([`BunsenDiskCache`], resolved from options and the
//! environment), the digest-verified fetch that puts them there
//! ([`fetch_verified`], [`sha256_of`], [`link_or_copy`]), and the
//! [`TransferObserver`] stack every transfer is reported to.
mod digest;
mod disk_cache;
mod downloader_progress;
mod fetch;
#[cfg(feature = "indicatif")]
mod indicatif_observer;
mod path_resolver;
mod path_utils;
mod transfer;

#[doc(inline)]
pub use digest::*;
#[doc(inline)]
pub use disk_cache::*;
#[doc(inline)]
pub use fetch::*;
#[cfg(feature = "indicatif")]
#[doc(inline)]
pub use indicatif_observer::*;
#[doc(inline)]
pub use path_resolver::*;
#[doc(inline)]
pub use path_utils::*;
#[doc(inline)]
pub use transfer::*;
