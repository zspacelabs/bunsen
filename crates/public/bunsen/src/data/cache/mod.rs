//! # Local disk cache
//!
//! Where files land ([`BunsenDiskCache`], resolved from options and the
//! environment), digests and links ([`sha256_of`], [`link_or_copy`]), and the
//! [`TransferObserver`] stack every transfer is reported to. All of that is
//! local; reaching the network is the `fetch` feature.
//!
//! ## Where files land
//!
//! A [`BunsenDiskCache`] has two roots: a cache directory, for files that
//! can be fetched again, and a data directory, for datasets. Its
//! [`BunsenDiskCacheOptions`] may name either one; otherwise
//! [`BUNSEN_CACHE_CONFIG`] resolves them, from `BUNSEN_CACHE_DIR` and
//! `BUNSEN_DATA_DIR` first and then the platform's directories.
//! [`cache_path`](BunsenDiskCache::cache_path) and
//! [`data_path`](BunsenDiskCache::data_path) name a file under either root
//! without touching the disk.
//!
//! The layers above choose the paths. The
//! [`PretrainedCache`](crate::data::pretrained::PretrainedCache) keeps
//! digest-pinned files under `<cache>/pretrained/`, and a
//! [`ShardSet`](crate::data::shards::ShardSet) keeps a set's shards under
//! `<data>/shards/<name>/`.
//!
//! ## Digests and files
//!
//! SHA-256 over files and streams ([`sha256_of`], [`verify_sha256`],
//! [`HashingReader`]) serves digest-pinned entries. A file whose path
//! carries its digest was verified when it was written, and is trusted on
//! later runs without re-hashing it: the pretrained cache's
//! [trust model](crate::data::pretrained::PretrainedCache#trust-model).
//!
//! The file helpers name and place files in the cache: the `.partial`
//! beside a destination that a write goes to before it is renamed into
//! place ([`partial_path`]), a link where a copy would do
//! ([`link_or_copy`]), and the file a URL names ([`file_name_from_url`]).
//! None of them reaches the network.
//!
//! ## Transfers
//!
//! A transfer is one file moving into the cache: a download today, and a
//! copy or a verification pass later. Observers watch transfers without
//! taking part in them. The disk cache carries a stack of them
//! ([`BunsenDiskCacheOptions::transfer_observers`]) and tells every
//! observer about every transfer, in registration order. With the
//! `indicatif` feature, the default stack is one progress bar per
//! transfer, drawn on stderr.
//!
//! ## Fetching
//!
//! This module is the one place bunsen reaches the network at run time,
//! and only with the `fetch` feature. Without it nothing here downloads,
//! and a file that only a URL can supply is not found.
#![cfg_attr(
    feature = "fetch",
    doc = "",
    doc = "With it, [`fetch_file`] streams a URL into the cache, hashed as it lands, and",
    doc = "renames the file into place only once it checks out; [`fetch_verified`] is its",
    doc = "pinned form. [`BunsenDiskCache::fetch_many`] runs a batch of [`FetchJob`]s under a",
    doc = "[`FetchPolicy`] and reports on each in a [`FetchReport`]."
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
