//! # Datasets: shard sets
//!
//! A data source is a build of a dataset contract in the catalog, such as
//! `parquet-shards` ([`PARQUET_SHARDS_CONTRACT`]), whose [`SHARDS`] slot is
//! a templated resource: a family of numbered files under one name,
//! `shard_00000.parquet` … `shard_01822.parquet`, named by a template and an
//! index width, pinned by `cargo catalog` from its Hub repo's listing. A
//! [`ShardSetDescriptor`] is the view of one: its name, where its shards
//! come from, how they are named and numbered, and whether they are pinned
//! ([`ShardSetDescriptor::from_build`], [`ShardSetDescriptor::from_builtin`]).
//! With the `cache` feature, [`ShardSet`] binds one to a directory and
//! brings shards in through the disk cache. Sets of interest are named by
//! the kit that trains on them, such as
//! [`kits::language::generation::nanochat::datasets`](crate::kits::language::generation::nanochat::datasets).
//!
//! ## Lifecycle
//!
//! 1. A build gives a [`ShardSetDescriptor`], and
//!    [`ShardSetDescriptor::select`] turns `burn` slices into [`ShardId`]s.
//! 2. [`ShardSet::in_cache`] or [`ShardSet::at_dir`] binds the descriptor to a
//!    directory.
//! 3. [`ShardSet::locate`] and [`ShardSet::fetch`] give one shard's path; with
//!    the `fetch` feature, `ShardSet::fetch_many` brings many in under a
//!    `FetchPolicy` and reports on each.
//! 4. The paths go to a data loader. `examples/train-chat` hands them to
//!    `bunsen-arrow-dataloaders`' `ChatDataLoader`.
//!
//! A pinned set carries one SHA-256 per shard ([`ShardDigests::Table`]),
//! which the fetch verifies as the bytes land. A selection of shards is
//! also a [`ResourceSet`](crate::data::resources::ResourceSet)
//! ([`ShardSetDescriptor::to_resource_set`]).
//!
//! A `webdataset-shards` set ([`WEBDATASET_SHARDS_CONTRACT`]) is a shard set
//! too: tar files, each a run of samples, whose members are a sample's files
//! (`<key>.jpg`, `<key>.cls`). With the `store` feature,
//! `try_read_webdataset` reads a shard's samples. `ImageNet`'s validation
//! images, which the image validation pulls from a gated Hub mirror, are one.

mod descriptor;
#[cfg(feature = "cache")]
mod set;
#[cfg(feature = "store")]
mod webdataset;

pub use descriptor::*;
#[cfg(feature = "cache")]
pub use set::*;
#[cfg(feature = "store")]
pub use webdataset::*;

mod contracts;

pub use contracts::*;
