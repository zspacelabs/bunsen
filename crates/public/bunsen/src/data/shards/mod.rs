//! # Shard sets
//!
//! A shard set is a family of numbered files under one name:
//! `shard_00000.parquet` … `shard_01821.parquet` at one or more base URLs,
//! named by a template and an index width. The types here say *what* a set
//! is: its name, where its shards come from, how they are named and
//! numbered, and whether they are pinned. With the `cache` feature,
//! [`ShardSet`] binds one to a directory and brings shards in through the
//! disk cache.
//!
//! The pattern is [`crate::data::pretrained`]'s: a static twin for
//! compiled-in tables ([`StaticShardSetDescriptor`], [`StaticShardSetMap`])
//! and an owned twin built at runtime or converted from the static one
//! ([`ShardSetDescriptor`], [`ShardSetMap`]). Sets of interest live with the
//! kit that trains on them, such as
//! [`kits::gpts::nanochat::datasets`](crate::kits::gpts::nanochat::datasets).
//!
//! ## Lifecycle
//!
//! 1. A kit's [`StaticShardSetMap`] names its sets; a lookup gives a
//!    [`ShardSetDescriptor`], and [`ShardSetDescriptor::select`] turns `burn`
//!    slices into [`ShardId`]s.
//! 2. [`ShardSet::in_cache`] or [`ShardSet::at_dir`] binds the descriptor to a
//!    directory.
//! 3. [`ShardSet::locate`] and [`ShardSet::fetch`] give one shard's path; with
//!    the `fetch` feature, `ShardSet::fetch_many` brings many in under a
//!    `FetchPolicy` and reports on each.
//! 4. The paths go to a data loader. `examples/train-chat` hands them to
//!    `bunsen-arrow-dataloaders`' `ChatDataLoader`.
//!
//! A pinned set carries one SHA-256 per shard ([`ShardDigests::Table`]),
//! which the fetch verifies as the bytes land. The digest tables in the
//! tree are generated from the source's listing by
//! `tools/gen_shard_digests.py`. A selection of shards is also a
//! [`ResourceMap`](crate::data::pretrained::ResourceMap)
//! ([`ShardSetDescriptor::to_resource_map`]), for a pretrained row that
//! fuses a shard set in.

mod descriptor;
mod map;
#[cfg(feature = "cache")]
mod set;

pub use descriptor::*;
pub use map::*;
#[cfg(feature = "cache")]
pub use set::*;
