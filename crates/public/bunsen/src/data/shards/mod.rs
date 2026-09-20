//! # Shard sets
//!
//! A shard set is a family of numbered files under one name:
//! `shard_00000.parquet` … `shard_01821.parquet` at one or more base URLs,
//! named by a template and an index width. The types here say *what* a set
//! is; with the `cache` feature, [`ShardSet`] binds one to a directory and
//! brings shards in through the disk cache.
//!
//! The pattern is [`crate::data::pretrained`]'s: a static twin for
//! compiled-in tables ([`StaticShardSetDescriptor`], [`StaticShardSetMap`])
//! and an owned twin built at runtime or converted from the static one
//! ([`ShardSetDescriptor`], [`ShardSetMap`]). Sets of interest live with the
//! kit that trains on them, such as
//! [`kits::gpts::nanochat::datasets`](crate::kits::gpts::nanochat::datasets).

mod descriptor;
mod map;
#[cfg(feature = "cache")]
mod set;

#[doc(inline)]
pub use descriptor::*;
#[doc(inline)]
pub use map::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use set::*;
