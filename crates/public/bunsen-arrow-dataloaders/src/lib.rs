#![warn(missing_docs)]
//! # bunsen-arrow-dataloaders
//!
//! Streams a text corpus stored as Parquet shards into [`burn`] training
//! batches for a language model:
//!
//! ```text
//! Parquet shards -> a text column -> tokens -> dense token blocks -> shuffled batches
//! ```
//!
//! [`ChatDataLoader`] runs that pipeline as a burn
//! [`DataLoader`](burn::data::dataloader::DataLoader). Each item is a
//! `[batch_size, batch_seq_len]` `Tensor<B, 2, Int>` of token ids, packed
//! end to end, with no padding.
//!
//! ## Preview
//!
//! This is a *preview* crate. It was built for one consumer,
//! `examples/train-chat`, which trains bunsen's `NanoChat` GPT, and its API is
//! shaped by that example's needs. Expect it to change between releases.
//!
//! ## The pipeline
//!
//! [`dataloaders::chat`] draws the pipeline with the item type of each
//! stage. The stages are public, and compose on their own:
//!
//! 1. [`read_parquet_shards`](arrow::read_parquet_shards) opens each shard path
//!    in turn and streams its Arrow record batches.
//! 2. [`select_text_column`](arrow::select_text_column) takes one string column
//!    out of each batch, skipping nulls. [`ChatDataLoader`] reads the column
//!    named `text`.
//! 3. [`tokenize_text_batches`](tokens::tokenize_text_batches) encodes each
//!    text with a [`wordchipper::Tokenizer`].
//! 4. [`DenseTokenBlockBatcher`](tokens::DenseTokenBlockBatcher) packs the
//!    token sequences, each bracketed by optional BOS and EOS tokens, into full
//!    `[batch_size, batch_seq_len]` blocks. When no buffered sequence fits the
//!    rest of a row, the shortest is cut to fill it, and a partial last block
//!    is dropped.
//! 5. With an rng, each epoch shuffles the shard order with it, and a
//!    [`ShuffleIter`](iterators::ShuffleIter) mixes the blocks through a
//!    128-block reservoir. Without one, shards and blocks keep their order.
//! 6. Each block becomes a tensor on the loader's device.
//!
//! [`IterWatcher`](iterators::IterWatcher)s between the stages count
//! shards, bytes and tokens into [`EpochStats`], and burn's progress counts
//! shards: [`num_items`](burn::data::dataloader::DataLoader::num_items) is
//! the number of shard paths, and `slice` splits the loader by shard.
//!
//! The stages pass a read, decode or tokenizer error along as an `Err`
//! item, and the last stage unwraps it, so the epoch's iterator panics:
//! burn's data-loader iterators yield tensors, not results. A shard without
//! a `text` column of Arrow `Utf8` strings panics in the column select.
//!
//! ## Where the shard paths come from
//!
//! The crate takes a list of shard paths and does not depend on bunsen. In
//! this workspace the paths come from bunsen's
//! [`data::shards`](https://docs.rs/bunsen/latest/bunsen/data/shards/index.html):
//! a [`ShardSet`](https://docs.rs/bunsen/latest/bunsen/data/shards/struct.ShardSet.html)
//! binds a named, optionally pinned corpus to a directory, fetches the
//! selected shards into it, and returns their local paths.
//!
//! `examples/train-chat` joins the two. It selects shards of `NanoChat`'s
//! corpus from its command line and fetches them through a `ShardSet`. It
//! keeps the last tenth of the paths (at least one) for validation, and
//! builds two [`ChatDataLoader`]s: the training loader with a seeded rng,
//! the validation loader without one. This crate's `examples/dl-test` runs
//! one epoch of one loader the same way and reports its throughput.
//!
//! ## Module map
//!
//! - [`dataloaders`]: the loaders; today one, [`ChatDataLoader`].
//! - [`arrow`]: Parquet reading, column selection and rebatching.
//! - [`tokens`]: tokenization and dense-block packing.
//! - [`iterators`]: the iterator adapters the pipeline is built with.
//!
//! [`ChatDataLoader`]: dataloaders::chat::ChatDataLoader
//! [`EpochStats`]: dataloaders::chat::EpochStats

pub mod arrow;
pub mod dataloaders;
pub mod iterators;
pub mod tokens;
