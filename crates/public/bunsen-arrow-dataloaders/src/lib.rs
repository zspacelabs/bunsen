#![warn(missing_docs)]
//! # bunsen-preview-chat-dataloader

/// Arrow / Parquet streaming building blocks.
///
/// Provides shard readers, column selection, and rebatching utilities used
/// to feed [`tokens`] from on-disk Parquet data.
pub mod arrow;

/// Various [`burn::data::dataloader::DataLoader`] implementations.
pub mod dataloaders;

/// Generic iterator adapters used to compose the data pipeline.
pub mod iterators;

/// Tokenization and dense-block packing utilities.
pub mod tokens;
