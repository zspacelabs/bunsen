#![warn(missing_docs)]
//! # bunsen-preview-chat-dataloader
//!
//! A preview implementation of a streaming chat data loader for training
//! LLM-style models on top of the [`burn`] tensor library.
//!
//! The pipeline reads Parquet shards, selects a text column, tokenizes the
//! text with a [`wordchipper::Tokenizer`], packs the tokens into dense
//! fixed-shape blocks, optionally shuffles the blocks through a bounded
//! reservoir, and finally materializes each block as a 2D integer
//! [`burn::tensor::Tensor`].
//!
//! ## Pipeline overview
//!
//! ```text
//! shard paths
//!     -> read_parquet_shards          (Iterator<ArrowResult<RecordBatch>>)
//!     -> select_text_column           (Iterator<ArrowResult<Vec<String>>>)
//!     -> tokenize_text_batches        (Iterator<ArrowResult<Vec<Vec<u32>>>>)
//!     -> DenseTokenBlockBatcher       (Iterator<ArrowResult<Vec<Vec<u32>>>>)
//!     -> ShuffleIter (optional)
//!     -> Tensor<B, 2, Int>
//! ```
//!
//! Counters layered into the pipeline via [`iterators::IterWatcher`] feed
//! [`dataloader::EpochStats`], which drives the
//! [`burn::data::dataloader::Progress`] reporting expected by the burn
//! training loop.
//!
//! ## Example Use
//!
//! The following example builds a data loader for a chat dataset consisting
//! of a training and validation set of Parquet shards.
//!
//! The batch items are `Tensor<B, 2, Int>`, where `B` is the burn backend
//! type (e.g. `Cuda` or `Cpu`) and `Int` is the integer tensor type
//! (e.g. `Int32` or `Int64`).
//!
//! ```rust,ignore
//! let training_data_loader: ChatDataLoader<B> = ChatDataLoader::new(
//!     training_paths,
//!     Some(Arc::new(Mutex::new(StdRng::seed_from_u64(0)))),
//!     &device,
//!     tok.clone(),
//!     dl_config.clone(),
//! );
//! let validation_data_loader: ChatDataLoader<B::InnerBackend> =
//!     ChatDataLoader::new(validation_paths, None, &device, tok.clone(), dl_config);
//! ```

/// Arrow / Parquet streaming building blocks.
///
/// Provides shard readers, column selection, and rebatching utilities used
/// to feed [`crate::tokens`] from on-disk Parquet data.
pub mod arrow;

/// The top-level [`burn::data::dataloader::DataLoader`] implementation.
pub mod dataloader;

/// Generic iterator adapters used to compose the data pipeline.
pub mod iterators;

/// Tokenization and dense-block packing utilities.
pub mod tokens;

#[doc(inline)]
pub use dataloader::*;
