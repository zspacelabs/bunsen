//! # Burn integration
//!
//! Bridges the firehose pipeline to [`burn`]'s data-loading stack:
//!
//! - [`batcher`] —
//!   [`FirehoseExecutorBatcher`](crate::burn_support::batcher::FirehoseExecutorBatcher),
//!   a [`burn`] `Batcher` that turns a `Vec<I>` of dataset items into a
//!   [`FirehoseRowBatch`](crate::core::FirehoseRowBatch) (via a
//!   [`BatcherInputAdapter`](crate::burn_support::batcher::BatcherInputAdapter)),
//!   runs it through a
//!   [`FirehoseBatchExecutor`](crate::core::operations::executor::FirehoseBatchExecutor),
//!   and materializes tensors out the other side (via a
//!   [`BatcherOutputAdapter`](crate::burn_support::batcher::BatcherOutputAdapter)).
//! - [`path_scanning`] — helpers such as
//!   [`image_dataset_for_folder`](crate::burn_support::path_scanning::image_dataset_for_folder)
//!   that scan `$ROOT/$CLASS/$IMG.{jpg,png}` layouts into a `burn` dataset.
//!
//! # Wiring sketch
//!
//! The batcher is generic over the dataset item type `I` and the produced
//! tensor batch `O`; you supply adapters for each side and a schema-driven
//! executor in the middle, then hand it to a `DataLoaderBuilder`:
//!
//! ```ignore
//! let batcher = FirehoseExecutorBatcher::new(
//!     Arc::new(SequentialBatchExecutor::new(schema.clone(), env.clone())?),
//!     Arc::new(MyInputAdapter::new(schema.clone())),
//!     Arc::new(MyOutputAdapter::default()),
//! );
//!
//! let loader = DataLoaderBuilder::new(batcher)
//!     .batch_size(batch_size)
//!     .shuffle(seed)
//!     .build(dataset);
//! ```
//!
//! See the `resnet_tiny` example (`examples/resnet_tiny`) for a complete,
//! compiling training pipeline built on these pieces.

/// The burn `Batcher` that runs a firehose executor, and its adapters.
pub mod batcher;
/// Util functions for path scanning.
pub mod path_scanning;
