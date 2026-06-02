//! Arrow / Parquet streaming building blocks.
//!
//! * [read_parquet_shards](crate::arrow::read_parquet_shards) flattens an
//!   iterator of Parquet shard paths into a single iterator of `RecordBatch`
//!   results.
//! * [select_text_column](crate::arrow::select_text_column) projects a named
//!   UTF-8 column out of an iterator of record batches.
//! * [Rebatcher](crate::arrow::Rebatcher) re-chunks an iterator of record
//!   batches to a target batch size using Arrow's `BatchCoalescer`.

mod parquet;
mod rebatcher;
mod select_columns;

pub use parquet::*;
pub use rebatcher::*;
pub use select_columns::*;
