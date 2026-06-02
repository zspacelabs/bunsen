//! # Core data model
//!
//! The `core` module holds the schema and runtime types of the pipeline:
//!
//! - [schema](crate::core::schema) — the symbolic
//!   [FirehoseTableSchema](crate::core::schema::FirehoseTableSchema): typed
//!   [ColumnSchema](crate::core::schema::ColumnSchema)s plus the
//!   [BuildPlan](crate::core::schema::BuildPlan)s that derive columns.
//! - [rows](crate::core::rows) — runtime data
//!   ([FirehoseRowBatch](crate::core::rows::FirehoseRowBatch) /
//!   [FirehoseRow](crate::core::rows::FirehoseRow)), accessed through the
//!   [FirehoseRowReader](crate::core::rows::FirehoseRowReader) /
//!   [FirehoseRowWriter](crate::core::rows::FirehoseRowWriter) traits.
//! - [values](crate::core::values) —
//!   [FirehoseValue](crate::core::values::FirehoseValue), the per-cell sum type
//!   of "serialized JSON" vs "boxed `Any`".
//! - [operations](crate::core::operations) — operators, factories,
//!   environments, and the executor that runs a schema over a batch.
//!
//! # Example: build a schema and fill a batch
//!
//! A schema is a typed column list; a batch is a growable set of rows over that
//! schema. Values are set and read by column name.
//!
//! ```
//! use std::sync::Arc;
//!
//! use bunsen_firehose::core::{
//!     FirehoseRowBatch,
//!     FirehoseRowReader,
//!     FirehoseRowWriter,
//!     FirehoseTableSchema,
//!     schema::ColumnSchema,
//! };
//!
//! let schema = Arc::new(FirehoseTableSchema::from_columns(&[
//!     ColumnSchema::new::<String>("path")
//!         .with_description("source image path"),
//!     ColumnSchema::new::<i32>("label"),
//! ]));
//!
//! let mut batch = FirehoseRowBatch::new(schema.clone());
//! let row = batch.new_row();
//! row.expect_set_serialized("path", "cat/0001.png");
//! row.expect_set_serialized("label", 3_i32);
//!
//! assert_eq!(batch.len(), 1);
//! assert_eq!(batch[0].expect_get_parsed::<String>("path"), "cat/0001.png");
//! assert_eq!(batch[0].expect_get_parsed::<i32>("label"), 3);
//! ```
//!
//! See the crate root for an end-to-end example that derives columns with an
//! [operations](crate::core::operations)-registered operator.

/// Defines legal identifiers for firehose tables.
pub mod identifiers;
/// Defines the operator environment for firehose tables.
pub mod operations;
/// Defines rows and row batches for firehose tables.
pub mod rows;
/// Defines the symbolic schema for firehose tables.
pub mod schema;

/// Defines `ValueBox`, a sum type for Json Values and boxed values.
pub mod values;

// TODO: Work out what the `$crate::core::*` re-exports should be.
pub use rows::{
    FirehoseBatchTransaction,
    FirehoseRow,
    FirehoseRowBatch,
    FirehoseRowReader,
    FirehoseRowTransaction,
    FirehoseRowWriter,
};
pub use schema::FirehoseTableSchema;
pub use values::FirehoseValue;
