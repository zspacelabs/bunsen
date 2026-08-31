#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! # bunsen-firehose - Burn-based Data Pipeline
//!
//! `bunsen-firehose` is a column-oriented data pipeline for feeding [`burn`]
//! models. You describe *what* each column is and *how* derived columns are
//! computed (a [`FirehoseTableSchema`] of [`ColumnSchema`] + [`BuildPlan`]s),
//! then run batches of rows ([`FirehoseRowBatch`]) through an executor that
//! applies the registered operators in dependency order.
//!
//! The moving parts:
//!
//! - [`core::schema`] — symbolic [`FirehoseTableSchema`]: typed columns plus
//!   the [`BuildPlan`]s that derive new columns from existing ones.
//! - [`core::rows`] — runtime data: [`FirehoseRowBatch`] / [`FirehoseRow`],
//!   read and written through the [`FirehoseRowReader`] / [`FirehoseRowWriter`]
//!   traits.
//! - [`core::operations`] — operators ([`FirehoseOperator`]), their factories,
//!   the lookup [`environment`](core::operations::environment), and the
//!   [`executor`](core::operations::executor) that runs a schema over a batch.
//! - [`ops`] — the registry of globally-registered operators and
//!   [`init_default_operator_environment`](ops::init_default_operator_environment).
//! - [`burn`] — adapters that expose a schema as a [`burn`] `Batcher`, plus
//!   dataset path-scanning helpers.
//!
//! # Example: define an operator, plan a column, run a batch
//!
//! This wires up a one-operator pipeline end to end: a custom `Add` operator is
//! registered into an environment, a `result = x + y + bias` column is planned
//! onto a schema, and a two-row batch is executed.
//!
//! ```
//! use std::sync::Arc;
//!
//! use serde::{Deserialize, Serialize};
//!
//! use bunsen_firehose::core::{
//!     FirehoseRowBatch, FirehoseRowReader, FirehoseRowTransaction, FirehoseRowWriter,
//!     FirehoseTableSchema,
//!     operations::{
//!         environment::MapOpEnvironment,
//!         executor::{FirehoseBatchExecutor, SequentialBatchExecutor},
//!         factory::SimpleConfigOperatorFactory,
//!         operator::FirehoseOperator,
//!         planner::OperationPlan,
//!         signature::{FirehoseOperatorSignature, ParameterSpec},
//!     },
//!     schema::ColumnSchema,
//! };
//! use bunsen_firehose::define_firehose_operator_id;
//!
//! // Defines `pub static ADD: &str = "fh:op://<module>::ADD";`.
//! define_firehose_operator_id!(ADD);
//!
//! /// Computes `x + y + bias`. The struct *is* the (deserializable) config.
//! #[derive(Debug, Serialize, Deserialize)]
//! struct Add {
//!     bias: i64,
//! }
//!
//! impl FirehoseOperator for Add {
//!     fn apply_to_row(&self, txn: &mut FirehoseRowTransaction) -> anyhow::Result<()> {
//!         let x = txn.expect_get_parsed::<i64>("x");
//!         let y = txn.expect_get_parsed::<i64>("y");
//!         txn.expect_set_serialized("result", x + y + self.bias);
//!         Ok(())
//!     }
//! }
//!
//! fn main() -> anyhow::Result<()> {
//!     // An environment maps operator ids to factories that build operators
//!     // from their (JSON) config.
//!     let factory: Arc<SimpleConfigOperatorFactory<Add>> =
//!         Arc::new(SimpleConfigOperatorFactory::new(
//!             FirehoseOperatorSignature::from_operator_id(ADD)
//!                 .with_description("Adds x + y + bias")
//!                 .with_input(ParameterSpec::new::<i64>("x"))
//!                 .with_input(ParameterSpec::new::<i64>("y"))
//!                 .with_output(ParameterSpec::new::<i64>("result")),
//!         ));
//!     let env = Arc::new(MapOpEnvironment::from_operators(vec![factory])?);
//!
//!     // Start from the base columns, then *plan* the derived `result` column.
//!     // Planning validates the operator's parameter types against the schema
//!     // and appends both the new column and its build plan.
//!     let mut schema = FirehoseTableSchema::from_columns(&[
//!         ColumnSchema::new::<i64>("x"),
//!         ColumnSchema::new::<i64>("y"),
//!     ]);
//!     OperationPlan::for_operation_id(ADD)
//!         .with_input("x", "x")
//!         .with_input("y", "y")
//!         .with_output("result", "result")
//!         .with_config(Add { bias: 100 })
//!         .apply_to_schema(&mut schema, env.as_ref())?;
//!     let schema = Arc::new(schema);
//!
//!     // A `SequentialBatchExecutor` runs every build plan in dependency order.
//!     let executor = SequentialBatchExecutor::new(schema.clone(), env.clone())?;
//!
//!     // Fill a batch with the base columns; the executor derives `result`.
//!     let mut batch = FirehoseRowBatch::new_with_size(schema.clone(), 2);
//!     batch[0].expect_set_serialized("x", 10_i64);
//!     batch[0].expect_set_serialized("y", 20_i64);
//!     batch[1].expect_set_serialized("x", -5_i64);
//!     batch[1].expect_set_serialized("y", 2_i64);
//!
//!     executor.execute_batch(&mut batch)?;
//!
//!     assert_eq!(batch[0].expect_get_parsed::<i64>("result"), 130);
//!     assert_eq!(batch[1].expect_get_parsed::<i64>("result"), 97);
//!     Ok(())
//! }
//! ```
//!
//! For a full training pipeline — image loading, augmentation, and a [`burn`]
//! `DataLoaderBuilder` driven by a
//! [`FirehoseExecutorBatcher`](burn::batcher::FirehoseExecutorBatcher) — see
//! the `resnet_tiny` example under `demos/bimm/examples`.
//!
//! [`FirehoseTableSchema`]: core::schema::FirehoseTableSchema
//! [`ColumnSchema`]: core::schema::ColumnSchema
//! [`BuildPlan`]: core::schema::BuildPlan
//! [`FirehoseRowBatch`]: core::rows::FirehoseRowBatch
//! [`FirehoseRow`]: core::rows::FirehoseRow
//! [`FirehoseRowReader`]: core::rows::FirehoseRowReader
//! [`FirehoseRowWriter`]: core::rows::FirehoseRowWriter
//! [`FirehoseOperator`]: core::operations::operator::FirehoseOperator

/// New Data Table module.
pub mod core;

/// Namespace of common operators.
pub mod ops;

/// Burn Integration Module.
pub mod burn;
pub mod utility;

/// Experimental Data Pipeline module; non-public.
#[allow(dead_code)]
mod pipeline;

/// Define a self-referential ID.
///
/// The id will be defined as a static string constant that refers to its own
/// namespace path.
///
/// # Arguments
///
/// * `$name`: The final path name of the ID to define; the rest of the name
///   will be taken from the module context.
///
/// # Example
/// ```
/// // In module "foo::bar"
/// bunsen_firehose::define_self_referential_id!(ID);
/// // pub static ID: &str = "foo::bar::ID";
/// ```
#[macro_export]
macro_rules! define_self_referential_id {
    ($name:ident) => {
        /// Self-referential ID.
        pub static $name: &str = concat!(module_path!(), "::", stringify!($name),);
    };
    ($schema:literal, $name:ident) => {
        #[doc = concat!("Self-referential '", $schema, "' ID.")]
        pub static $name: &str = concat!($schema, "://", module_path!(), "::", stringify!($name),);
    };
}
