//! # Operators: from signature to executor
//!
//! An operator is one step of a pipeline: it reads some columns of a row
//! and writes others. This module holds its lifecycle, a file per stage:
//!
//! 1. [`signature`]: a [`FirehoseOperatorSignature`] carries the operator's id
//!    and its input and output parameters, each a [`ParameterSpec`] with a name
//!    and a data type.
//! 2. [`factory`]: a [`FirehoseOperatorFactory`] holds a signature and builds
//!    the operator for one build plan. The usual factory is
//!    [`SimpleConfigOperatorFactory<T>`](factory::SimpleConfigOperatorFactory),
//!    where the operator type `T` is its own config, deserialized from the
//!    plan.
//! 3. [`environment`] and [`registration`]: a [`FirehoseOperatorEnvironment`]
//!    maps operator ids to factories. Fill a [`MapOpEnvironment`] by hand, or
//!    collect every factory that any linked crate registered with
//!    [`define_firehose_operator!`] by calling
//!    [`init_default_operator_environment`].
//! 4. [`planner`]: an [`OperationPlan`] is one call of an operator: its id, the
//!    parameter-to-column bindings, and the config. Its
//!    [`apply_to_schema`](planner::OperationPlan::apply_to_schema) checks the
//!    call, adds its output columns, typed from the signature, and records it
//!    on the [`FirehoseTableSchema`] as a [`BuildPlan`], the serializable form
//!    of the call.
//! 5. [`operator`]: a [`FirehoseOperator`] is the built step. An
//!    [`OperationRunner`] binds one to its schema and build plan, and runs it
//!    over a batch.
//! 6. [`executor`]: a [`FirehoseBatchExecutor`] builds a runner for each build
//!    plan of a schema, in dependency order, and runs them over every
//!    [`FirehoseRowBatch`] it is given. [`SequentialBatchExecutor`] is the
//!    implementation, on the calling thread.
//!
//! The crate root's example walks all six stages with one operator.
//!
//! ## What is checked, and when
//!
//! Planning does the checking. [`OperationPlan::apply_to_schema`] tries the
//! call on a copy of the schema first:
//!
//! - the operator id must be in the environment;
//! - the bindings must name exactly the signature's parameters, and each bound
//!   column's type must be the parameter's, compared by
//!   [`type_name`](std::any::type_name);
//! - each output column must be a new, valid identifier, and every column must
//!   stay computable: one plan per output, and no cycles;
//! - the factory then builds the operator once, so a config that does not
//!   deserialize fails here too.
//!
//! Each failed check is an `Err`, and only a call that passes changes the
//! schema. An executor built from a schema that did not come through
//! planning, such as one read from JSON, makes the same type checks when it
//! builds its runners.
//!
//! ## Running a plan
//!
//! An operator implements
//! [`apply_to_row`](operator::FirehoseOperator::apply_to_row), or overrides
//! [`apply_to_batch`](operator::FirehoseOperator::apply_to_batch) to see the
//! whole batch. It reads and writes through a transaction
//! ([`FirehoseRowTransaction`], [`FirehoseBatchTransaction`]) by *parameter*
//! name. The build plan maps those names to columns, so one operator serves
//! any column names. Reading a parameter that is not an input, or writing
//! one that is not an output, panics. Writes are held aside and committed to
//! the batch only when the operator returns `Ok`, so a failed operator leaves
//! its output columns as they were.
//!
//! ## Registration
//!
//! [`define_firehose_operator!`] defines an id,
//! `"fh:op://<module_path>::<NAME>"`, and registers a factory under it with
//! [`inventory`]. The registration is part of every binary that links the
//! crate, so a crate publishes operators just by being linked in. See
//! [`ops`](crate::ops).
//!
//! [`FirehoseOperatorSignature`]: signature::FirehoseOperatorSignature
//! [`ParameterSpec`]: signature::ParameterSpec
//! [`FirehoseOperatorFactory`]: factory::FirehoseOperatorFactory
//! [`FirehoseOperatorEnvironment`]: environment::FirehoseOperatorEnvironment
//! [`MapOpEnvironment`]: environment::MapOpEnvironment
//! [`define_firehose_operator!`]: crate::define_firehose_operator
//! [`init_default_operator_environment`]: crate::ops::init_default_operator_environment
//! [`OperationPlan`]: planner::OperationPlan
//! [`OperationPlan::apply_to_schema`]: planner::OperationPlan::apply_to_schema
//! [`FirehoseTableSchema`]: crate::core::schema::FirehoseTableSchema
//! [`BuildPlan`]: crate::core::schema::BuildPlan
//! [`FirehoseOperator`]: operator::FirehoseOperator
//! [`OperationRunner`]: operator::OperationRunner
//! [`FirehoseBatchExecutor`]: executor::FirehoseBatchExecutor
//! [`SequentialBatchExecutor`]: executor::SequentialBatchExecutor
//! [`FirehoseRowBatch`]: crate::core::rows::FirehoseRowBatch
//! [`FirehoseRowTransaction`]: crate::core::rows::FirehoseRowTransaction
//! [`FirehoseBatchTransaction`]: crate::core::rows::FirehoseBatchTransaction

mod macros;

/// Stage 3: operator lookup environments, id to factory.
pub mod environment;
/// Stage 6: executors, which run a schema's build plans over a batch.
pub mod executor;
/// Stage 2: operator factories, which build an operator for a build plan.
pub mod factory;
/// Stage 5: the runtime operator interface, and the runner that applies it.
pub mod operator;
/// Stage 4: operation plans, which add an operator call to a schema.
pub mod planner;
/// Stage 3: global operator registration, through `inventory`.
pub mod registration;
/// Stage 1: operator signatures and parameter specifications.
pub mod signature;

#[cfg(test)]
mod tests;
