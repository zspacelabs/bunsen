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
//! Only a call that passes changes the schema. A new output name that is
//! already a column, or is not an identifier, panics rather than returning
//! an error. An executor built from a schema that did not come through
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

/// Combined macro to define and register a firehose operator.
///
/// # Arguments
///
/// * `$name`: The name of the operator ID to define; will create a
///   self-referential static string constant.
/// * `$constructor`: An expression that builds the operator's factory, a
///   `FirehoseOperatorFactory` (or an `Arc` of one). It is evaluated each time
///   the registration is read.
///
/// This macro combines the functionality of `define_firehose_operator_id`
/// and `register_firehose_operator_factory`.
#[macro_export]
macro_rules! define_firehose_operator {
    ($name:ident, $constructor:expr) => {
        $crate::define_firehose_operator_id!($name);
        $crate::register_firehose_operator_factory!($name, $constructor);
    };
}

/// Define a self-referential operator ID.
///
/// The id will be defined as a static string constant that refers to its own
/// namespace path.
///
/// # Arguments
///
/// * `$name`: The name of the operator ID to define.
#[macro_export]
macro_rules! define_firehose_operator_id {
    ($name:ident) => {
        $crate::define_self_referential_id!("fh:op", $name);
    };
}

/// Macro to register a default operator factory.
///
/// Builders which do not require runtime configuration can be registered
/// using this macro; and collected globally using
/// [`FirehoseOperatorFactoryRegistration::list_default_registrations`](crate::core::operations::registration::FirehoseOperatorFactoryRegistration::list_default_registrations).
///
/// You can also collect a default environment with all registered builders
/// using [`init_default_operator_environment`](crate::ops::init_default_operator_environment).
#[macro_export]
macro_rules! register_firehose_operator_factory {
    ($name:ident, $constructor:expr) => {
        inventory::submit! {
            $crate::core::operations::registration::FirehoseOperatorFactoryRegistration {
                operator_id: $name,
                supplier: || {
                    let v = ($constructor);
                    std::sync::Arc::from(v) as
                    std::sync::Arc<dyn $crate::core::operations::factory::FirehoseOperatorFactory>
                },
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fmt::Debug,
        sync::Arc,
    };

    // use crate::define_firehose_operator_id;
    use indoc::indoc;
    use serde::{
        Deserialize,
        Serialize,
    };

    use crate::core::{
        FirehoseValue,
        operations::{
            environment::{
                FirehoseOperatorEnvironment,
                MapOpEnvironment,
            },
            factory::SimpleConfigOperatorFactory,
            operator::{
                FirehoseOperator,
                OperationRunner,
                OperatorSchedulingMetadata,
            },
            signature::{
                FirehoseOperatorSignature,
                ParameterSpec,
            },
        },
        rows::{
            FirehoseRowBatch,
            FirehoseRowReader,
            FirehoseRowTransaction,
            FirehoseRowWriter,
        },
        schema::{
            BuildPlan,
            ColumnSchema,
            DataTypeDescription,
            FirehoseTableSchema,
        },
    };

    define_firehose_operator_id!(ADD);

    #[derive(Debug, Serialize, Deserialize)]
    struct AddOperator {
        bias: i32,
    }

    fn add_operator_op_binding() -> Arc<SimpleConfigOperatorFactory<AddOperator>> {
        Arc::new(SimpleConfigOperatorFactory::new(
            FirehoseOperatorSignature::new()
                .with_operator_id(ADD)
                .with_description("Adds inputs with a bias")
                .with_input(ParameterSpec::new::<i32>("x").with_description("First input"))
                .with_input(ParameterSpec::new::<i32>("y").with_description("Second input"))
                .with_output(
                    ParameterSpec::new::<i32>("result")
                        .with_description("Result of addition with bias"),
                ),
        ))
    }

    impl FirehoseOperator for AddOperator {
        fn apply_to_row(
            &self,
            txn: &mut FirehoseRowTransaction,
        ) -> anyhow::Result<()> {
            let x = txn.maybe_get("x").unwrap().parse_as::<i32>()?;
            let y = txn.maybe_get("y").unwrap().parse_as::<i32>()?;

            let result: i32 = x + y + self.bias;

            txn.expect_set("result", FirehoseValue::serialized(result)?);

            Ok(())
        }
    }

    #[test]
    #[should_panic(expected = "'x' expected type")]
    fn test_bad_input_data_type() {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<String>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<i32>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(ADD)
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env =
            Arc::new(MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap())
                as Arc<dyn FirehoseOperatorEnvironment>;

        let _builder = OperationRunner::new_for_plan(
            Arc::new(schema.clone()),
            Arc::new(schema.build_plans[0].clone()),
            env.as_ref(),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "'result' expected type")]
    fn test_bad_output_data_type() {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<String>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(ADD)
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env = MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap();

        let _builder = OperationRunner::new_for_plan(
            Arc::new(schema.clone()),
            Arc::new(schema.build_plans[0].clone()),
            &env,
        )
        .unwrap();
    }

    #[test]
    fn test_simple_op() -> anyhow::Result<()> {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<i32>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(ADD)
                    .with_description("Adds inputs with a bias")
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env = MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap();

        let runner = OperationRunner::new_for_plan(
            Arc::new(schema.clone()),
            Arc::new(schema.build_plans[0].clone()),
            &env,
        )
        .unwrap();

        assert_eq!(
            format!("{runner:#?}"),
            indoc! {r#"
               ColumnBuilder {
                   build_plan: BuildPlan {
                       operator_id: "fh:op://bunsen_firehose::core::operations::tests::ADD",
                       description: Some(
                           "Adds inputs with a bias",
                       ),
                       config: Object {
                           "bias": Number(10),
                       },
                       inputs: {
                           "x": "a",
                           "y": "b",
                       },
                       outputs: {
                           "result": "c",
                       },
                   },
               }"#,
            }
        );

        assert_eq!(
            runner.scheduling_metadata(),
            OperatorSchedulingMetadata {
                effective_batch_size: 1,
            }
        );

        assert_eq!(runner.build_plan.operator_id, ADD);

        let mut batch = FirehoseRowBatch::new_with_size(Arc::new(schema.clone()), 2);
        batch[0].expect_set("a", FirehoseValue::serialized(10)?);
        batch[0].expect_set("b", FirehoseValue::serialized(20)?);
        batch[1].expect_set("a", FirehoseValue::serialized(-5)?);
        batch[1].expect_set("b", FirehoseValue::serialized(2)?);

        runner.apply_to_batch(&mut batch).unwrap();

        assert_eq!(batch[0].maybe_get("c").unwrap().parse_as::<i32>()?, 40);
        assert_eq!(batch[1].maybe_get("c").unwrap().parse_as::<i32>()?, 7);

        Ok(())
    }

    #[test]
    fn test_operator_spec_validation() {
        let spec = FirehoseOperatorSignature::new()
            .with_input(ParameterSpec::new::<i32>("input1"))
            .with_input(ParameterSpec::new::<String>("input2"))
            .with_output(ParameterSpec::new::<f64>("output"));

        let mut input_types = BTreeMap::new();
        input_types.insert("input1".to_string(), DataTypeDescription::new::<i32>());
        input_types.insert("input2".to_string(), DataTypeDescription::new::<String>());

        let mut output_types = BTreeMap::new();
        output_types.insert("output".to_string(), DataTypeDescription::new::<f64>());

        assert!(spec.validate(&input_types, &output_types).is_ok());
    }

    #[test]
    fn test_path_ident() {
        define_firehose_operator_id!(FOO);

        assert_eq!(FOO, concat!("fh:op://", module_path!(), "::FOO"));
    }

    #[test]
    fn test_map_op_environment() {
        let _env = MapOpEnvironment::new();
    }
}
