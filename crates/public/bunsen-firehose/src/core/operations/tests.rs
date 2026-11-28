use std::{
    collections::BTreeMap,
    fmt::Debug,
    sync::Arc,
};

use bunsen::errors::{
    BunsenResult,
    LookupError,
    LookupProblem,
    testing::{
        ErrorMatcher,
        predicate,
    },
};
// use crate::define_firehose_operator_id;
use indoc::indoc;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    core::{
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
            planner::OperationPlan,
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
    },
    define_firehose_operator_id,
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
    ) -> BunsenResult<()> {
        let x = txn.maybe_get("x").unwrap().parse_as::<i32>()?;
        let y = txn.maybe_get("y").unwrap().parse_as::<i32>()?;

        let result: i32 = x + y + self.bias;

        txn.expect_set("result", FirehoseValue::serialized(result)?);

        Ok(())
    }
}

#[test]
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

    let env = Arc::new(MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap())
        as Arc<dyn FirehoseOperatorEnvironment>;

    ErrorMatcher::illegal()
        .message_contains("'x' expected type")
        .assert_err(&OperationRunner::new_for_plan(
            Arc::new(schema.clone()),
            Arc::new(schema.build_plans[0].clone()),
            env.as_ref(),
        ));
}

#[test]
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

    ErrorMatcher::illegal()
        .message_contains("'result' expected type")
        .assert_err(&OperationRunner::new_for_plan(
            Arc::new(schema.clone()),
            Arc::new(schema.build_plans[0].clone()),
            &env,
        ));
}

#[test]
fn test_environment_operator_lookup() {
    let mut env = MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap();

    assert!(env.lookup_operator_factory(ADD).is_ok());

    ErrorMatcher::lookup()
        .cause(predicate("a missing \"nope\"", |l: &LookupError| {
            l.key == "nope"
                && l.problem == LookupProblem::Missing
                && l.candidates == [ADD.to_string()]
        }))
        .assert_err(&env.lookup_operator_factory("nope").map(|_| ()));

    ErrorMatcher::illegal()
        .cause(predicate("a duplicate ADD", |l: &LookupError| {
            l.key == ADD && l.problem == LookupProblem::Duplicate
        }))
        .assert_err(&env.add_operator(add_operator_op_binding()));
}

#[test]
fn test_simple_op() -> BunsenResult<()> {
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

/// Plans the test `ADD` with inputs `a`, `b` and output `output`.
fn plan_add_to(
    schema: &mut FirehoseTableSchema,
    output: &str,
) -> BunsenResult<BuildPlan> {
    let env = MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap();
    OperationPlan::for_operation_id(ADD)
        .with_config(AddOperator { bias: 0 })
        .with_input("x", "a")
        .with_input("y", "b")
        .with_output("result", output)
        .apply_to_schema(schema, &env)
}

#[test]
fn test_plan_rejects_clashing_output_column() {
    let mut schema = FirehoseTableSchema::from_columns(&[
        ColumnSchema::new::<i32>("a"),
        ColumnSchema::new::<i32>("b"),
        ColumnSchema::new::<i32>("c"),
    ]);
    let before = schema.clone();

    ErrorMatcher::illegal()
        .display_contains("duplicate column \"c\"")
        .assert_err(&plan_add_to(&mut schema, "c"));
    assert_eq!(schema, before);
}

#[test]
fn test_plan_rejects_non_identifier_output_column() {
    let mut schema = FirehoseTableSchema::from_columns(&[
        ColumnSchema::new::<i32>("a"),
        ColumnSchema::new::<i32>("b"),
    ]);
    let before = schema.clone();

    ErrorMatcher::illegal()
        .display_contains("Invalid identifier: 'not an ident'")
        .assert_err(&plan_add_to(&mut schema, "not an ident"));
    assert_eq!(schema, before);
}

#[test]
fn test_plan_adds_new_output_column() {
    let mut schema = FirehoseTableSchema::from_columns(&[
        ColumnSchema::new::<i32>("a"),
        ColumnSchema::new::<i32>("b"),
    ]);

    let plan = plan_add_to(&mut schema, "c").unwrap();
    assert_eq!(schema.build_plans, vec![plan]);
    assert!(schema.column_index("c").is_some());
}
