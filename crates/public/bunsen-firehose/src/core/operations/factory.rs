use std::{
    fmt::Debug,
    marker::PhantomData,
};

use bunsen::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
};
use serde::de::DeserializeOwned;

use crate::core::{
    operations::{
        operator::FirehoseOperator,
        signature::FirehoseOperatorSignature,
    },
    schema::{
        BuildPlan,
        FirehoseTableSchema,
    },
};

/// A factory for creating [`FirehoseOperator`] instances from a build plan.
///
/// It holds the operator's [`FirehoseOperatorSignature`], which planning
/// checks a call against, and builds the operator for each plan that calls
/// it.
pub trait FirehoseOperatorFactory: Debug + Send + Sync {
    /// Returns the operator ID.
    fn operator_id(&self) -> &String {
        self.signature()
            .operator_id
            .as_ref()
            .expect("Spec must have an operator id")
    }

    /// Returns the operator specification.
    fn signature(&self) -> &FirehoseOperatorSignature;

    /// Builds the operator for one build plan.
    ///
    /// # Arguments
    ///
    /// * `context` - The schema, the build plan, and this factory's signature,
    ///   already checked against each other.
    ///
    /// # Returns
    ///
    /// The boxed operator, or an error if it cannot be built from the plan
    /// (for example, a config that does not deserialize).
    fn init(
        &self,
        context: &dyn FirehoseOperatorInitContext,
    ) -> BunsenResult<Box<dyn FirehoseOperator>>;
}

/// The init interface for `FirehoseOperatorFactory`.
pub trait FirehoseOperatorInitContext {
    /// Returns the operator ID for the operator being initialized.
    fn operator_id(&self) -> &str;

    /// Returns the table schema for the operator being initialized.
    fn table_schema(&self) -> &FirehoseTableSchema;

    /// Returns a reference to the build plan.
    fn build_plan(&self) -> &BuildPlan;

    /// The operator signature for the operator being initialized.
    fn signature(&self) -> &FirehoseOperatorSignature;
}

/// A simple operator factory for types implementing `DeserializeOwned` and
/// `FirehoseOperator`: the operator is its own config, deserialized from
/// the build plan's [`config`](BuildPlan::config).
#[derive(Debug)]
pub struct SimpleConfigOperatorFactory<T>
where
    T: DeserializeOwned + FirehoseOperator,
{
    /// The operator signature.
    signature: FirehoseOperatorSignature,

    /// Phantom data to ensure the factory is generic over the operator type.
    phantom_data: PhantomData<T>,
}

impl<T> SimpleConfigOperatorFactory<T>
where
    T: DeserializeOwned + FirehoseOperator,
{
    /// Creates a new `SimpleConfigOperatorFactory` with the given operator
    /// signature.
    ///
    /// # Panics
    ///
    /// If the signature has no operator id.
    pub fn new(spec: FirehoseOperatorSignature) -> Self {
        if spec.operator_id.is_none() {
            panic!("OperatorSpec must have an operator_id");
        }
        Self {
            signature: spec,
            phantom_data: PhantomData,
        }
    }
}

impl<T> FirehoseOperatorFactory for SimpleConfigOperatorFactory<T>
where
    T: DeserializeOwned + FirehoseOperator,
{
    fn signature(&self) -> &FirehoseOperatorSignature {
        &self.signature
    }

    fn init(
        &self,
        context: &dyn FirehoseOperatorInitContext,
    ) -> BunsenResult<Box<dyn FirehoseOperator>> {
        let config = &context.build_plan().config;
        let op: T = serde_json::from_value(config.clone()).map_err(|e| {
            BunsenError::from_cause(BunsenErrorKind::Illegal, e).context_details(
                format!(
                    "deserializing the operator config for {}",
                    self.signature.operator_id.as_deref().unwrap_or("unknown"),
                ),
                format!("{config:#}"),
            )
        })?;
        Ok(Box::new(op))
    }
}

#[cfg(test)]
mod tests {
    use bunsen::errors::BunsenResult;
    use serde::{
        Deserialize,
        Serialize,
    };

    use crate::{
        core::{
            operations::{
                factory::SimpleConfigOperatorFactory,
                operator::FirehoseOperator,
                signature::{
                    FirehoseOperatorSignature,
                    ParameterSpec,
                },
            },
            rows::FirehoseRowTransaction,
        },
        define_firehose_operator_id,
    };

    define_firehose_operator_id!(TEST_OP);

    #[derive(Debug, Serialize, Deserialize)]
    struct TestOperator {
        pub value: i32,
    }

    impl FirehoseOperator for TestOperator {
        fn apply_to_row(
            &self,
            _row: &mut FirehoseRowTransaction,
        ) -> BunsenResult<()> {
            todo!()
        }
    }

    #[test]
    fn test_simple_config_operator_factory() {
        let signature = FirehoseOperatorSignature::from_operator_id(TEST_OP)
            .with_input(ParameterSpec::new::<i32>("input"))
            .with_output(ParameterSpec::new::<i32>("output"));

        let _factory = SimpleConfigOperatorFactory::<TestOperator>::new(signature);
    }

    #[should_panic(expected = "OperatorSpec must have an operator_id")]
    #[test]
    fn test_simple_config_operator_factory_without_id() {
        let signature = FirehoseOperatorSignature::default()
            .with_input(ParameterSpec::new::<i32>("input"))
            .with_output(ParameterSpec::new::<i32>("output"));

        // This should panic because the operator_id is not set.
        let _factory = SimpleConfigOperatorFactory::<TestOperator>::new(signature);
    }
}
