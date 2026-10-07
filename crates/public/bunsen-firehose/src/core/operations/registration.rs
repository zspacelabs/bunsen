use std::{
    fmt::Debug,
    sync::Arc,
};

use crate::core::operations::factory::FirehoseOperatorFactory;

// This leverages the `inventory` crate to define a collection scheme for
// later invocations of `inventory::submit! { <instance> }` by the registration
// macro.
inventory::collect!(FirehoseOperatorFactoryRegistration);

/// One global registration: an operator id and the function that builds its
/// factory.
///
/// Submitted to [`inventory`] by
/// [`register_firehose_operator_factory`](crate::register_firehose_operator_factory)
/// (and so by [`define_firehose_operator`](crate::define_firehose_operator)),
/// for factories which need no arguments to build.
#[derive(Clone)]
pub struct FirehoseOperatorFactoryRegistration {
    /// The operator ID.
    pub operator_id: &'static str,

    /// The builder.
    pub supplier: fn() -> Arc<dyn FirehoseOperatorFactory>,
}

impl Debug for FirehoseOperatorFactoryRegistration {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("FirehoseOperatorFactoryRegistration")
            .field("operator_id", &self.operator_id)
            .finish()
    }
}

impl FirehoseOperatorFactoryRegistration {
    /// Returns the builder.
    pub fn get_builder(&self) -> Arc<dyn FirehoseOperatorFactory> {
        let builder = (self.supplier)();

        assert_eq!(
            builder.operator_id(),
            self.operator_id,
            "Builder operator ID does not match registration ID: {} != {}",
            builder.operator_id(),
            self.operator_id,
        );

        builder
    }

    /// List all default operator registrations.
    pub fn list_default_registrations() -> Vec<&'static FirehoseOperatorFactoryRegistration> {
        inventory::iter::<FirehoseOperatorFactoryRegistration>
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use bunsen::errors::BunsenResult;
    use serde::{
        Deserialize,
        Serialize,
    };

    use super::*;
    use crate::{
        core::operations::{
            factory::SimpleConfigOperatorFactory,
            operator::FirehoseOperator,
            signature::{
                FirehoseOperatorSignature,
                ParameterSpec,
            },
        },
        define_firehose_operator,
    };

    define_firehose_operator!(
        EXAMPLE_OPERATOR,
        SimpleConfigOperatorFactory::<ExampleOp>::new(
            FirehoseOperatorSignature::new()
                .with_operator_id(EXAMPLE_OPERATOR)
                .with_input(ParameterSpec::new::<i32>("x"))
                .with_input(ParameterSpec::new::<i32>("y"))
        )
    );

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ExampleOp {
        #[serde(default)]
        pub z: bool,
    }

    impl FirehoseOperator for ExampleOp {
        fn apply_to_row(
            &self,
            _row: &mut crate::core::rows::FirehoseRowTransaction,
        ) -> BunsenResult<()> {
            todo!()
        }
    }

    #[should_panic(expected = "Builder operator ID does not match registration ID")]
    #[test]
    fn test_get_builder_mismatched_id() {
        let reg = FirehoseOperatorFactoryRegistration {
            operator_id: EXAMPLE_OPERATOR,
            supplier: || {
                Arc::new(SimpleConfigOperatorFactory::<ExampleOp>::new(
                    FirehoseOperatorSignature::new().with_operator_id("mismatched_id"),
                ))
            },
        };

        // This should panic because the operator ID does not match the
        // registration ID.
        reg.get_builder();
    }

    #[test]
    fn test_list_default_registrations() {
        let reg = FirehoseOperatorFactoryRegistration::list_default_registrations()
            .into_iter()
            .find(|r| r.operator_id == EXAMPLE_OPERATOR)
            .expect("Could not find registration");

        assert_eq!(reg.operator_id, EXAMPLE_OPERATOR);

        assert_eq!(
            format!("{reg:?}"),
            format!(
                "FirehoseOperatorFactoryRegistration {{ operator_id: \"{}\" }}",
                EXAMPLE_OPERATOR
            )
        );
    }
}
