use std::{
    fmt::Debug,
    sync::Arc,
};

use bunsen::errors::BunsenResult;
use serde::{
    Deserialize,
    Serialize,
};

use crate::core::{
    operations::{
        environment::{
            BuildPlanContext,
            FirehoseOperatorEnvironment,
        },
        signature::FirehoseOperatorSignature,
    },
    rows::{
        FirehoseBatchTransaction,
        FirehoseRowBatch,
        FirehoseRowTransaction,
    },
    schema::{
        BuildPlan,
        FirehoseTableSchema,
    },
};

/// Scheduling metadata for an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorSchedulingMetadata {
    /// The largest effective batch size for the operator.
    pub effective_batch_size: usize,
}

/// An instantiated column => column operator.
///
/// Built by a
/// [`FirehoseOperatorFactory`](super::factory::FirehoseOperatorFactory) for one
/// build plan, and run by an [`OperationRunner`]. It reads its inputs and
/// writes its outputs by parameter name, through a transaction.
pub trait FirehoseOperator: 'static + Send + Sync + Debug {
    /// Gets the scheduling metadata for the operator.
    fn scheduling_metadata(&self) -> OperatorSchedulingMetadata {
        OperatorSchedulingMetadata {
            // By default, we process one row at a time.
            effective_batch_size: 1,
        }
    }

    /// Apply the operator to a batch of rows in the provided context.
    ///
    /// # Returns
    ///
    /// A `BunsenResult<()>` indicating success or containing an error if the
    /// operation fails.
    #[must_use]
    fn apply_to_batch(
        &self,
        txn: &mut FirehoseBatchTransaction,
    ) -> BunsenResult<()> {
        for index in 0..txn.len() {
            let mut row_txn = txn.mut_row_transaction(index);
            self.apply_to_row(&mut row_txn)?;
        }
        Ok(())
    }

    /// Apply the operator to a single row in the provided context.
    ///
    /// # Returns
    ///
    /// A `BunsenResult<()>` indicating success or containing an error if the
    /// operation fails.
    #[must_use]
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> BunsenResult<()>;
}

/// Represents a schema + instantiated column operator for a particular build
/// plan.
pub struct OperationRunner {
    /// The table schema that this operator is bound to.
    pub table_schema: Arc<FirehoseTableSchema>,

    /// A reference to the build plan that this operator is part of.
    pub build_plan: Arc<BuildPlan>,

    /// Signature of the operator.
    pub signature: Arc<FirehoseOperatorSignature>,

    /// The operator that this builder wraps.
    operator: Box<dyn FirehoseOperator>,
}

impl Debug for OperationRunner {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if !f.alternate() {
            f.debug_struct("ColumnBuilder")
                .field("id", &self.build_plan.operator_id)
                .field("inputs", &self.build_plan.inputs)
                .field("outputs", &self.build_plan.outputs)
                .finish()
        } else {
            f.debug_struct("ColumnBuilder")
                .field("build_plan", &self.build_plan)
                .finish()
        }
    }
}

impl OperationRunner {
    /// Create a new `OperationRunner` by binding a `BuildPlan` to a
    /// `FirehoseTableSchema`.
    ///
    /// # Arguments
    ///
    /// * `table_schema` - The schema of the table to which this plan is bound.
    /// * `build_plan` - The build plan that describes the operator and its
    ///   inputs/outputs.
    /// * `env` - An environment that can create the operator based on the build
    ///   plan.
    ///
    /// # Returns
    ///
    /// A result containing an `OperationRunner` if successful, or an error
    /// if the binding fails.
    #[must_use]
    pub fn new_for_plan(
        table_schema: Arc<FirehoseTableSchema>,
        build_plan: Arc<BuildPlan>,
        env: &dyn FirehoseOperatorEnvironment,
    ) -> BunsenResult<OperationRunner> {
        let table_schema = table_schema.clone();
        let build_plan = build_plan.clone();

        let build_plan_context = BuildPlanContext::new(table_schema.clone(), build_plan.clone());

        let signature = Arc::new(
            env.lookup_operator_factory(build_plan_context.operator_id())?
                .signature()
                .clone(),
        );

        let operator = env.init_operator(build_plan_context)?;

        Ok(OperationRunner {
            table_schema,
            build_plan,
            operator,
            signature,
        })
    }

    /// Gets the scheduling metadata for the operator.
    pub fn scheduling_metadata(&self) -> OperatorSchedulingMetadata {
        self.operator.scheduling_metadata()
    }

    /// Apply the operator to a batch of rows.
    ///
    /// # Arguments
    ///
    /// * `batch` - The rows to process. The operator's outputs are written back
    ///   only if it succeeds.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error message if the operation fails.
    pub fn apply_to_batch(
        &self,
        batch: &mut FirehoseRowBatch,
    ) -> BunsenResult<()> {
        // TODO: sub-batch based upon scheduling metadata.effective_batch_size.
        // Requires batch.slice(); batch.assign_slice_from();

        let mut txn =
            FirehoseBatchTransaction::new(batch, self.build_plan.clone(), self.signature.clone());

        self.operator.apply_to_batch(&mut txn)?;

        txn.commit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPERATION_RUNNER_IS_SEND: fn() = || {
        fn assert_send<T: Send>() {}
        assert_send::<OperationRunner>();
    };
    #[test]
    fn test_operation_runner_is_send() {
        OPERATION_RUNNER_IS_SEND();
    }
}
