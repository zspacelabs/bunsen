use std::sync::Arc;

use bunsen::errors::{
    BunsenResult,
    ResultContext,
    WithOkOrPanic,
};
use burn::{
    data::dataloader::batcher::Batcher,
    tensor::Device,
};

use crate::core::{
    FirehoseRowBatch,
    operations::executor::FirehoseBatchExecutor,
};

/// Input Adapter for [`FirehoseExecutorBatcher`]: turns dataset items into
/// the base columns of a row batch.
pub trait BatcherInputAdapter<I>: Send + Sync
where
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
{
    /// Converts a vector of inputs of type `I` to a `FirehoseRowBatch`.
    fn apply(
        &self,
        inputs: Vec<I>,
    ) -> BunsenResult<FirehoseRowBatch>;
}

/// Output Adapter for [`FirehoseExecutorBatcher`]: turns the executed row
/// batch into the batch burn trains on.
pub trait BatcherOutputAdapter<O>: Send + Sync
where
    O: Send + Clone + std::fmt::Debug + 'static,
{
    /// Converts a `FirehoseRowBatch` to an output of type `O`.
    fn apply(
        &self,
        batch: &FirehoseRowBatch,
        device: &Device,
    ) -> BunsenResult<O>;
}

/// Firehose Row Burn Batcher.
///
/// A burn [`Batcher`] that turns a `Vec<I>` of dataset items into a row
/// batch with a [`BatcherInputAdapter`], runs a [`FirehoseBatchExecutor`]
/// over it, and makes the output `O` with a [`BatcherOutputAdapter`].
///
/// # Panics
///
/// [`Batcher::batch`] returns no error, so a failure in any of the three
/// steps panics.
pub struct FirehoseExecutorBatcher<I, O>
where
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    /// The executor used to run the batch.
    executor: Arc<dyn FirehoseBatchExecutor>,

    /// Map `Vec<I>` input to a `FirehoseRowBatch`.
    input_adapter: Arc<dyn BatcherInputAdapter<I>>,

    /// Map a `FirehoseRowBatch` to an output of type `O`.
    output_adapter: Arc<dyn BatcherOutputAdapter<O>>,
}

impl<I, O> FirehoseExecutorBatcher<I, O>
where
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    /// Creates a new `FirehoseExecutorBatcher` with the given executor, input
    /// adapter, and output adapter.
    pub fn new(
        executor: Arc<dyn FirehoseBatchExecutor>,
        input_adapter: Arc<dyn BatcherInputAdapter<I>>,
        output_adapter: Arc<dyn BatcherOutputAdapter<O>>,
    ) -> Self {
        Self {
            executor,
            input_adapter,
            output_adapter,
        }
    }

    /// Executes a batch of items and returns the result.
    ///
    /// # Arguments
    ///
    /// * `items` - A vector of items to be processed.
    /// * `device` - The device on which the output will be processed.
    ///
    /// # Returns
    ///
    /// A `BunsenResult` containing the output of type `O`.
    fn batch_result(
        &self,
        items: Vec<I>,
        device: &Device,
    ) -> BunsenResult<O> {
        let mut batch = self.input_adapter.apply(items)?;

        self.executor
            .execute_batch(&mut batch)
            .context("executing batch")?;

        self.output_adapter.apply(&batch, device)
    }
}

impl<I, O> Batcher<I, O> for FirehoseExecutorBatcher<I, O>
where
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    fn batch(
        &self,
        items: Vec<I>,
        device: &Device,
    ) -> O {
        self.batch_result(items, device).ok_or_panic()
    }
}
