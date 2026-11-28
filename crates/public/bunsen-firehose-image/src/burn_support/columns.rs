//! Stacking a column into a tensor.

use bunsen::errors::BunsenResult;
use bunsen_firehose::core::{
    FirehoseRowBatch,
    FirehoseRowReader,
};
use burn::tensor::TensorData;

/// Stacks the tensor data from a batch of rows into a single `TensorData`.
///
/// # Arguments
///
/// * `batch` - The batch of rows containing the tensor data.
/// * `column_name` - The name of the column containing the tensor data.
///
/// # Returns
///
/// A `BunsenResult<TensorData`.
pub fn stack_tensor_data_column(
    batch: &FirehoseRowBatch,
    column_name: &str,
) -> BunsenResult<TensorData> {
    assert!(!batch.is_empty());

    let item_shape = batch[0]
        .expect_get_ref::<TensorData>(column_name)
        .shape()
        .clone();
    let stack_shape = [batch.len(), item_shape[0], item_shape[1], item_shape[2]];

    let data_vec = batch
        .iter()
        .map(|row| {
            row.expect_get_ref::<TensorData>(column_name)
                .as_slice::<f32>()
                .map_err(|_| "Failed to get slice from tensor data")
                .unwrap()
        })
        .collect::<Vec<_>>();

    let total_len = data_vec.iter().map(|&d| d.len()).sum::<usize>();
    let mut stack_data = Vec::with_capacity(total_len);
    data_vec.iter().for_each(|d| {
        stack_data.extend_from_slice(d);
    });

    Ok(TensorData::new(stack_data, stack_shape))
}
