//! # Image/Tensor conversion utilities
//!
//! [`ImageToTensorData`] is the `IMAGE_TO_TENSOR_DATA` operator: it reads a
//! [`DynamicImage`] column and writes a `[height, width, channels]` `f32`
//! [`TensorData`] column. [`stack_tensor_data_column`] then stacks a batch of
//! those per-row tensors into a single `[batch, height, width, channels]`
//! `TensorData` — the bridge used by a [`burn`] batcher.
//!
//! ```
//! use std::sync::Arc;
//!
//! use bunsen::errors::BunsenResult;
//! use bunsen_firehose::{
//!     core::{
//!         FirehoseRowBatch,
//!         FirehoseRowReader,
//!         FirehoseRowWriter,
//!         FirehoseTableSchema,
//!         operations::executor::{
//!             FirehoseBatchExecutor,
//!             SequentialBatchExecutor,
//!         },
//!         schema::ColumnSchema,
//!     },
//!     ops::init_default_operator_environment,
//! };
//! use bunsen_firehose_image::burn_support::ImageToTensorData;
//! use burn::prelude::TensorData;
//! use image::{
//!     DynamicImage,
//!     RgbImage,
//! };
//!
//! fn main() -> BunsenResult<()> {
//!     let env = Arc::new(init_default_operator_environment());
//!
//!     let mut schema =
//!         FirehoseTableSchema::from_columns(&[ColumnSchema::new::<
//!             DynamicImage,
//!         >("image")]);
//!     ImageToTensorData::default()
//!         .to_plan("image", "data")
//!         .apply_to_schema(&mut schema, env.as_ref())?;
//!     let schema = Arc::new(schema);
//!
//!     let executor =
//!         SequentialBatchExecutor::new(schema.clone(), env.clone())?;
//!     let mut batch = FirehoseRowBatch::new_with_size(schema.clone(), 1);
//!     // `DynamicImage` columns are stored as boxed values, not serialized.
//!     batch[0].expect_set_boxing(
//!         "image",
//!         DynamicImage::from(RgbImage::new(8, 4)),
//!     );
//!     executor.execute_batch(&mut batch)?;
//!
//!     let data =
//!         batch[0].maybe_get("data").unwrap().as_ref::<TensorData>()?;
//!     // [height, width, channels]
//!     assert_eq!(
//!         [data.shape()[0], data.shape()[1], data.shape()[2]],
//!         [4, 8, 3]
//!     );
//!     Ok(())
//! }
//! ```
use bunsen::errors::BunsenResult;
use bunsen_firehose::{
    core::{
        FirehoseRowBatch,
        FirehoseRowReader,
        operations::{
            factory::SimpleConfigOperatorFactory,
            signature::{
                FirehoseOperatorSignature,
                ParameterSpec,
            },
        },
    },
    define_firehose_operator,
};
use burn::tensor::TensorData;
use image::DynamicImage;

pub mod image_to_tensor_data;

pub mod pixeldepth_support;

pub use image_to_tensor_data::*;
use pixeldepth_support::image_to_pixeldepth_vec;

define_firehose_operator!(
    IMAGE_TO_TENSOR_DATA,
    SimpleConfigOperatorFactory::<ImageToTensorData>::new(
        FirehoseOperatorSignature::new()
            .with_operator_id(IMAGE_TO_TENSOR_DATA)
            .with_description("Converts an image to TensorData.")
            .with_input(
                ParameterSpec::new::<DynamicImage>("image")
                    .with_description("Image to convert to a tensor."),
            )
            .with_output(
                ParameterSpec::new::<TensorData>("data")
                    .with_description("TensorData representation of the image."),
            ),
    )
);

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

/// An image as an `f32` `[height, width, channels]` tensor: bunsen's
/// `support::images::image_to_f32_tensor`
/// (8-bit values over 255, 16-bit over 65535, float values as they are).
pub use bunsen::support::images::image_to_f32_tensor;
