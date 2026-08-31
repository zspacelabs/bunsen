//! # bunsen-firehose-image - Image operators for the firehose pipeline
//!
//! This crate provides ready-to-use [`bunsen_firehose`] operators for image
//! data: loading from disk, resizing/recoloring, seedable augmentation, and
//! conversion to [`burn`] tensors. Each operator registers itself into the
//! global registry, so [`init_default_operator_environment`] returns an
//! environment that already knows about all of them.
//!
//! The modules:
//!
//! - [`loader`] — [`ImageLoader`](crate::loader::ImageLoader) (`LOAD_IMAGE`):
//!   open an image from a path column, optionally resizing and recoloring it
//!   into a [`DynamicImage`](image::DynamicImage) column.
//! - [`augmentation`] — the
//!   [`AugmentationStage`](crate::augmentation::AugmentationStage) plugin
//!   system and the [`AUGMENT_IMAGE`](crate::augmentation::AUGMENT_IMAGE)
//!   operator, which applies a seeded sequence of stages (flips, blur, speckle,
//!   probabilistic control flow, ...).
//! - [`burn_support`] —
//!   [`ImageToTensorData`](crate::burn_support::ImageToTensorData)
//!   (`IMAGE_TO_TENSOR_DATA`) plus helpers like
//!   [`image_to_f32_tensor`](crate::burn_support::image_to_f32_tensor) and
//!   [`stack_tensor_data_column`](crate::burn_support::stack_tensor_data_column)
//!   for assembling tensor batches.
//! - [`colortype_support`] — [`ColorType`] conversion and (de)serialization
//!   helpers.
//! - [`test_util`] — image generation/comparison helpers for tests.
//!
//! # Example: load, resize, and tensorize an image
//!
//! Build a schema that loads a `path` column into a resized `image` column and
//! then converts it to a `data` ([`TensorData`](burn::prelude::TensorData))
//! column, then run a one-row batch through a
//! [`SequentialBatchExecutor`](bunsen_firehose::core::operations::executor::SequentialBatchExecutor).
//!
//! ```
//! use std::sync::Arc;
//!
//! use bunsen_firehose::{
//!     core::{
//!         FirehoseRowBatch,
//!         FirehoseRowReader,
//!         FirehoseRowWriter,
//!         FirehoseTableSchema,
//!         FirehoseValue,
//!         operations::executor::{
//!             FirehoseBatchExecutor,
//!             SequentialBatchExecutor,
//!         },
//!         schema::ColumnSchema,
//!     },
//!     ops::init_default_operator_environment,
//! };
//! use bunsen_firehose_image::{
//!     ImageShape,
//!     burn_support::ImageToTensorData,
//!     loader::{
//!         ImageLoader,
//!         ResizeSpec,
//!     },
//! };
//! use burn::prelude::TensorData;
//! use image::{
//!     ColorType,
//!     DynamicImage,
//!     RgbImage,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     // Every image operator registers itself globally; this collects them
//!     // (LOAD_IMAGE, IMAGE_TO_TENSOR_DATA, AUGMENT_IMAGE, ...) into one env.
//!     let env = Arc::new(init_default_operator_environment());
//!
//!     // path --LOAD_IMAGE--> image --IMAGE_TO_TENSOR_DATA--> data
//!     let mut schema =
//!         FirehoseTableSchema::from_columns(&[ColumnSchema::new::<String>(
//!             "path",
//!         )
//!         .with_description("path to the image")]);
//!     ImageLoader::default()
//!         .with_resize(ResizeSpec::new(ImageShape {
//!             width: 16,
//!             height: 16,
//!         }))
//!         .with_recolor(ColorType::Rgb8)
//!         .to_plan("path", "image")
//!         .apply_to_schema(&mut schema, env.as_ref())?;
//!     ImageToTensorData::default()
//!         .to_plan("image", "data")
//!         .apply_to_schema(&mut schema, env.as_ref())?;
//!     let schema = Arc::new(schema);
//!
//!     // Write a throwaway PNG to feed the pipeline.
//!     let dir = tempfile::tempdir()?;
//!     let path = dir.path().join("img.png");
//!     DynamicImage::from(RgbImage::new(32, 32)).save(&path)?;
//!
//!     let executor =
//!         SequentialBatchExecutor::new(schema.clone(), env.clone())?;
//!     let mut batch = FirehoseRowBatch::new_with_size(schema.clone(), 1);
//!     batch[0].expect_set(
//!         "path",
//!         FirehoseValue::serialized(path.to_string_lossy().to_string())?,
//!     );
//!     executor.execute_batch(&mut batch)?;
//!
//!     // Both derived columns are now populated.
//!     let image = batch[0]
//!         .maybe_get("image")
//!         .unwrap()
//!         .as_ref::<DynamicImage>()?;
//!     assert_eq!((image.width(), image.height()), (16, 16));
//!
//!     let data =
//!         batch[0].maybe_get("data").unwrap().as_ref::<TensorData>()?;
//!     // [height, width, channels]
//!     assert_eq!([data.shape[0], data.shape[1], data.shape[2]], [16, 16, 3]);
//!     Ok(())
//! }
//! ```
//!
//! For a complete training pipeline that wires these operators into a [`burn`]
//! `DataLoaderBuilder` — including per-row augmentation seeds — see the
//! `resnet_tiny` example under `demos/bimm/examples`.
//!
//! [`init_default_operator_environment`]: bunsen_firehose::ops::init_default_operator_environment

use serde::{
    Deserialize,
    Serialize,
};

pub mod augmentation;
pub mod burn_support;
pub mod colortype_support;
pub mod loader;
pub mod test_util;

/// Represents the shape of an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageShape {
    /// The width of the image in pixels.
    pub width: u32,

    /// The height of the image in pixels.
    pub height: u32,
}

pub use image::ColorType;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bunsen::support::testing::PerformanceBackend;
    use bunsen_firehose::{
        core::{
            FirehoseRowBatch,
            FirehoseTableSchema,
            FirehoseValue,
            operations::executor::{
                FirehoseBatchExecutor,
                SequentialBatchExecutor,
            },
            rows::{
                FirehoseRowReader,
                FirehoseRowWriter,
            },
            schema::ColumnSchema,
        },
        ops::init_default_operator_environment,
    };
    use burn::prelude::TensorData;
    use image::{
        ColorType,
        DynamicImage,
        imageops::FilterType,
    };
    use indoc::indoc;

    use crate::{
        ImageShape,
        burn_support::{
            ImageToTensorData,
            image_to_f32_tensor,
        },
        loader::{
            ImageLoader,
            ResizeSpec,
        },
        test_util,
        test_util::assert_image_close,
    };

    #[test]
    fn test_example() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir().unwrap();

        type B = PerformanceBackend;

        let device = Default::default();

        let env = Arc::new(init_default_operator_environment());

        let schema = {
            let mut schema =
                FirehoseTableSchema::from_columns(&[
                    ColumnSchema::new::<String>("path").with_description("path to the image")
                ]);

            ImageLoader::default()
                .with_resize(
                    ResizeSpec::new(ImageShape {
                        width: 16,
                        height: 24,
                    })
                    .with_filter(FilterType::Nearest),
                )
                .with_recolor(ColorType::L16)
                .to_plan("path", "image")
                .apply_to_schema(&mut schema, env.as_ref())?;

            ImageToTensorData::default()
                .to_plan("image", "data")
                .apply_to_schema(&mut schema, env.as_ref())?;

            Arc::new(schema)
        };

        let executor = SequentialBatchExecutor::new(schema.clone(), env.clone())?;

        assert_eq!(
            serde_json::to_string_pretty(schema.as_ref()).unwrap(),
            indoc! {r#"
                {
                  "columns": [
                    {
                      "name": "path",
                      "description": "path to the image",
                      "data_type": {
                        "type_name": "alloc::string::String"
                      }
                    },
                    {
                      "name": "image",
                      "description": "Image loaded from disk.",
                      "data_type": {
                        "type_name": "image::images::dynimage::DynamicImage"
                      }
                    },
                    {
                      "name": "data",
                      "description": "TensorData representation of the image.",
                      "data_type": {
                        "type_name": "burn_backend::data::tensor::TensorData"
                      }
                    }
                  ],
                  "build_plans": [
                    {
                      "operator_id": "fh:op://bunsen_firehose_image::loader::LOAD_IMAGE",
                      "description": "Loads an image from disk.",
                      "config": {
                        "recolor": "L16",
                        "resize": {
                          "filter": "Nearest",
                          "shape": {
                            "height": 24,
                            "width": 16
                          }
                        }
                      },
                      "inputs": {
                        "path": "path"
                      },
                      "outputs": {
                        "image": "image"
                      }
                    },
                    {
                      "operator_id": "fh:op://bunsen_firehose_image::burn_support::IMAGE_TO_TENSOR_DATA",
                      "description": "Converts an image to TensorData.",
                      "config": {},
                      "inputs": {
                        "image": "image"
                      },
                      "outputs": {
                        "data": "data"
                      }
                    }
                  ]
                }"#,
            }
        );

        let mut batch = FirehoseRowBatch::new_with_size(schema.clone(), 1);

        let source_image: DynamicImage = test_util::generate_gradient_pattern(ImageShape {
            width: 32,
            height: 32,
        })
        .into();

        {
            let image_path = temp_dir
                .path()
                .join("test.png")
                .to_string_lossy()
                .to_string();

            source_image
                .save(&image_path)
                .expect("Failed to save test image");

            batch[0].expect_set("path", FirehoseValue::serialized(image_path)?);
        }

        executor.execute_batch(&mut batch)?;

        let row = &batch[0];

        let row_image = row.maybe_get("image").unwrap().as_ref::<DynamicImage>()?;
        assert_image_close(
            row_image,
            &source_image
                .resize_exact(16, 24, FilterType::Nearest)
                .to_luma8()
                .into(),
            None,
        );

        let row_data = row.maybe_get("data").unwrap().as_ref::<TensorData>()?;
        row_data.assert_eq(
            &image_to_f32_tensor::<B>(row_image, &device).to_data(),
            true,
        );

        Ok(())
    }
}
