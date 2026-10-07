//! Stage that does nothing.
use std::sync::Arc;

use bunsen::errors::BunsenResult;
use image::DynamicImage;
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::{
    augmentation::{
        AugmentationStage,
        AugmentationStageConfig,
        ImageAugContext,
        PluginBuilder,
        WithAugmentationStageBuilder,
    },
    define_image_aug_plugin,
};

define_image_aug_plugin!(NOOP_STAGE, NoOpStage::build_stage);

/// A no-operation plugin for image augmentation that does nothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoOpStage;

impl WithAugmentationStageBuilder for NoOpStage {
    fn build_stage(
        config: &AugmentationStageConfig,
        _builder: &dyn PluginBuilder,
    ) -> BunsenResult<Arc<dyn AugmentationStage>> {
        Ok(Arc::new(config.parse_body::<NoOpStage>()?))
    }
}

impl AugmentationStage for NoOpStage {
    fn name(&self) -> &str {
        NOOP_STAGE
    }

    fn as_config_body(&self) -> Value {
        Value::Null
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        _ctx: &mut ImageAugContext,
    ) -> BunsenResult<DynamicImage> {
        Ok(image)
    }
}
