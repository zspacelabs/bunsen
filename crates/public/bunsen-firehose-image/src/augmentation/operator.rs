//! The augmentation operator.

use std::{
    fmt::Debug,
    sync::Arc,
};

use bunsen::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
};
use bunsen_firehose::core::{
    FirehoseRowReader,
    FirehoseRowTransaction,
    FirehoseRowWriter,
    operations::{
        factory::{
            FirehoseOperatorFactory,
            FirehoseOperatorInitContext,
        },
        operator::FirehoseOperator,
        planner::OperationPlan,
        signature::{
            FirehoseOperatorSignature,
            ParameterSpec,
        },
    },
};
use rand::SeedableRng;
use serde::{
    Deserialize,
    Serialize,
};

use super::{
    stage::{
        AugmentationStage,
        AugmentationStageConfig,
        ImageAugContext,
    },
    *,
};

/// Image augmentation operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentImageConfig {
    /// The stages to apply.
    pub stages: Vec<AugmentationStageConfig>,
}

impl AugmentImageConfig {
    /// Converts into an `OperationPlanner`
    ///
    /// # Arguments
    ///
    /// * `seed_column`: The name of the input column containing the
    ///   augmentation seed.
    /// * `source_column`: The name of the input image column.
    /// * `result_column`: The name of the output image column.
    pub fn to_plan(
        &self,
        seed_column: &str,
        source_column: &str,
        result_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(AUGMENT_IMAGE)
            .with_input("seed", seed_column)
            .with_input("source", source_column)
            .with_output("result", result_column)
            .with_config(self.clone())
    }
}

/// Image augmentation operator factory.
#[derive(Debug)]
pub struct AugmentImageOperatorFactory {
    /// The operator signature.
    signature: FirehoseOperatorSignature,
}

impl Default for AugmentImageOperatorFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl AugmentImageOperatorFactory {
    /// Construct a new factory.
    pub fn new() -> Self {
        Self {
            signature: FirehoseOperatorSignature::from_operator_id(AUGMENT_IMAGE)
                .with_description("Loads an image from disk.")
                .with_input(
                    ParameterSpec::new::<u64>("seed").with_description("Augmentation seed."),
                )
                .with_input(
                    ParameterSpec::new::<DynamicImage>("source").with_description("Source image."),
                )
                .with_output(
                    ParameterSpec::new::<DynamicImage>("result").with_description("Result image."),
                ),
        }
    }
}

impl FirehoseOperatorFactory for AugmentImageOperatorFactory {
    fn signature(&self) -> &FirehoseOperatorSignature {
        &self.signature
    }

    fn init(
        &self,
        context: &dyn FirehoseOperatorInitContext,
    ) -> BunsenResult<Box<dyn FirehoseOperator>> {
        let config = &context.build_plan().config;
        let cfg: AugmentImageConfig = serde_json::from_value(config.clone()).map_err(|e| {
            BunsenError::from_cause(BunsenErrorKind::Illegal, e).context_details(
                format!(
                    "deserializing the operator config for {}",
                    self.signature.operator_id.as_deref().unwrap_or("unknown"),
                ),
                format!("{config:#}"),
            )
        })?;

        let builder = GlobalRegistryBuilder {};
        let stages = builder.build_stage_vector(&cfg.stages)?;

        Ok(Box::new(AugmentImageOperation { stages }))
    }
}

/// Image augmentation operator.
#[derive(Debug, Clone)]
pub struct AugmentImageOperation {
    /// The stages to apply.
    stages: Vec<Arc<dyn AugmentationStage>>,
}

impl AugmentImageOperation {
    /// Construct a new operation.
    pub fn new(stages: Vec<Arc<dyn AugmentationStage>>) -> Self {
        Self { stages }
    }

    /// Convert into a config.
    pub fn to_config(&self) -> AugmentImageConfig {
        AugmentImageConfig {
            stages: self.stages.iter().map(|s| s.as_config()).collect(),
        }
    }

    /// Converts into an `OperationPlanner`
    ///
    /// # Arguments
    ///
    /// * `seed_column`: The name of the input column containing the
    ///   augmentation seed.
    /// * `source_column`: The name of the input image column.
    /// * `result_column`: The name of the output image column.
    pub fn to_plan(
        &self,
        seed_column: &str,
        source_column: &str,
        result_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(AUGMENT_IMAGE)
            .with_input("seed", seed_column)
            .with_input("source", source_column)
            .with_output("result", result_column)
            .with_config(self.to_config())
    }
}

impl FirehoseOperator for AugmentImageOperation {
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> BunsenResult<()> {
        let mut image = txn.expect_get_ref::<DynamicImage>("source").clone();
        let mut ctx = ImageAugContext::new(rand::rngs::StdRng::seed_from_u64(
            txn.expect_get_parsed("seed"),
        ));

        for stage in &self.stages {
            image = stage.augment_image(image, &mut ctx)?;
        }

        txn.expect_set_from_box("result", Box::new(image));

        Ok(())
    }
}
