//! # Image augmentation operators
//!
//! Augmentation is expressed as a sequence of [`AugmentationStage`]s — flips
//! ([`orientation`]), [`noise`], and control-flow combinators ([`control`],
//! e.g. apply-with-probability or choose-one). Stages register themselves with
//! [`define_image_aug_plugin!`](crate::define_image_aug_plugin) so they can be
//! reconstructed from a serialized [`AugmentationStageConfig`].
//!
//! At pipeline scale, an [`AugmentImageOperation`] (the `AUGMENT_IMAGE`
//! operator) runs a stage list against a `source` image column using a per-row
//! `seed` column, so augmentation is reproducible. A single stage can also be
//! applied directly:
//!
//! ```
//! use bunsen::errors::BunsenResult;
//! use bunsen_firehose_image::augmentation::{
//!     AugmentationStage,
//!     ImageAugContext,
//!     orientation::flip::HorizontalFlipStage,
//! };
//! use image::{
//!     DynamicImage,
//!     RgbImage,
//! };
//! use rand::{
//!     SeedableRng,
//!     rngs::StdRng,
//! };
//!
//! fn main() -> BunsenResult<()> {
//!     let stage = HorizontalFlipStage::new();
//!
//!     // The context carries a seeded RNG; stages draw from it for randomness.
//!     let mut ctx = ImageAugContext::new(StdRng::seed_from_u64(0));
//!
//!     let source: DynamicImage = RgbImage::new(8, 4).into();
//!     let flipped = stage.augment_image(source, &mut ctx)?;
//!
//!     // A horizontal flip preserves the dimensions.
//!     assert_eq!((flipped.width(), flipped.height()), (8, 4));
//!     Ok(())
//! }
//! ```
//!
//! See the crate root for an end-to-end `LOAD_IMAGE` → augment → tensor schema.
use std::{
    fmt::Debug,
    sync::Arc,
};

use bunsen::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    LookupError,
};
use bunsen_firehose::{
    core::{
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
    },
    define_firehose_operator,
};
pub use image::{
    ColorType,
    DynamicImage,
    imageops::FilterType,
};
use rand::SeedableRng;
use serde::{
    Deserialize,
    Serialize,
    de::DeserializeOwned,
};

pub mod control;

pub mod noise;
pub mod orientation;

/// Defines an ID and registers a [`AugmentationStage`].
#[macro_export]
macro_rules! define_image_aug_plugin {
    ($name:ident, $builder:expr) => {
        $crate::define_image_aug_plugin_id!($name);
        $crate::register_image_aug_plugin!($name, $builder);
    };
}

/// Registers a [`AugmentationStage`].
///
/// # Arguments
///
/// - `name`: the local reference name of the plugin ID; used by
///   `define_image_aug_plugin!()`.
/// - `builder`: the plugin builder operation.
#[macro_export]
macro_rules! register_image_aug_plugin {
    ($name:ident, $builder:expr) => {
        inventory::submit! {
            $crate::augmentation::AugmentationStageGlobalRegistration {
                name: $name,
                build_stage: |cfg, builder| ($builder)(cfg, builder),
            }
        }
    };
}

/// Defines a self-referential plugin ID for [`AugmentationStage`].
#[macro_export]
macro_rules! define_image_aug_plugin_id {
    ($name:ident) => {
        bunsen_firehose::define_self_referential_id!("fh:iaug", $name);
    };
}

inventory::collect!(AugmentationStageGlobalRegistration);

/// Registration record for [`AugmentationStage`] builders.
pub struct AugmentationStageGlobalRegistration {
    /// The plugin ID.
    pub name: &'static str,

    /// The plugin builder.
    pub build_stage: fn(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> BunsenResult<Arc<dyn AugmentationStage>>,
}

/// Builder for the `PluginConfig` to `ImageAugPlugin` path.
pub trait PluginBuilder {
    /// Build an `AugmentationStage` from config.
    fn build_stage(
        &self,
        config: &AugmentationStageConfig,
    ) -> BunsenResult<Arc<dyn AugmentationStage>>;

    /// Build a vector of
    fn build_stage_vector(
        &self,
        configs: &Vec<AugmentationStageConfig>,
    ) -> BunsenResult<Vec<Arc<dyn AugmentationStage>>> {
        let mut plugins = Vec::with_capacity(configs.len());
        for config in configs {
            plugins.push(self.build_stage(config)?);
        }
        Ok(plugins)
    }
}

/// Trait defining an associated builder for a plugin.
pub trait WithAugmentationStageBuilder {
    /// Build a plugin.
    fn build_stage(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> BunsenResult<Arc<dyn AugmentationStage>>;
}

/// Global plugin registry builder.
pub struct GlobalRegistryBuilder;

impl PluginBuilder for GlobalRegistryBuilder {
    fn build_stage(
        &self,
        config: &AugmentationStageConfig,
    ) -> BunsenResult<Arc<dyn AugmentationStage>> {
        let name = config.name.as_str();

        let reg = inventory::iter::<AugmentationStageGlobalRegistration>
            .into_iter()
            .find(|reg| reg.name == name)
            .ok_or_else(|| {
                BunsenError::lookup(
                    LookupError::missing("augmentation stage", name).with_candidates(
                        inventory::iter::<AugmentationStageGlobalRegistration>
                            .into_iter()
                            .map(|reg| reg.name),
                    ),
                )
            })?;

        (reg.build_stage)(config, self)
    }
}

/// Augmentation context, used by `AugmentationStage::augment_image`.
#[derive(Debug)]
pub struct ImageAugContext {
    /// The context random number generator.
    rng: rand::rngs::StdRng,
}

impl ImageAugContext {
    /// Construct a new context.
    pub fn new(rng: rand::rngs::StdRng) -> Self {
        Self { rng }
    }

    /// Get a mutable reference to the context's random number generator.
    pub fn rng_mut(&mut self) -> &mut rand::rngs::StdRng {
        &mut self.rng
    }
}

/// A trait defining a plugin for image augmentation.
pub trait AugmentationStage: Debug + Send + Sync {
    /// Get the stage name.
    fn name(&self) -> &str;

    /// Construct the body of a config for this stage.
    fn as_config_body(&self) -> serde_json::Value;

    /// Construct a config for this.
    fn as_config(&self) -> AugmentationStageConfig {
        AugmentationStageConfig {
            name: self.name().to_string(),
            body: self.as_config_body(),
        }
    }

    /// Apply the stage to the image.
    ///
    /// # Arguments
    ///
    /// - `image`: the image to augment.
    /// - `ctx`: the `AugmentationContex` being operated in.
    ///
    /// # Returns
    ///
    /// A (modified?) image.
    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> BunsenResult<DynamicImage>;
}

/// Serializable config for name and body for `AugmentationStage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationStageConfig {
    /// The plugin name.
    pub name: String,

    /// The body of the plugin.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub body: serde_json::Value,
}

impl AugmentationStageConfig {
    /// Parses the body as a stage's config `T`.
    ///
    /// # Errors
    ///
    /// [`Illegal`](BunsenErrorKind::Illegal), with the `serde_json::Error` as
    /// the cause, if the body is not a `T`.
    pub fn parse_body<T: DeserializeOwned>(&self) -> BunsenResult<T> {
        serde_json::from_value(self.body.clone()).map_err(|e| {
            BunsenError::from_cause(BunsenErrorKind::Illegal, e)
                .context(format!("parsing the {:?} stage config", self.name))
        })
    }
}

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

define_firehose_operator!(AUGMENT_IMAGE, AugmentImageOperatorFactory::new());

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

#[cfg(test)]
mod tests {
    use bunsen::errors::{
        LookupProblem,
        testing::{
            ErrorMatcher,
            predicate,
        },
    };
    use serde_json::json;

    use super::*;
    use crate::augmentation::control::{
        noop::NOOP_STAGE,
        sequence::STAGE_SEQUENCE,
    };

    #[test]
    fn test_plugin_builder_errors() {
        let builder = GlobalRegistryBuilder {};

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .cause(predicate("a missing \"nope\"", |l: &LookupError| {
                l.key == "nope"
                    && l.problem == LookupProblem::Missing
                    && l.candidates.iter().any(|c| c == NOOP_STAGE)
            }))
            .assert_err(
                &builder
                    .build_stage(&AugmentationStageConfig {
                        name: "nope".to_string(),
                        body: serde_json::Value::Null,
                    })
                    .map(|_| ()),
            );

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .frame_contains(STAGE_SEQUENCE)
            .has_cause::<serde_json::Error>()
            .assert_err(
                &builder
                    .build_stage(&AugmentationStageConfig {
                        name: STAGE_SEQUENCE.to_string(),
                        body: json!({ "stages": 3 }),
                    })
                    .map(|_| ()),
            );
    }

    #[test]
    fn test_plugin_builder() -> BunsenResult<()> {
        let cfg_json = json! {
            {
                "name": STAGE_SEQUENCE,
                "body": {
                  "stages": [
                    {
                      "name": NOOP_STAGE,
                    }
                  ]
                }
            }
        };
        let pretty = serde_json::to_string_pretty(&cfg_json).unwrap();
        println!("{pretty}");

        let cfg = serde_json::from_value::<AugmentationStageConfig>(cfg_json.clone()).unwrap();

        let builder = GlobalRegistryBuilder {};

        let plugin = builder.build_stage(&cfg)?;

        let new_cfg = plugin.as_config();
        let new_pretty = serde_json::to_string_pretty(&new_cfg).unwrap();
        println!("{new_pretty}");

        // assert!(false);

        Ok(())
    }
}
