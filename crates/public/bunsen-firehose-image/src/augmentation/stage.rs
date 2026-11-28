//! A stage and its context.

use std::fmt::Debug;

use bunsen::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
};
use serde::{
    Deserialize,
    Serialize,
    de::DeserializeOwned,
};

use super::*;

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
