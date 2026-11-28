//! The stage plugins and their registry.

use std::sync::Arc;

use bunsen::errors::{
    BunsenError,
    BunsenResult,
    LookupError,
};

use super::stage::{
    AugmentationStage,
    AugmentationStageConfig,
};

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

inventory::collect!(AugmentationStageGlobalRegistration);
