//! # `ResNetV2` pretrained providers

use alloc::sync::Arc;

use crate::data::builds::BuildProvider;

/// The `ResNetV2` kit's name: its rows' `kit` tag.
pub const RESNETV2_KIT: &str = "resnetv2";

/// The key of the checkpoint in a `ResNetV2` resource set.
pub use crate::data::builds::CHECKPOINT;

/// What the resnetv2 kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const RESNETV2_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const RESNETV2_CATALOG_CONTRACT: &str = "resnetv2-classifier";

/// `ResNetV2`'s compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a timm hub repo of a `ResNetV2` the kit builds
/// (`hf:timm/resnetv2_50x1_bit.goog_in21k_ft_in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_resnetv2_providers() -> Vec<Arc<dyn BuildProvider>> {
    crate::kits::images::support::pretrained::timm_providers(RESNETV2_CATALOG_CONTRACT)
}

/// `ResNetV2`'s factory: [`default_resnetv2_providers`] behind
/// [`ResNetV2ConstructConfig`](super::ResNetV2ConstructConfig).
///
/// # Errors
/// As [`BuildFactory::register`](crate::data::builds::BuildFactory::register), if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_resnetv2_factory()
-> crate::errors::BunsenResult<crate::data::builds::BuildFactory<super::ResNetV2ConstructConfig>> {
    crate::data::builds::BuildFactory::new().with_providers(default_resnetv2_providers())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::kits::images::classification::resnetv2::ResNetV2ContractConfig;

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint, its `config.json`, and its labels: `ImageNet`-1k's, or
    /// Google's 21k for the `BiT` `goog_in21k` heads.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        use crate::kits::images::support::pretrained::testing::{
            LabelRule,
            assert_timm_rows,
        };
        assert_timm_rows::<ResNetV2ContractConfig>(
            &super::super::default_resnetv2_factory().unwrap(),
            24,
            LabelRule::Counts(&[
                ("imagenet/synsets-1k", 18),
                ("imagenet/synsets-21k-goog", 6),
            ]),
        );
    }

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        crate::kits::images::support::pretrained::testing::assert_timm_defaults_register(
            &default_resnetv2_factory().unwrap(),
        );
    }
}
