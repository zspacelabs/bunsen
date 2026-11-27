//! # `DeiT` pretrained providers

use alloc::sync::Arc;

use crate::data::builds::BuildProvider;

/// The `DeiT` kit's name: its rows' `kit` tag.
pub const DEIT_KIT: &str = "deit";

/// The key of the checkpoint in a `DeiT` resource set.
pub use crate::data::builds::CHECKPOINT;

/// What the deit kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const DEIT_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const DEIT_CATALOG_CONTRACT: &str = "deit-classifier";

/// `DeiT`'s compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a timm hub repo of a `DeiT` the kit builds
/// (`hf:timm/deit_base_distilled_patch16_224.fb_in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_deit_providers() -> Vec<Arc<dyn BuildProvider>> {
    crate::kits::images::support::pretrained::timm_providers(DEIT_CATALOG_CONTRACT)
}

/// `DeiT`'s factory: [`default_deit_providers`] behind [`DeiTConstructConfig`](super::DeiTConstructConfig).
///
/// # Errors
/// As [`BuildFactory::register`](crate::data::builds::BuildFactory::register), if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_deit_factory()
-> crate::errors::BunsenResult<crate::data::builds::BuildFactory<super::DeiTConstructConfig>> {
    crate::data::builds::BuildFactory::new().with_providers(default_deit_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::deit::DeiTContractConfig;

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint and its `config.json`, and `ImageNet`-1k's labels.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        use crate::kits::images::support::pretrained::testing::{
            LabelRule,
            assert_timm_rows,
        };
        assert_timm_rows::<DeiTContractConfig>(
            &super::super::default_deit_factory().unwrap(),
            24,
            LabelRule::Counts(&[("imagenet/synsets-1k", 24)]),
        );
    }

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        crate::kits::images::support::pretrained::testing::assert_timm_defaults_register(
            &default_deit_factory().unwrap(),
        );
    }
}
