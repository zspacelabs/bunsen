//! # `ResNeSt` pretrained providers

use alloc::sync::Arc;

use crate::data::builds::BuildProvider;

/// The `ResNeSt` kit's name: its rows' `kit` tag.
pub const RESNEST_KIT: &str = "resnest";

/// The key of the checkpoint in a `ResNeSt` resource set.
pub use crate::data::builds::CHECKPOINT;

/// What the resnest kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const RESNEST_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const RESNEST_CATALOG_CONTRACT: &str = "resnest-classifier";

/// `ResNeSt`'s compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a timm hub repo of a `ResNeSt` the kit builds
/// (`hf:timm/resnest50d.in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_resnest_providers() -> Vec<Arc<dyn BuildProvider>> {
    crate::kits::images::support::pretrained::timm_providers(RESNEST_CATALOG_CONTRACT)
}

/// `ResNeSt`'s factory: [`default_resnest_providers`] behind
/// [`ResNeStConstructConfig`](super::ResNeStConstructConfig).
///
/// # Errors
/// As [`BuildFactory::register`](crate::data::builds::BuildFactory::register), if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_resnest_factory()
-> crate::errors::BunsenResult<crate::data::builds::BuildFactory<super::ResNeStConstructConfig>> {
    crate::data::builds::BuildFactory::new().with_providers(default_resnest_providers())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::kits::images::classification::resnest::ResNeStContractConfig;

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint, its `config.json`, and `ImageNet`-1k's labels.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        use crate::kits::images::support::pretrained::testing::{
            LabelRule,
            assert_timm_rows,
        };
        assert_timm_rows::<ResNeStContractConfig>(
            &super::super::default_resnest_factory().unwrap(),
            8,
            LabelRule::Counts(&[("imagenet/synsets-1k", 8)]),
        );
    }

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        crate::kits::images::support::pretrained::testing::assert_timm_defaults_register(
            &default_resnest_factory().unwrap(),
        );
    }
}
