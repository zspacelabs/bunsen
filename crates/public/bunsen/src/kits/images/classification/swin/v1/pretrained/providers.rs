//! # Swin V1 pretrained providers

use alloc::sync::Arc;

use crate::data::builds::BuildProvider;

/// The Swin V1 kit's name: its rows' `kit` tag.
pub const SWIN_V1_KIT: &str = "swin-v1";

/// The key of the checkpoint in a Swin V1 resource set.
pub use crate::data::builds::CHECKPOINT;

/// What the swin v1 kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const SWIN_V1_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const SWIN_V1_CATALOG_CONTRACT: &str = "swin-v1-classifier";

/// Swin V1's compiled-in providers, in search order: the catalog provider, then
/// `hf:org/repo`, a timm hub repo of a Swin V1 the kit builds
/// (`hf:timm/swin_tiny_patch4_window7_224.ms_in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_swin_v1_providers() -> Vec<Arc<dyn BuildProvider>> {
    crate::kits::images::support::pretrained::timm_providers(SWIN_V1_CATALOG_CONTRACT)
}

/// Swin V1's factory: [`default_swin_v1_providers`] behind
/// [`SwinV1ConstructConfig`](super::SwinV1ConstructConfig).
///
/// # Errors
/// As [`BuildFactory::register`](crate::data::builds::BuildFactory::register), if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_swin_v1_factory()
-> crate::errors::BunsenResult<crate::data::builds::BuildFactory<super::SwinV1ConstructConfig>> {
    crate::data::builds::BuildFactory::new().with_providers(default_swin_v1_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::swin::v1::SwinV1ContractConfig;

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint and its `config.json`. Each binds `ImageNet`-1k's labels
    /// but the `ms_in22k` pretrainings, whose 21841 classes no label set of
    /// the catalog's fits.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        use crate::kits::images::support::pretrained::testing::{
            LabelRule,
            assert_timm_rows,
        };
        let rows = assert_timm_rows::<SwinV1ContractConfig>(
            &super::super::default_swin_v1_factory().unwrap(),
            19,
            LabelRule::Any,
        );
        for (name, row) in rows {
            let labelled = row
                .resource_set()
                .get(crate::kits::images::support::card::LABELS)
                .is_some();
            assert_eq!(labelled, !name.ends_with(".ms_in22k"), "{name}");
        }
    }

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        crate::kits::images::support::pretrained::testing::assert_timm_defaults_register(
            &default_swin_v1_factory().unwrap(),
        );
    }
}
