//! # `RegNet` pretrained providers

use alloc::sync::Arc;

use crate::data::builds::BuildProvider;

/// The `RegNet` kit's name: its rows' `kit` tag.
pub const REGNET_KIT: &str = "regnet";

/// The key of the checkpoint in a `RegNet` resource set.
pub use crate::data::builds::CHECKPOINT;

/// What the regnet kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const REGNET_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const REGNET_CATALOG_CONTRACT: &str = "regnet-classifier";

/// `RegNet`'s compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a timm hub repo of a `RegNet` the kit builds
/// (`hf:timm/regnety_032.ra_in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_regnet_providers() -> Vec<Arc<dyn BuildProvider>> {
    crate::kits::images::support::pretrained::timm_providers(REGNET_CATALOG_CONTRACT)
}

/// `RegNet`'s factory: [`default_regnet_providers`] behind
/// [`RegNetConstructConfig`](super::RegNetConstructConfig).
///
/// # Errors
/// As [`BuildFactory::register`](crate::data::builds::BuildFactory::register), if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_regnet_factory()
-> crate::errors::BunsenResult<crate::data::builds::BuildFactory<super::RegNetConstructConfig>> {
    crate::data::builds::BuildFactory::new().with_providers(default_regnet_providers())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::kits::images::classification::regnet::RegNetContractConfig;

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint and its `config.json`. A row with labels names them too;
    /// the headless `SEER` trunks have none.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        use crate::kits::images::support::pretrained::testing::{
            LabelRule,
            assert_timm_rows,
        };
        let rows = assert_timm_rows::<RegNetContractConfig>(
            &super::super::default_regnet_factory().unwrap(),
            65,
            LabelRule::Counts(&[
                ("", 3),
                ("imagenet/synsets-12k", 2),
                ("imagenet/synsets-1k", 60),
            ]),
        );
        for (name, row) in rows {
            let unlabelled = row
                .resource_set()
                .get(crate::kits::images::support::card::LABELS)
                .is_none();
            assert_eq!(unlabelled, name.ends_with(".seer"), "{name}");
        }
    }

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        crate::kits::images::support::pretrained::testing::assert_timm_defaults_register(
            &default_regnet_factory().unwrap(),
        );
    }
}
