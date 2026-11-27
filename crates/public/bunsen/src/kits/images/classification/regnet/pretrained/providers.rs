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
    use alloc::collections::BTreeMap;

    use super::*;
    use crate::{
        data::{
            builds::CONFIG,
            pretrained::Prefabs,
        },
        kits::images::{
            classification::regnet::{
                RegNetContractConfig,
                pretrained::REGNET_KIT,
            },
            support::card::{
                LABEL_NAMES,
                LABELS,
            },
        },
    };

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint and its `config.json`. A row with labels names them too;
    /// the headless `SEER` trunks have none.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        let factory = super::super::default_regnet_factory().unwrap();
        let rows = factory.list();
        assert_eq!(rows.len(), 65);
        let architectures = RegNetContractConfig::prefab_names();
        let mut by_labels: BTreeMap<String, usize> = BTreeMap::new();
        for (_, row) in &rows {
            let name = row.name.to_string();
            let map = &row.resource_set();
            map.validate().unwrap();
            let arch = name
                .strip_prefix("timm/")
                .and_then(|n| n.split_once('.'))
                .map(|(arch, _)| arch)
                .unwrap_or_else(|| panic!("{name}: a timm repo's name"));
            assert!(architectures.contains(&arch), "{name}");
            assert!(map.get(CHECKPOINT).unwrap().sha256.is_some(), "{name}");
            assert!(map.get(CONFIG).is_some(), "{name}");
            let labels = map.get(LABELS).map(|l| l.name.to_string());
            assert_eq!(
                labels.is_some(),
                map.get(LABEL_NAMES).is_some(),
                "{name}: labels and their names together"
            );
            assert_eq!(labels.is_none(), name.ends_with(".seer"), "{name}");
            *by_labels.entry(labels.unwrap_or_default()).or_default() += 1;
        }
        assert_eq!(
            by_labels,
            BTreeMap::from([
                (String::new(), 3),
                ("imagenet/synsets-12k".into(), 2),
                ("imagenet/synsets-1k".into(), 60),
            ])
        );
    }

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_regnet_factory().unwrap();
        assert_eq!(factory.kit(), REGNET_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
