//! # `ResNeSt` pretrained providers

use alloc::{
    sync::Arc,
    vec,
};

use crate::data::builds::{
    BuildProvider,
    HfProvider,
};

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
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            RESNEST_CATALOG_CONTRACT,
        )),
        Arc::new(HfProvider::new()),
    ]
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
            classification::resnest::ResNeStContractConfig,
            support::card::{
                LABEL_NAMES,
                LABELS,
            },
        },
    };

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint, its `config.json`, and `ImageNet`-1k's labels.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        let factory = super::super::default_resnest_factory().unwrap();
        let rows = factory.list();
        assert_eq!(rows.len(), 8);
        let architectures = ResNeStContractConfig::prefab_names();
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
            assert!(map.get(LABEL_NAMES).is_some(), "{name}");
            let labels = map
                .get(LABELS)
                .unwrap_or_else(|| panic!("{name}: labels"))
                .name
                .to_string();
            *by_labels.entry(labels).or_default() += 1;
        }
        assert_eq!(
            by_labels,
            BTreeMap::from([("imagenet/synsets-1k".into(), 8),])
        );
    }
}
