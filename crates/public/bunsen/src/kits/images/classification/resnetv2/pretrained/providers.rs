//! # `ResNetV2` pretrained providers

use alloc::{
    sync::Arc,
    vec,
};

use crate::data::builds::{
    BuildProvider,
    HfProvider,
};

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
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            RESNETV2_CATALOG_CONTRACT,
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
            classification::resnetv2::ResNetV2ContractConfig,
            support::card::{
                LABEL_NAMES,
                LABELS,
            },
        },
    };

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint, its `config.json`, and its labels: `ImageNet`-1k's, or
    /// Google's 21k for the `BiT` `goog_in21k` heads.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        let factory = super::super::default_resnetv2_factory().unwrap();
        let rows = factory.list();
        assert_eq!(rows.len(), 24);
        let architectures = ResNetV2ContractConfig::prefab_names();
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
            BTreeMap::from([
                ("imagenet/synsets-1k".into(), 18),
                ("imagenet/synsets-21k-goog".into(), 6),
            ])
        );
    }
}
