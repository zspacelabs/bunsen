//! # `ViT` pretrained providers

use alloc::{
    sync::Arc,
    vec,
};

use crate::data::builds::{
    BuildProvider,
    HfProvider,
};

/// The `ViT` kit's name: its rows' `kit` tag.
pub const VIT_KIT: &str = "vit";

/// The key of the checkpoint in a `ViT` resource set.
pub use crate::data::builds::CHECKPOINT;

/// What the vit kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const VIT_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const VIT_CATALOG_CONTRACT: &str = "vit-classifier";

/// `ViT`'s compiled-in providers, in search order: the catalog provider, then
/// `hf:org/repo`, a timm hub repo of a `ViT` the kit builds
/// (`hf:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_vit_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            VIT_CATALOG_CONTRACT,
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
            classification::vit::ViTContractConfig,
            support::card::{
                LABEL_NAMES,
                LABELS,
            },
        },
    };

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint and its `config.json`. A row with labels names them too;
    /// the headless towers (`DINOv2`'s, `SigLIP`'s, C-RADIO's), the CLIP
    /// projections and the iNaturalist heads have none.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        let factory = super::super::default_vit_factory().unwrap();
        let rows = factory.list();
        assert_eq!(rows.len(), 305);
        let architectures = ViTContractConfig::prefab_names();
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
            *by_labels.entry(labels.unwrap_or_default()).or_default() += 1;
        }
        assert_eq!(
            by_labels,
            BTreeMap::from([
                (String::new(), 177),
                ("imagenet/synsets-12k".into(), 14),
                ("imagenet/synsets-1k".into(), 99),
                ("imagenet/synsets-21k-goog".into(), 14),
                ("imagenet/synsets-21k-miil".into(), 1),
            ])
        );
    }
}
