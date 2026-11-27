//! # Swin V2 pretrained providers

use alloc::{
    sync::Arc,
    vec,
};

use crate::data::builds::{
    BuildProvider,
    HfProvider,
};

/// The Swin V2 kit's name: its rows' `kit` tag.
pub const SWIN_V2_KIT: &str = "swin-v2";

/// The key of the checkpoint in a Swin V2 resource set.
pub use crate::data::builds::CHECKPOINT;

/// What the swin v2 kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const SWIN_V2_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const SWIN_V2_CATALOG_CONTRACT: &str = "swin-v2-classifier";

/// Swin V2's compiled-in providers, in search order: the catalog provider, then
/// `hf:org/repo`, a timm hub repo of a Swin V2 the kit builds
/// (`hf:timm/swinv2_tiny_window8_256.ms_in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_swin_v2_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            SWIN_V2_CATALOG_CONTRACT,
        )),
        Arc::new(HfProvider::new()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{
            builds::CONFIG,
            pretrained::Prefabs,
        },
        kits::images::{
            classification::swin::v2::SwinV2ContractConfig,
            support::card::{
                LABEL_NAMES,
                LABELS,
            },
        },
    };

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint and its `config.json`. Each binds `ImageNet`-1k's labels
    /// but the two `ms_in22k` pretrainings, whose 21841 classes no label set
    /// of the catalog's fits.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        let factory = super::super::default_swin_v2_factory().unwrap();
        let rows = factory.list();
        assert_eq!(rows.len(), 12);
        let architectures = SwinV2ContractConfig::prefab_names();
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
            let labelled = !name.ends_with(".ms_in22k");
            assert_eq!(map.get(LABELS).is_some(), labelled, "{name}");
            assert_eq!(map.get(LABEL_NAMES).is_some(), labelled, "{name}");
        }
    }
}
