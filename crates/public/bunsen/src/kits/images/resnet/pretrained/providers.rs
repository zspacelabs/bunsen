//! # `ResNet` pretrained providers

use alloc::{
    sync::Arc,
    vec,
};

use crate::data::builds::BuildProvider;

/// The `ResNet` kit's name: its rows' `kit` tag.
pub const RESNET_KIT: &str = "resnet";

/// The key of the checkpoint in a `ResNet` resource set.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` label of every checkpoint here: a `PyTorch` state dict in
/// fp32.
pub const PYTORCH_FP32: &str = "pytorch fp32";

/// What the resnet kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const RESNET_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const RESNET_CATALOG_CONTRACT: &str = "resnet-classifier";

/// `ResNet`'s compiled-in providers, in search order: the catalog provider.
pub fn default_resnet_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![Arc::new(
        crate::data::catalog::BuiltinCatalog::expect_provider(RESNET_CATALOG_CONTRACT, RESNET_KIT),
    )]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::resnet::pretrained::RESNET_PREFABS;

    /// Every row states the geometry of a prefab the map has and is one
    /// pinned checkpoint under its namespace, fetched from that namespace's
    /// base; the digest's first eight hex digits are the tag in the file's
    /// name.
    #[test]
    fn test_every_row_states_a_prefab_and_one_checkpoint() {
        use crate::data::builds::GEOMETRY;
        let factory = super::super::default_resnet_factory().unwrap();
        let rows = factory.list();
        for (_, row) in &rows {
            let name = &row.name;
            let geometry = row
                .args
                .get(GEOMETRY)
                .expect("every resnet row states its geometry");
            let config: crate::kits::images::resnet::ResNetContractConfig =
                serde_json::from_value(geometry.clone()).unwrap();
            assert!(
                RESNET_PREFABS
                    .iter()
                    .any(|p| serde_json::to_value(p.to_config()).unwrap() == *geometry),
                "{name}: geometry {config:?} is no prefab in RESNET_PREFABS",
            );
            let map = &row.resources;
            map.validate().unwrap();
            assert_eq!(map.keys(), [CHECKPOINT], "{name}");
            let checkpoint = map.get(CHECKPOINT).unwrap();
            let namespace = name.split('/').next().unwrap();
            assert!(["torchvision", "timm"].contains(&namespace), "{name}");
            assert!(
                checkpoint.name.starts_with(&format!("{namespace}/")),
                "{name}: {}",
                checkpoint.name
            );
            assert_eq!(checkpoint.format.as_deref(), Some(PYTORCH_FP32));
            let sha256 = checkpoint.sha256.as_deref().expect("every row is pinned");
            let tag = checkpoint
                .file
                .rsplit_once('-')
                .map(|(_, tail)| tail.split('.').next().unwrap_or(""))
                .unwrap_or("");
            assert_eq!(tag.len(), 8, "{}", checkpoint.file);
            assert!(sha256.starts_with(tag), "{}: {sha256}", checkpoint.file);
            let urls = checkpoint.urls();
            assert_eq!(urls.len(), 1, "{name}");
            assert!(
                urls[0].ends_with(&format!("/{}", checkpoint.file)),
                "{}",
                urls[0]
            );
        }
        assert_eq!(rows.len(), 14);
        assert_eq!(RESNET_PREFABS.iter().count(), 6);
    }

    /// The factory answers full names and shorter spellings, torchvision's
    /// rows listed first; a bare name both namespaces have is ambiguous.
    #[test]
    fn test_the_factory_lists_fourteen_refs() {
        let factory = super::super::default_resnet_factory().unwrap();
        let ids = factory.ids();
        assert_eq!(ids.len(), 14);
        assert_eq!(ids[0], "catalog:torchvision/resnet18");
        assert_eq!(ids[5], "catalog:timm/resnet18_a1");

        let file = |spec: &str| {
            factory
                .lookup(spec)
                .ok()
                .map(|(_, row)| row.resources.get(CHECKPOINT).unwrap().file.clone())
        };
        assert_eq!(file("resnet50").as_deref(), Some("resnet50-0676ba61.pth"));
        assert!(
            factory.lookup("resnet34").is_err(),
            "both namespaces have a resnet34"
        );
        assert_eq!(
            file("torchvision/resnet50").as_deref(),
            Some("resnet50-0676ba61.pth")
        );
        assert_eq!(
            file("timm/resnet34").as_deref(),
            Some("resnet34-43635321.pth")
        );
        assert_eq!(
            file("resnet18_a1").as_deref(),
            Some("resnet18_a1_0-d63eafa0.pth")
        );
        assert_eq!(file("resnet50_a1"), None);
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::kits::support::catalog_contracts::testing::contract_lines;
        assert_eq!(
            contract_lines(RESNET_CONTRACT),
            [
                format!("resnet-classifier: {CHECKPOINT} [{PYTORCH_FP32}]"),
                "resnet-classifier: geometry: \"geometry\"".to_string(),
            ]
        );
    }
}
