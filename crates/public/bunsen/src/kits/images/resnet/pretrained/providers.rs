//! # `ResNet` pretrained providers

use alloc::{
    sync::Arc,
    vec,
};

use crate::data::pretrained::{
    EmbeddedCatalog,
    PretrainedProvider,
};

/// The kit segment of a `ResNet` resource's path in the cache:
/// `<cache>/pretrained/resnet/<namespace>/<sha256>/<file>`.
pub const RESNET_KIT: &str = "resnet";

/// The key of the checkpoint in a `ResNet` resource bundle.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` label of every checkpoint here: a `PyTorch` state dict in
/// fp32.
pub const PYTORCH_FP32: &str = "pytorch fp32";

/// The namespace the torchvision checkpoints are cached under.
pub const TORCHVISION_NAMESPACE: &str = "torchvision";

/// The namespace the `pytorch-image-models` checkpoints are cached under.
pub const TIMM_NAMESPACE: &str = "timm";

/// The `ResNet` checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider: 14 rows in
/// two groups, `torchvision`, the reference `ImageNet` weights, and `timm`,
/// the `ResNet` Strikes Back weights and earlier ones.
///
/// Built from `validation/catalogs/resnet.yml` by `cargo validate catalog`.
pub static RESNET_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("resnet", include_str!("catalog.json"));

/// `ResNet`'s compiled-in providers, in search order: the well-known table.
pub fn default_resnet_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![Arc::new(RESNET_CATALOG.table())]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::resnet::pretrained::RESNET_PREFABS;

    /// Every row names a prefab the map has and is one pinned checkpoint
    /// under its group's namespace, fetched from the group's base; the
    /// digest's first eight hex digits are the tag in the file's name.
    #[test]
    fn test_every_row_names_a_prefab_and_one_checkpoint() {
        let prefabs: Vec<&str> = RESNET_PREFABS.iter().map(|p| p.name).collect();
        RESNET_CATALOG.catalog().validate(&prefabs).unwrap();
        let mut n = 0;
        for group in &RESNET_CATALOG.table().groups {
            for row in &group.items {
                n += 1;
                let prefab = row
                    .prefab
                    .as_deref()
                    .expect("every resnet row names a prefab");
                assert!(
                    RESNET_PREFABS.lookup_prefab(prefab).is_some(),
                    "{}: prefab {prefab:?} is not in RESNET_PREFABS",
                    group.id(row),
                );
                let map = &row.resources;
                map.validate().unwrap();
                assert_eq!(map.keys(), [CHECKPOINT], "{}", group.id(row));
                let checkpoint = map.get(CHECKPOINT).unwrap();
                assert_eq!(checkpoint.namespace, group.name, "{}", group.id(row));
                assert_eq!(checkpoint.kind.as_deref(), Some(PYTORCH_FP32));
                let sha256 = checkpoint.sha256.as_deref().expect("every row is pinned");
                let tag = checkpoint
                    .file
                    .rsplit_once('-')
                    .map(|(_, tail)| tail.split('.').next().unwrap_or(""))
                    .unwrap_or("");
                assert_eq!(tag.len(), 8, "{}", checkpoint.file);
                assert!(sha256.starts_with(tag), "{}: {sha256}", checkpoint.file);
                let urls = checkpoint.urls();
                assert_eq!(urls.len(), 1, "{}", group.id(row));
                assert!(
                    urls[0].ends_with(&format!("/{}", checkpoint.file)),
                    "{}",
                    urls[0]
                );
            }
        }
        assert_eq!(n, 14);
        assert_eq!(RESNET_PREFABS.iter().count(), 6);
    }

    /// The table answers `group/name` and bare names, torchvision first.
    #[test]
    fn test_the_table_lists_fourteen_refs() {
        let table = RESNET_CATALOG.table();
        let ids = table.ids();
        assert_eq!(ids.len(), 14);
        assert_eq!(ids[0], "well-known:torchvision/resnet18");
        assert_eq!(ids[5], "well-known:timm/resnet18_a1");

        let file = |spec: &str| {
            table
                .lookup(spec)
                .unwrap()
                .map(|row| row.resources.get(CHECKPOINT).unwrap().file.clone())
        };
        assert_eq!(
            file("resnet50").as_deref(),
            Some("resnet50-0676ba61.pth"),
            "torchvision answers a bare name first"
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
        assert_eq!(
            table
                .lookup("timm/resnet34")
                .unwrap()
                .unwrap()
                .prefab
                .as_deref(),
            Some("resnet34")
        );
    }
}
