//! # `ResNet` pretrained providers

use alloc::{
    sync::Arc,
    vec,
};

use crate::data::builds::{
    BuildProvider,
    HfProvider,
};

/// The `ResNet` kit's name: its rows' `kit` tag.
pub const RESNET_KIT: &str = "resnet";

/// The key of the checkpoint in a `ResNet` resource set.
pub use crate::data::builds::CHECKPOINT;

/// The `kind` label of every checkpoint here: a `PyTorch` state dict in
/// fp32.
pub const PYTORCH_FP32: &str = "pytorch fp32";

/// What the resnet kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const RESNET_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const RESNET_CATALOG_CONTRACT: &str = "resnet-classifier";

/// `ResNet`'s compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a timm hub repo of a `ResNet` the kit builds
/// (`hf:timm/resnet50.a1_in1k`): its `model.safetensors` and its
/// `config.json`, whose architecture names the geometry.
pub fn default_resnet_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            RESNET_CATALOG_CONTRACT,
        )),
        Arc::new(HfProvider::new()),
    ]
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{
        data::pretrained::Prefabs,
        kits::images::classification::resnet::ResNetContractConfig,
    };

    /// Every row binds its head's `ImageNet` labels. A torchvision row states
    /// the geometry of a prefab and is one pinned `.pth` under torchvision's
    /// base, the digest's first eight hex digits the tag in its name. A timm
    /// row is a timm hub repo of a prefab's architecture, pinned: its
    /// safetensors and its `config.json`, which give the geometry.
    #[test]
    fn test_every_row_binds_labels_and_a_pinned_checkpoint() {
        use crate::{
            data::builds::{
                CONFIG,
                GEOMETRY,
            },
            kits::images::support::card::{
                LABEL_NAMES,
                LABELS,
            },
        };
        let factory = super::super::default_resnet_factory().unwrap();
        let rows = factory.list();
        let mut counts = (0, 0);
        for (_, row) in &rows {
            let name = &row.name;
            let map = &row.resource_set();
            map.validate().unwrap();
            let labels = &map.get(LABELS).unwrap().name;
            // An `ImageNet`-12k pretraining is tagged `in12k`; its 1k
            // fine-tunes `in12k_ft_in1k`.
            if name.to_string().ends_with("in12k") {
                assert_eq!(labels, "imagenet/synsets-12k", "{name}");
            } else {
                assert_eq!(labels, "imagenet/synsets-1k", "{name}");
            }
            assert_eq!(
                map.get(LABEL_NAMES).unwrap().name,
                "imagenet/lemmas",
                "{name}"
            );
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.sha256.is_some(), "{name}: every row is pinned");
            match name.segments()[0].as_str() {
                "torchvision" => {
                    counts.0 += 1;
                    let geometry = &row
                        .args
                        .get(GEOMETRY)
                        .expect("every torchvision row states its geometry")
                        .value;
                    assert!(
                        ResNetContractConfig::PREFABS
                            .iter()
                            .any(|(_, prefab)| serde_json::to_value(prefab()).unwrap() == *geometry),
                        "{name}: a geometry no prefab of ResNetContractConfig has",
                    );
                    assert_eq!(map.keys(), [CHECKPOINT, LABEL_NAMES, LABELS], "{name}");
                    assert_eq!(checkpoint.format.as_deref(), Some(PYTORCH_FP32));
                    let tag = checkpoint
                        .file
                        .rsplit_once('-')
                        .map(|(_, tail)| tail.split('.').next().unwrap_or(""))
                        .unwrap_or("");
                    assert_eq!(tag.len(), 8, "{}", checkpoint.file);
                    assert!(
                        checkpoint.sha256.as_deref().unwrap().starts_with(tag),
                        "{}",
                        checkpoint.file
                    );
                }
                "timm" => {
                    counts.1 += 1;
                    assert!(row.args.get(GEOMETRY).is_none(), "{name}");
                    let repo = name.segments()[1].as_str();
                    let architecture = repo.split_once('.').map(|(a, _)| a).unwrap_or(repo);
                    assert!(
                        ResNetContractConfig::prefab(architecture).is_some(),
                        "{name}: {architecture} is no prefab"
                    );
                    assert_eq!(
                        map.keys(),
                        [CHECKPOINT, CONFIG, LABEL_NAMES, LABELS],
                        "{name}"
                    );
                    assert_eq!(checkpoint.file, "model.safetensors");
                    assert_eq!(checkpoint.format.as_deref(), Some("safetensors"));
                    assert_eq!(map.get(CONFIG).unwrap().file, "config.json");
                }
                other => panic!("{name}: a row in {other}"),
            }
        }
        assert_eq!(counts, (5, 140));
        assert_eq!(ResNetContractConfig::PREFABS.len(), 76);
    }

    /// The factory answers full names and shorter spellings, torchvision's
    /// rows listed first; a legacy timm name is an alias of its hub repo; a
    /// bare name both namespaces have is ambiguous.
    #[test]
    fn test_the_factory_lists_its_refs() {
        let factory = super::super::default_resnet_factory().unwrap();
        let ids = factory.ids();
        assert_eq!(ids.len(), 145);
        assert_eq!(ids[0], "catalog:torchvision/resnet18");
        assert_eq!(ids[5], "catalog:timm/resnet18.a1_in1k");

        let file = |spec: &str| {
            factory.lookup(spec).ok().map(|(_, row)| {
                (
                    row.name.to_string(),
                    row.resource_set().get(CHECKPOINT).unwrap().file.clone(),
                )
            })
        };
        let row = |name: &str, file: &str| Some((name.to_string(), file.to_string()));
        assert_eq!(
            file("resnet50"),
            row("torchvision/resnet50", "resnet50-0676ba61.pth")
        );
        assert!(
            factory.lookup("resnet34").is_err(),
            "both namespaces have a resnet34"
        );
        assert_eq!(
            file("timm/resnet34"),
            row("timm/resnet34.bt_in1k", "model.safetensors")
        );
        assert_eq!(
            file("resnet18_a1"),
            row("timm/resnet18.a1_in1k", "model.safetensors")
        );
        assert_eq!(
            file("resnet50.a1_in1k"),
            row("timm/resnet50.a1_in1k", "model.safetensors")
        );
        assert_eq!(file("resnet50_a1"), None);
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::{
            data::builds::CONFIG,
            kits::{
                images::support::card::{
                    EVAL_TRANSFORM,
                    LABEL_NAMES,
                    LABELS,
                },
                support::pretrained::catalog_contracts::testing::contract_lines,
            },
        };
        assert_eq!(
            contract_lines(RESNET_CONTRACT),
            [
                format!("resnet-classifier: {CHECKPOINT} [{PYTORCH_FP32}, safetensors]"),
                format!("resnet-classifier: {CONFIG}? [json]"),
                format!("resnet-classifier: {LABEL_NAMES}? [synset-lemmas]"),
                format!("resnet-classifier: {LABELS}? [synsets]"),
                format!("resnet-classifier: {EVAL_TRANSFORM}?: \"$object\""),
                "resnet-classifier: geometry?: \"geometry\"".to_string(),
            ]
        );
    }
}
