//! # Pretrained `ResNet` models and configs
//!
//! The geometries, as [`PREFAB_RESNET_MAP`]; the checkpoints, as rows in
//! two groups of the [`WELL_KNOWN_TABLE`]: `torchvision`, the reference
//! `ImageNet` weights, and `timm`, the `ResNet` Strikes Back (`a1`, `a2`,
//! `a3`) and earlier `pytorch-image-models` releases. Each row names the
//! prefab it instantiates and is one `.pth` under its group's base URL,
//! pinned to its digest: `torchvision/resnet50`, `timm/resnet18_a1`, or
//! bare `resnet50`, which the torchvision group answers first.
//!
//! [`default_resnet_factory`] is the index; [`ResNetConstruct`] builds the
//! prefab's model and reads the checkpoint into it.

use alloc::vec;
use std::sync::Arc;

use crate::{
    data::pretrained::{
        PretrainedProvider,
        StaticBase,
        StaticPreFabConfig,
        StaticPreFabMap,
        StaticPretrained,
        StaticPretrainedGroup,
        StaticPretrainedTable,
        StaticResource,
        StaticResourceMap,
        WELL_KNOWN,
    },
    kits::bimm::resnet::ResNetContractConfig,
};

/// The kit segment of a `ResNet` resource's path in the cache:
/// `<cache>/pretrained/resnet/<namespace>/<sha256>/<file>`.
pub const RESNET_KIT: &str = "resnet";

/// The key of the checkpoint in a `ResNet` resource map.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` label of every checkpoint here: a `PyTorch` state dict in
/// fp32.
pub const PYTORCH_FP32: &str = "pytorch fp32";

/// The namespace the torchvision checkpoints are cached under.
pub const TORCHVISION_NAMESPACE: &str = "torchvision";

/// The namespace the `pytorch-image-models` checkpoints are cached under.
pub const TIMM_NAMESPACE: &str = "timm";

const TORCHVISION_BASE: StaticBase<'static> =
    StaticBase::Url("https://download.pytorch.org/models");

const TIMM_RSB_BASE: StaticBase<'static> = StaticBase::Url(
    "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights",
);

const TIMM_V01_BASE: StaticBase<'static> = StaticBase::Url(
    "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights",
);

/// One checkpoint map: a single `.pth` under a base URL, pinned to its
/// digest, whose first eight hex digits are the tag in the file's name.
macro_rules! checkpoint {
    ($name:ident, $map:literal, $file:literal, $sha256:literal, $namespace:expr, $base:expr) => {
        static $name: StaticResourceMap<'static> = StaticResourceMap {
            name: $map,
            description: "an ImageNet checkpoint",
            license: None,
            origin: None,
            namespace: $namespace,
            bases: &[$base],
            resources: &[StaticResource {
                key: CHECKPOINT,
                file: $file,
                sha256: Some($sha256),
                kind: Some(PYTORCH_FP32),
                sources: &[],
            }],
        };
    };
}

checkpoint!(
    TV_RESNET18,
    "torchvision/resnet18-f37072fd.pth",
    "resnet18-f37072fd.pth",
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    TORCHVISION_NAMESPACE,
    TORCHVISION_BASE
);
checkpoint!(
    TV_RESNET34,
    "torchvision/resnet34-b627a593.pth",
    "resnet34-b627a593.pth",
    "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
    TORCHVISION_NAMESPACE,
    TORCHVISION_BASE
);
checkpoint!(
    TV_RESNET50,
    "torchvision/resnet50-0676ba61.pth",
    "resnet50-0676ba61.pth",
    "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
    TORCHVISION_NAMESPACE,
    TORCHVISION_BASE
);
checkpoint!(
    TV_RESNET101,
    "torchvision/resnet101-63fe2227.pth",
    "resnet101-63fe2227.pth",
    "63fe2227b86e8f1f2063f43a75c84d195911b6a0eace650907dd3dc62dd49a0a",
    TORCHVISION_NAMESPACE,
    TORCHVISION_BASE
);
checkpoint!(
    TV_RESNET152,
    "torchvision/resnet152-394f9c45.pth",
    "resnet152-394f9c45.pth",
    "394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58",
    TORCHVISION_NAMESPACE,
    TORCHVISION_BASE
);
checkpoint!(
    TIMM_RESNET18_A1,
    "timm/resnet18_a1_0-d63eafa0.pth",
    "resnet18_a1_0-d63eafa0.pth",
    "d63eafa07a6e32a39d328e364f8c9f89d671444ecc7f02aa0f7eb8882af3dd29",
    TIMM_NAMESPACE,
    TIMM_RSB_BASE
);
checkpoint!(
    TIMM_RESNET18_A2,
    "timm/resnet18_a2_0-b61bd467.pth",
    "resnet18_a2_0-b61bd467.pth",
    "b61bd467647a03d3f7db590a1158552d2e66bf5ba62c2158ca9594c03de49923",
    TIMM_NAMESPACE,
    TIMM_RSB_BASE
);
checkpoint!(
    TIMM_RESNET18_A3,
    "timm/resnet18_a3_0-40c531c8.pth",
    "resnet18_a3_0-40c531c8.pth",
    "40c531c8324d963735b0c7ab9a9b2d7715c573a04d0b95776d29cc816a2ef93e",
    TIMM_NAMESPACE,
    TIMM_RSB_BASE
);
checkpoint!(
    TIMM_RESNET26,
    "timm/resnet26-9aa10e23.pth",
    "resnet26-9aa10e23.pth",
    "9aa10e237cec57dcaa067ee38d19ffc38a6323d185ef8a9903434291fd4187cb",
    TIMM_NAMESPACE,
    TIMM_V01_BASE
);
checkpoint!(
    TIMM_RESNET34_A1,
    "timm/resnet34_a1_0-46f8f793.pth",
    "resnet34_a1_0-46f8f793.pth",
    "46f8f7930534e471b9a139b278896db5d11d17b8ea3fcc90ad43deed9e5e27d7",
    TIMM_NAMESPACE,
    TIMM_RSB_BASE
);
checkpoint!(
    TIMM_RESNET34_A2,
    "timm/resnet34_a2_0-82d47d71.pth",
    "resnet34_a2_0-82d47d71.pth",
    "82d47d71dafac5343b070e5883c41321b45e4759bfe6aa25d375300c38b386a0",
    TIMM_NAMESPACE,
    TIMM_RSB_BASE
);
checkpoint!(
    TIMM_RESNET34_A3,
    "timm/resnet34_a3_0-a20cabb6.pth",
    "resnet34_a3_0-a20cabb6.pth",
    "a20cabb63e1baf7963ede853c5a38046aede79a7e675d3576714b59cb4ab7519",
    TIMM_NAMESPACE,
    TIMM_RSB_BASE
);
checkpoint!(
    TIMM_RESNET34,
    "timm/resnet34-43635321.pth",
    "resnet34-43635321.pth",
    "436353219a11d07c7c74cf1a686ed3b02f6c954d4c7863e2046822e047c6f246",
    TIMM_NAMESPACE,
    TIMM_V01_BASE
);
checkpoint!(
    TIMM_RESNET101_A1,
    "timm/resnet101_a1_0-cdcb52a9.pth",
    "resnet101_a1_0-cdcb52a9.pth",
    "cdcb52a9df09a641606696e5520e4fbbe8256ee4983d8b4dd1a1ef87a9b8846f",
    TIMM_NAMESPACE,
    TIMM_RSB_BASE
);

/// One torchvision row: the reference `ImageNet-1k` weights of a prefab, and
/// named after it.
const fn torchvision(
    prefab: &'static str,
    description: &'static str,
    maps: &'static [&'static StaticResourceMap<'static>],
) -> StaticPretrained<'static> {
    StaticPretrained {
        name: prefab,
        aliases: &[],
        description,
        license: Some("bsd-3-clause"),
        origin: Some("https://github.com/pytorch/vision"),
        prefab: Some(prefab),
        maps,
    }
}

/// One `pytorch-image-models` row.
const fn timm(
    name: &'static str,
    prefab: &'static str,
    description: &'static str,
    maps: &'static [&'static StaticResourceMap<'static>],
) -> StaticPretrained<'static> {
    StaticPretrained {
        name,
        aliases: &[],
        description,
        license: None,
        origin: Some("https://github.com/huggingface/pytorch-image-models"),
        prefab: Some(prefab),
        maps,
    }
}

static TV_RESNET18_ROW: StaticPretrained<'static> =
    torchvision("resnet18", "TorchVision ResNet-18", &[&TV_RESNET18]);
static TV_RESNET34_ROW: StaticPretrained<'static> =
    torchvision("resnet34", "TorchVision ResNet-34", &[&TV_RESNET34]);
static TV_RESNET50_ROW: StaticPretrained<'static> =
    torchvision("resnet50", "TorchVision ResNet-50", &[&TV_RESNET50]);
static TV_RESNET101_ROW: StaticPretrained<'static> =
    torchvision("resnet101", "TorchVision ResNet-101", &[&TV_RESNET101]);
static TV_RESNET152_ROW: StaticPretrained<'static> =
    torchvision("resnet152", "TorchVision ResNet-152", &[&TV_RESNET152]);

static TIMM_RESNET18_A1_ROW: StaticPretrained<'static> = timm(
    "resnet18_a1",
    "resnet18",
    "RSB Paper ResNet-18 a1",
    &[&TIMM_RESNET18_A1],
);
static TIMM_RESNET18_A2_ROW: StaticPretrained<'static> = timm(
    "resnet18_a2",
    "resnet18",
    "RSB Paper ResNet-18 a2",
    &[&TIMM_RESNET18_A2],
);
static TIMM_RESNET18_A3_ROW: StaticPretrained<'static> = timm(
    "resnet18_a3",
    "resnet18",
    "RSB Paper ResNet-18 a3",
    &[&TIMM_RESNET18_A3],
);
static TIMM_RESNET26_ROW: StaticPretrained<'static> = timm(
    "resnet26",
    "resnet26",
    "ResNet-26 pretrained on ImageNet",
    &[&TIMM_RESNET26],
);
static TIMM_RESNET34_A1_ROW: StaticPretrained<'static> = timm(
    "resnet34_a1",
    "resnet34",
    "RSB Paper ResNet-34 a1",
    &[&TIMM_RESNET34_A1],
);
static TIMM_RESNET34_A2_ROW: StaticPretrained<'static> = timm(
    "resnet34_a2",
    "resnet34",
    "RSB Paper ResNet-34 a2",
    &[&TIMM_RESNET34_A2],
);
static TIMM_RESNET34_A3_ROW: StaticPretrained<'static> = timm(
    "resnet34_a3",
    "resnet34",
    "RSB Paper ResNet-34 a3",
    &[&TIMM_RESNET34_A3],
);
static TIMM_RESNET34_ROW: StaticPretrained<'static> = timm(
    "resnet34",
    "resnet34",
    "ResNet-34 pretrained on ImageNet",
    &[&TIMM_RESNET34],
);
static TIMM_RESNET101_A1_ROW: StaticPretrained<'static> = timm(
    "resnet101_a1",
    "resnet101",
    "ResNet-101 pretrained on ImageNet",
    &[&TIMM_RESNET101_A1],
);

/// `TorchVision`'s reference `ImageNet-1k` weights, one per prefab.
pub static TORCHVISION: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "torchvision",
    description: "TorchVision's ImageNet-1k reference weights",
    license: Some("bsd-3-clause"),
    origin: Some("https://github.com/pytorch/vision"),
    items: &[
        &TV_RESNET18_ROW,
        &TV_RESNET34_ROW,
        &TV_RESNET50_ROW,
        &TV_RESNET101_ROW,
        &TV_RESNET152_ROW,
    ],
};

/// `pytorch-image-models`' weights: the `ResNet` Strikes Back recipes and
/// the earlier release.
pub static TIMM: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "timm",
    description: "pytorch-image-models' ImageNet-1k weights: ResNet Strikes Back (a1, a2, a3) and earlier",
    license: None,
    origin: Some("https://github.com/huggingface/pytorch-image-models"),
    items: &[
        &TIMM_RESNET18_A1_ROW,
        &TIMM_RESNET18_A2_ROW,
        &TIMM_RESNET18_A3_ROW,
        &TIMM_RESNET26_ROW,
        &TIMM_RESNET34_A1_ROW,
        &TIMM_RESNET34_A2_ROW,
        &TIMM_RESNET34_A3_ROW,
        &TIMM_RESNET34_ROW,
        &TIMM_RESNET101_A1_ROW,
    ],
};

/// The checkpoints bunsen knows by name, behind the [`WELL_KNOWN`]
/// provider: `well-known:torchvision/resnet50`, `torchvision/resnet50`,
/// or `resnet50`.
pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
    name: WELL_KNOWN,
    description: "the ResNet checkpoints bunsen knows by name",
    groups: &[&TORCHVISION, &TIMM],
};

/// `ResNet`'s compiled-in providers, in search order: the well-known table.
pub fn default_resnet_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![Arc::new(WELL_KNOWN_TABLE.to_table())]
}

/// The public geometries: [`ResNet`](super::`ResNet`) configs by name.
pub static PREFAB_RESNET_MAP: StaticPreFabMap<ResNetContractConfig> = StaticPreFabMap {
    name: "resnet",
    description: "Well-Know ResNet configs",

    items: &[
        &StaticPreFabConfig {
            name: "resnet18",
            description: "ResNet-18 [2, 2, 2, 2] BasicBlocks",
            builder: || ResNetContractConfig::new(vec![2, 2, 2, 2], 1000),
        },
        &StaticPreFabConfig {
            name: "resnet26",
            description: "ResNet-26 [2, 2, 2, 2] Bottleneck",
            builder: || ResNetContractConfig::new(vec![2, 2, 2, 2], 1000).with_bottleneck(true),
        },
        &StaticPreFabConfig {
            name: "resnet34",
            description: "ResNet-34 [3, 4, 6, 3] BasicBlocks",
            builder: || ResNetContractConfig::new(vec![3, 4, 6, 3], 1000),
        },
        &StaticPreFabConfig {
            name: "resnet50",
            description: "ResNet-50 [3, 4, 6, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 4, 6, 3], 1000).with_bottleneck(true),
        },
        &StaticPreFabConfig {
            name: "resnet101",
            description: "ResNet-101 [3, 4, 23, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 4, 23, 3], 1000).with_bottleneck(true),
        },
        &StaticPreFabConfig {
            name: "resnet152",
            description: "ResNet-152 [3, 8, 36, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 8, 36, 3], 1000).with_bottleneck(true),
        },
    ],
};

#[cfg(feature = "store")]
mod construct {
    use std::sync::Arc;

    use burn::prelude::Backend;

    use super::{
        CHECKPOINT,
        PREFAB_RESNET_MAP,
        RESNET_KIT,
        default_resnet_providers,
    };
    use crate::{
        burner::module::ModuleInit,
        data::pretrained::{
            Construct,
            LoadedResources,
            PretrainedFactory,
            PretrainedRef,
            ResourceMap,
        },
        errors::{
            BunsenError,
            BunsenResult,
        },
        kits::bimm::resnet::{
            ResNet,
            ResNetContractConfig,
        },
    };

    /// How a `ResNet` pretrained is built: the config the model is
    /// initialised from before the checkpoint is read into it.
    ///
    /// A checkpoint does not describe its own geometry, so the config comes
    /// from the prefab the row names, or from
    /// [`with_config`](Self::with_config), which wins and is what a given
    /// path needs. An example that swaps the activation before the weights
    /// land passes the prefab's config, modified, here.
    #[derive(Clone, Debug, Default)]
    pub struct ResNetConstruct {
        /// The config to build from, overriding the prefab's.
        pub config: Option<ResNetContractConfig>,
    }

    impl ResNetConstruct {
        /// Builds from the prefab the row names.
        pub fn new() -> Self {
            Self::default()
        }

        /// Sets the config to build from, which wins over the prefab's.
        pub fn with_config(
            mut self,
            config: ResNetContractConfig,
        ) -> Self {
            self.config = Some(config);
            self
        }

        /// The config `model` is built from: the explicit one, else the
        /// prefab the row names.
        ///
        /// # Errors
        /// [`BunsenError::Invalid`] when there is neither: a given
        /// checkpoint needs [`with_config`](Self::with_config).
        pub fn config_for(
            &self,
            model: &PretrainedRef,
        ) -> BunsenResult<ResNetContractConfig> {
            if let Some(config) = &self.config {
                return Ok(config.clone());
            }
            model
                .prefab(&PREFAB_RESNET_MAP)
                .map(|prefab| prefab.to_config())
                .ok_or_else(|| {
                    BunsenError::Invalid(format!(
                        "{}: names no prefab to build from; a given checkpoint needs `with_config`",
                        model.id()
                    ))
                })
        }
    }

    /// `ResNet`'s factory: [`default_resnet_providers`] behind
    /// [`ResNetConstruct`].
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if two of the defaults share a name, which
    /// the tests pin they do not.
    pub fn default_resnet_factory() -> BunsenResult<PretrainedFactory<ResNetConstruct>> {
        PretrainedFactory::new().with_providers(default_resnet_providers())
    }

    impl Construct for ResNetConstruct {
        type Built<B: Backend> = ResNet<B>;

        const KIT: &'static str = RESNET_KIT;

        /// Every row is a `PyTorch` state dict; the config comes at
        /// construction, from the row's prefab or the caller's.
        fn for_map(_map: &ResourceMap) -> BunsenResult<Self> {
            Ok(Self::new())
        }

        /// Initialises the config's model and reads the checkpoint into it.
        fn construct<B: Backend>(
            &self,
            model: &PretrainedRef,
            loaded: &LoadedResources,
            device: &B::Device,
        ) -> BunsenResult<Arc<ResNet<B>>> {
            let config = self.config_for(model)?;
            let resnet: ResNet<B> = config.try_init(device)?;
            Ok(Arc::new(
                resnet.load_pytorch_weights(loaded.expect(CHECKPOINT)?)?,
            ))
        }
    }
}

#[cfg(feature = "store")]
pub use construct::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row names a prefab the map has and is one pinned checkpoint
    /// under its group's namespace, fetched from the group's base; the
    /// digest's first eight hex digits are the tag in the file's name.
    #[test]
    fn test_every_row_names_a_prefab_and_one_checkpoint() {
        let mut n = 0;
        for group in WELL_KNOWN_TABLE.groups {
            for row in group.items {
                n += 1;
                let prefab = row.prefab.expect("every resnet row names a prefab");
                assert!(
                    PREFAB_RESNET_MAP.lookup_prefab(prefab).is_some(),
                    "{}: prefab {prefab:?} is not in PREFAB_RESNET_MAP",
                    group.id(row),
                );
                let map = row.try_to_map().unwrap();
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
        assert_eq!(PREFAB_RESNET_MAP.iter().count(), 6);
    }

    /// The table answers `group/name` and bare names, torchvision first.
    #[test]
    fn test_the_table_lists_fourteen_refs() {
        let table = WELL_KNOWN_TABLE.to_table();
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

    /// The default factory is the well-known table alone, serving the
    /// kit; one prefab has many rows across both groups.
    #[cfg(feature = "store")]
    #[test]
    fn test_the_defaults_register() {
        use crate::data::pretrained::PretrainedFactory;
        let factory = default_resnet_factory().unwrap();
        assert_eq!(factory.kit(), RESNET_KIT);
        assert_eq!(factory.providers().len(), 1);
        assert_eq!(factory.ids().len(), 14);
        let (provider, row) = factory.lookup("resnet18_a1").unwrap();
        assert_eq!(provider, "well-known");
        assert_eq!(row.name, "timm/resnet18_a1");
        let for_34: Vec<String> = factory
            .for_prefab("resnet34")
            .iter()
            .map(|(p, row)| format!("{p}:{}", row.name))
            .collect();
        assert_eq!(
            for_34,
            [
                "well-known:torchvision/resnet34",
                "well-known:timm/resnet34_a1",
                "well-known:timm/resnet34_a2",
                "well-known:timm/resnet34_a3",
                "well-known:timm/resnet34",
            ]
        );
        assert!(
            PretrainedFactory::<ResNetConstruct>::new()
                .with_providers(default_resnet_providers())
                .unwrap()
                .with_providers(default_resnet_providers())
                .is_err()
        );
    }

    /// The hook builds from the prefab a row names, or from an explicit
    /// config, which wins; a given checkpoint with neither is refused
    /// before its bytes are read.
    #[cfg(feature = "store")]
    #[test]
    fn test_the_hook_takes_the_prefab_or_an_explicit_config() {
        use std::fs;

        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::{
                    Construct,
                    Deferred,
                    PretrainedCache,
                    PretrainedCacheOptions,
                    PretrainedRef,
                    ResourceMap,
                },
            },
            errors::BunsenError,
            support::testing::{
                CpuBackend,
                default_device,
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    crate::data::cache::BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let factory = default_resnet_factory().unwrap();
        let named = factory.resolve("resnet50", &cache).unwrap();
        let hook = named.hook.clone();
        let named = named.model;
        let resnet50 = PREFAB_RESNET_MAP
            .expect_lookup_prefab("resnet50")
            .to_config();
        assert_eq!(
            format!("{:?}", hook.config_for(&named).unwrap()),
            format!("{resnet50:?}")
        );

        let resnet18 = PREFAB_RESNET_MAP
            .expect_lookup_prefab("resnet18")
            .to_config();
        let explicit = hook.clone().with_config(resnet18.clone());
        assert_eq!(
            format!("{:?}", explicit.config_for(&named).unwrap()),
            format!("{resnet18:?}"),
            "an explicit config wins over the prefab"
        );

        let file = dir.path().join("ckpt.pth");
        fs::write(&file, b"not a state dict").unwrap();
        assert!(
            factory.resolve(file.to_str().unwrap(), &cache).is_err(),
            "a path is not a name the factory knows"
        );
        let given = PretrainedRef::from(ResourceMap::given("mine", CHECKPOINT, &file));
        let err = hook.config_for(&given).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(msg) if msg.contains("with_config")),
            "{err}"
        );
        assert_eq!(
            format!("{:?}", explicit.config_for(&given).unwrap()),
            format!("{resnet18:?}")
        );

        // Through the pathway: the refusal comes before the file is read.
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let err =
            Deferred::<ResNetConstruct>::from_map(ResourceMap::given("mine", CHECKPOINT, &file))
                .unwrap()
                .load::<CpuBackend>(&cache, &default_device())
                .unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err}");
        assert_eq!(<ResNetConstruct as Construct>::KIT, RESNET_KIT);
    }
}
