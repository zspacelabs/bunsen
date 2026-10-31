//! # nanochat pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::NanoChatConstruct;
use crate::{
    data::pretrained::{
        PretrainedCache,
        PretrainedFactory,
        PretrainedProvider,
        StaticBase,
        StaticPretrained,
        StaticPretrainedGroup,
        StaticPretrainedTable,
        StaticResource,
        StaticResourceMap,
        WELL_KNOWN,
    },
    errors::BunsenResult,
    kits::gpts::nanochat::{
        CHECKPOINT,
        META,
        NanoChatBundle,
        TOKEN_BYTES,
        TOKENIZER,
    },
};

/// The kit segment of a nanochat resource's path in the cache:
/// `<cache>/pretrained/nanochat/<namespace>/<sha256>/<file>`.
pub const NANOCHAT_KIT: &str = "nanochat";

/// The `kind` of a `PyTorch` state dict (`torch.save`), as upstream writes
/// `model_<step>.pt` and `token_bytes.pt`.
pub const PYTORCH: &str = "pytorch";

/// The `kind` of a nanochat checkpoint's `meta_<step>.json`.
pub const META_JSON: &str = "nanochat-meta.json";

/// The `kind` of a pickled `tiktoken` `Encoding`, as upstream writes
/// `tokenizer.pkl`.
pub const TIKTOKEN_PICKLE: &str = "tiktoken-pickle";

/// `nanochat-students/base-d20`'s files, at a fixed revision.
pub static BASE_D20_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nanochat-students/base-d20",
    description: "nanochat-students' d20 after pretraining (step 21400): upstream's 2025 model, 561M, base: model_*.pt, meta, tokenizer",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/nanochat-students/base-d20"),
    namespace: "nanochat-students",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nanochat-students/base-d20/resolve/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model_021400.pt",
            sha256: Some("574670de45c667a98fe72c2d145f24007ecc1778721e2481e41500842dd3ba13"),
            kind: Some(PYTORCH),
            sources: &[],
        },
        StaticResource {
            key: META,
            file: "meta_021400.json",
            sha256: Some("a34a9b144a30d2a8066e64a91b7becc8579ea355c075e20427c28afb83ef106c"),
            kind: Some(META_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.pkl",
            sha256: Some("8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b"),
            kind: Some(TIKTOKEN_PICKLE),
            sources: &[],
        },
        StaticResource {
            key: TOKEN_BYTES,
            file: "token_bytes.pt",
            sha256: Some("e1b6cdee5d02fe1018b2b1d2ae5b736be665f9c0e7d10c81dcf935e7efaf8cb5"),
            kind: Some(PYTORCH),
            sources: &[],
        },
    ],
};

static BASE_D20: StaticPretrained<'static> = StaticPretrained {
    name: "base-d20",
    aliases: &["nanochat-d20-base"],
    description: "nanochat-students' d20 after pretraining (step 21400): upstream's 2025 model, 561M, base",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/nanochat-students/base-d20"),
    prefab: Some("nanochat-d20"),
    maps: &[&BASE_D20_FILES],
};

/// `nanochat-students/mid-d20`'s files, at a fixed revision.
pub static MID_D20_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nanochat-students/mid-d20",
    description: "nanochat-students' d20 after midtraining (step 765): upstream's 2025 model, 561M, chat format: model_*.pt, meta, tokenizer",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/nanochat-students/mid-d20"),
    namespace: "nanochat-students",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nanochat-students/mid-d20/resolve/8db9593636d2b4dd8546cb211f3586f90da49269",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model_000765.pt",
            sha256: Some("cc0560b6adf23e663191ba3e73707381b648e94a6489493c2589baf87eb1863a"),
            kind: Some(PYTORCH),
            sources: &[],
        },
        StaticResource {
            key: META,
            file: "meta_000765.json",
            sha256: Some("1282220593eaf3e120b3e6be0eb3894e054638d46021511ede8d60b074ece58d"),
            kind: Some(META_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.pkl",
            sha256: Some("8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b"),
            kind: Some(TIKTOKEN_PICKLE),
            sources: &[],
        },
        StaticResource {
            key: TOKEN_BYTES,
            file: "token_bytes.pt",
            sha256: Some("e1b6cdee5d02fe1018b2b1d2ae5b736be665f9c0e7d10c81dcf935e7efaf8cb5"),
            kind: Some(PYTORCH),
            sources: &[],
        },
    ],
};

static MID_D20: StaticPretrained<'static> = StaticPretrained {
    name: "mid-d20",
    aliases: &["nanochat-d20-mid"],
    description: "nanochat-students' d20 after midtraining (step 765): upstream's 2025 model, 561M, chat format",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/nanochat-students/mid-d20"),
    prefab: Some("nanochat-d20"),
    maps: &[&MID_D20_FILES],
};

/// `nanochat-students/rl-d20`'s files, at a fixed revision.
pub static RL_D20_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nanochat-students/rl-d20",
    description: "nanochat-students' d20 after its RL stage (step 650): upstream's 2025 model, 561M, chat: model_*.pt, meta, tokenizer",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/nanochat-students/rl-d20"),
    namespace: "nanochat-students",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nanochat-students/rl-d20/resolve/22fbb6a82b381a7b0b4f04f3bc5f3580a996745e",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model_000650.pt",
            sha256: Some("ff2eee182e2aa396615d3b481ddf17884a7dbabb3caa47f66eede343135accff"),
            kind: Some(PYTORCH),
            sources: &[],
        },
        StaticResource {
            key: META,
            file: "meta_000650.json",
            sha256: Some("353d774b8548a317a5e71762b775ee8ea506332f9b2acf8dff4fbfd9e813131a"),
            kind: Some(META_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.pkl",
            sha256: Some("8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b"),
            kind: Some(TIKTOKEN_PICKLE),
            sources: &[],
        },
        StaticResource {
            key: TOKEN_BYTES,
            file: "token_bytes.pt",
            sha256: Some("e1b6cdee5d02fe1018b2b1d2ae5b736be665f9c0e7d10c81dcf935e7efaf8cb5"),
            kind: Some(PYTORCH),
            sources: &[],
        },
    ],
};

static RL_D20: StaticPretrained<'static> = StaticPretrained {
    name: "rl-d20",
    aliases: &["nanochat-d20-rl"],
    description: "nanochat-students' d20 after its RL stage (step 650): upstream's 2025 model, 561M, chat",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/nanochat-students/rl-d20"),
    prefab: Some("nanochat-d20"),
    maps: &[&RL_D20_FILES],
};

/// `nitishpandey04/nanochat-d18-base`'s files, at a fixed revision.
pub static D18_BASE_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nitishpandey04/nanochat-d18-base",
    description: "nitishpandey04's d18 after pretraining (step 3712): upstream's 2026 model, 702M, base: model_*.pt, meta, tokenizer",
    license: Some("MIT"),
    origin: Some("https://huggingface.co/nitishpandey04/nanochat-d18-base"),
    namespace: "nitishpandey04",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nitishpandey04/nanochat-d18-base/resolve/bd4999f3186942207bac6b4b41ff743381e5fc8b",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "base_checkpoints/d18/model_003712.pt",
            sha256: Some("979a5ac9ea785da98d38dd9b58bd07d39d06230a6644d1f4e157f045499989b3"),
            kind: Some(PYTORCH),
            sources: &[],
        },
        StaticResource {
            key: META,
            file: "base_checkpoints/d18/meta_003712.json",
            sha256: Some("7ecc0e6f4f0315d106d4fb213fec593879456475a21de34b71c329a4a8a4e7b4"),
            kind: Some(META_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer/tokenizer.pkl",
            sha256: Some("ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2"),
            kind: Some(TIKTOKEN_PICKLE),
            sources: &[],
        },
        StaticResource {
            key: TOKEN_BYTES,
            file: "tokenizer/token_bytes.pt",
            sha256: Some("009ef93d20dd4684497c19c4b7fed29278b53f20022d2dc39264e27b9eefa2a8"),
            kind: Some(PYTORCH),
            sources: &[],
        },
    ],
};

static D18_BASE: StaticPretrained<'static> = StaticPretrained {
    name: "nanochat-d18-base",
    aliases: &[],
    description: "nitishpandey04's d18 after pretraining (step 3712): upstream's 2026 model, 702M, base",
    license: Some("MIT"),
    origin: Some("https://huggingface.co/nitishpandey04/nanochat-d18-base"),
    prefab: Some("nanochat-d18-2026"),
    maps: &[&D18_BASE_FILES],
};

/// `nitishpandey04/nanochat-d18-sft`'s files, at a fixed revision.
pub static D18_SFT_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nitishpandey04/nanochat-d18-sft",
    description: "nitishpandey04's d18 after SFT (step 478): upstream's 2026 model, 702M, chat: model_*.pt, meta, tokenizer",
    license: Some("MIT"),
    origin: Some("https://huggingface.co/nitishpandey04/nanochat-d18-sft"),
    namespace: "nitishpandey04",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nitishpandey04/nanochat-d18-sft/resolve/e2c7c119fee2a0caf6a0c69704ccbe57bad87e63",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "chatsft_checkpoints/d18/model_000478.pt",
            sha256: Some("951589ed950a12358d69ab3180c39a237cf24fea80f01a4b6691f90e3bacf17a"),
            kind: Some(PYTORCH),
            sources: &[],
        },
        StaticResource {
            key: META,
            file: "chatsft_checkpoints/d18/meta_000478.json",
            sha256: Some("e0acf64f9e195a42d5de9dc284f5d9a814a0b29d7a48e89f5f96ebb891967028"),
            kind: Some(META_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer/tokenizer.pkl",
            sha256: Some("ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2"),
            kind: Some(TIKTOKEN_PICKLE),
            sources: &[],
        },
        StaticResource {
            key: TOKEN_BYTES,
            file: "tokenizer/token_bytes.pt",
            sha256: Some("009ef93d20dd4684497c19c4b7fed29278b53f20022d2dc39264e27b9eefa2a8"),
            kind: Some(PYTORCH),
            sources: &[],
        },
    ],
};

static D18_SFT: StaticPretrained<'static> = StaticPretrained {
    name: "nanochat-d18-sft",
    aliases: &[],
    description: "nitishpandey04's d18 after SFT (step 478): upstream's 2026 model, 702M, chat",
    license: Some("MIT"),
    origin: Some("https://huggingface.co/nitishpandey04/nanochat-d18-sft"),
    prefab: Some("nanochat-d18-2026"),
    maps: &[&D18_SFT_FILES],
};

/// `nanochat-students`' d20 run of upstream's 2025 model.
pub static NANOCHAT_STUDENTS: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "nanochat-students",
    description: "nanochat-students' d20 run of upstream nanochat's 2025 model: base, mid and RL",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/nanochat-students"),
    items: &[&BASE_D20, &MID_D20, &RL_D20],
};

/// `nitishpandey04`'s d18 run of upstream's 2026 model.
pub static NITISHPANDEY04: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "nitishpandey04",
    description: "nitishpandey04's d18 run of upstream nanochat's 2026 model: base and SFT",
    license: Some("MIT"),
    origin: Some("https://huggingface.co/nitishpandey04"),
    items: &[&D18_BASE, &D18_SFT],
};

/// The nanochat checkpoints bunsen knows by name, behind the [`WELL_KNOWN`]
/// provider: `well-known:nanochat-students/rl-d20`,
/// `nanochat-students/rl-d20`, or its alias `nanochat-d20-rl`.
pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
    name: WELL_KNOWN,
    description: "the nanochat checkpoints bunsen knows by name",
    groups: &[&NANOCHAT_STUDENTS, &NITISHPANDEY04],
};

/// The name [`default_nanochat_factory`] loads when a caller names none.
pub const DEFAULT_NANOCHAT_MODEL: &str = "nanochat-students/rl-d20";

/// nanochat's compiled-in providers: the well-known table.
pub fn default_nanochat_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![Arc::new(WELL_KNOWN_TABLE.to_table())]
}

/// nanochat's factory over [`default_nanochat_providers`], building through
/// [`NanoChatConstruct`].
///
/// [`load_bundle`](PretrainedFactory::load_bundle) with `"[provider:]name"`
/// is the whole pathway from a name to a [`NanoChatBundle`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_nanochat_factory() -> BunsenResult<PretrainedFactory<NanoChatConstruct>> {
    PretrainedFactory::new().with_providers(default_nanochat_providers())
}

impl PretrainedFactory<NanoChatConstruct> {
    /// A name to a bundle: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_bundle(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<NanoChatBundle>> {
        Ok(self.load(spec, cache, device)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::gpts::nanochat::{
        pretrained::NANOCHAT_PREFABS,
        weights::{
            NanoChatGeneration,
            NanoChatMeta,
            nanochat_config,
        },
    };

    /// Every file of every row is pinned to one URL, each row's alias finds
    /// it, and each row's meta (committed under `testdata/nanochat/meta/`, at
    /// the row's digest) is the geometry of the prefab the row names.
    #[test]
    fn test_rows_are_pinned_and_match_their_prefabs() {
        let factory = default_nanochat_factory().unwrap();
        assert_eq!(factory.kit(), NANOCHAT_KIT);
        let rows = [
            ("nanochat-students", "base-d20", Some("nanochat-d20-base")),
            ("nanochat-students", "mid-d20", Some("nanochat-d20-mid")),
            ("nanochat-students", "rl-d20", Some("nanochat-d20-rl")),
            ("nitishpandey04", "nanochat-d18-base", None),
            ("nitishpandey04", "nanochat-d18-sft", None),
        ];
        assert_eq!(
            factory.ids(),
            rows.iter()
                .map(|(group, name, _)| format!("well-known:{group}/{name}"))
                .collect::<Vec<_>>()
        );
        for (group, name, alias) in rows {
            let (_, row) = factory.lookup(name).unwrap();
            assert_eq!(row.name, format!("{group}/{name}"));
            if let Some(alias) = alias {
                assert_eq!(factory.lookup(alias).unwrap().1.name, row.name, "{alias}");
            }
            let map = &row.resources;
            map.validate().unwrap();
            for resource in map.resources.values() {
                assert!(resource.is_pinned(), "{name}: {}", resource.key);
                assert_eq!(resource.urls().len(), 1, "{name}: {}", resource.key);
            }
            for key in [CHECKPOINT, META, TOKENIZER, TOKEN_BYTES] {
                assert!(map.get(key).is_some(), "{name}: {key}");
            }

            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(format!("testdata/nanochat/meta/{name}.meta.json"));
            assert_eq!(
                Some(crate::data::cache::sha256_of(&path).unwrap()),
                map.get(META).unwrap().sha256,
                "{name}: the committed meta is the pinned one"
            );
            let meta = NanoChatMeta::from_file(&path).unwrap();
            let generation = match meta.model_config.window_pattern {
                Some(_) => NanoChatGeneration::Upstream2026,
                None => NanoChatGeneration::Upstream2025,
            };
            let prefab = NANOCHAT_PREFABS
                .expect_lookup_prefab(row.prefab.as_deref().unwrap())
                .to_config();
            assert_eq!(
                format!("{:?}", nanochat_config(&meta, generation)),
                format!("{prefab:?}"),
                "{name}"
            );
        }
        assert_eq!(
            factory.lookup(DEFAULT_NANOCHAT_MODEL).unwrap().1.name,
            "nanochat-students/rl-d20"
        );
    }
}
