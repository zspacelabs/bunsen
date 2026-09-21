//! # Whisper resource maps
//!
//! What each `openai` pretrained is made of, as resource maps: one map per
//! checkpoint, one per vocabulary, each a single file under the `openai`
//! namespace. A pretrained row fuses a checkpoint map with the vocabulary
//! map its token layout selects; a path arrives as a one-resource map, and
//! the rule that selects its vocabulary is [`WhisperVocabulary::for_layout`].
//!
//! The checkpoints are `whisper/__init__.py`'s `_MODELS`: each file under
//! upstream's download root, a "trust me" base used in place, and then
//! upstream's digest-addressed URL. The vocabularies are the two `.tiktoken`
//! rank files at the commit that last touched them. Nothing here names a
//! bundle: a bundled copy is a cache directory populated ahead of time, and
//! the cache hits it without knowing.

use crate::{
    data::pretrained::{
        StaticBase,
        StaticResource,
        StaticResourceMap,
    },
    kits::speech::whisper::{
        driver::WhisperSpecialIds,
        pretrained::{
            OPENAI_LOCAL_DIR,
            openai_download_root,
        },
    },
};

/// The key of the checkpoint in a Whisper resource map.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the `.tiktoken` vocabulary in a Whisper resource map.
pub const VOCABULARY: &str = "vocabulary";

/// The namespace every `openai` resource is cached under:
/// `pretrained/whisper/openai/<sha256>/<file>`.
pub const OPENAI_NAMESPACE: &str = "openai";

/// The label on every `openai` checkpoint.
pub const PYTORCH_FP16: &str = "pytorch fp16";

/// The label on a `transformers` checkpoint, `model.safetensors`: what a
/// Hugging Face repo serves, at whatever precision it was saved.
pub const SAFETENSORS: &str = "safetensors";

/// The label on the vocabularies.
pub const TIKTOKEN: &str = "tiktoken";

const OPENAI_MODELS_ORIGIN: &str =
    "https://github.com/openai/whisper/blob/main/whisper/__init__.py";

const OPENAI_ASSETS_ORIGIN: &str = "https://github.com/openai/whisper/tree/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets";

/// Upstream's download root, as a base: `openai-whisper` keeps its
/// checkpoints there under their bare names, and a file found there is used
/// in place. A `PretrainedCacheOptions` override by [`OPENAI_LOCAL_DIR`]
/// points it elsewhere.
pub const UPSTREAM_BASE: StaticBase<'static> = StaticBase::LocalDir {
    name: OPENAI_LOCAL_DIR,
    default: openai_download_root,
};

/// One `openai` checkpoint map: the file under [`UPSTREAM_BASE`], then
/// upstream's URL, which is addressed by the same digest that pins it.
macro_rules! openai_checkpoint {
    ($name:ident, $file:literal, $sha256:literal, $description:literal) => {
        #[doc = concat!("`openai/", $file, "`: ", $description, ".")]
        pub static $name: StaticResourceMap<'static> = StaticResourceMap {
            name: concat!("openai/", $file),
            description: $description,
            license: Some("MIT"),
            origin: Some(OPENAI_MODELS_ORIGIN),
            namespace: OPENAI_NAMESPACE,
            bases: &[
                UPSTREAM_BASE,
                StaticBase::Url(concat!(
                    "https://openaipublic.azureedge.net/main/whisper/models/",
                    $sha256
                )),
            ],
            resources: &[StaticResource {
                key: CHECKPOINT,
                file: $file,
                sha256: Some($sha256),
                kind: Some(PYTORCH_FP16),
                sources: &[],
            }],
        };
    };
}

openai_checkpoint!(
    TINY_EN_CHECKPOINT,
    "tiny.en.pt",
    "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
    "39 M parameters, English-only"
);
openai_checkpoint!(
    TINY_CHECKPOINT,
    "tiny.pt",
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    "39 M parameters, multilingual"
);
openai_checkpoint!(
    BASE_EN_CHECKPOINT,
    "base.en.pt",
    "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead",
    "74 M parameters, English-only"
);
openai_checkpoint!(
    BASE_CHECKPOINT,
    "base.pt",
    "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
    "74 M parameters, multilingual"
);
openai_checkpoint!(
    SMALL_EN_CHECKPOINT,
    "small.en.pt",
    "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872",
    "244 M parameters, English-only"
);
openai_checkpoint!(
    SMALL_CHECKPOINT,
    "small.pt",
    "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    "244 M parameters, multilingual"
);
openai_checkpoint!(
    MEDIUM_EN_CHECKPOINT,
    "medium.en.pt",
    "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f",
    "769 M parameters, English-only"
);
openai_checkpoint!(
    MEDIUM_CHECKPOINT,
    "medium.pt",
    "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
    "769 M parameters, multilingual"
);
openai_checkpoint!(
    LARGE_V1_CHECKPOINT,
    "large-v1.pt",
    "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a",
    "1550 M parameters, multilingual"
);
openai_checkpoint!(
    LARGE_V2_CHECKPOINT,
    "large-v2.pt",
    "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
    "1550 M parameters, multilingual"
);
openai_checkpoint!(
    LARGE_V3_CHECKPOINT,
    "large-v3.pt",
    "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
    "1550 M parameters, multilingual, 128 mels"
);
openai_checkpoint!(
    LARGE_V3_TURBO_CHECKPOINT,
    "large-v3-turbo.pt",
    "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
    "809 M parameters, multilingual, 128 mels, four-layer decoder"
);

/// Every `openai` checkpoint map, in upstream's order.
pub static OPENAI_CHECKPOINTS: &[&StaticResourceMap<'static>] = &[
    &TINY_EN_CHECKPOINT,
    &TINY_CHECKPOINT,
    &BASE_EN_CHECKPOINT,
    &BASE_CHECKPOINT,
    &SMALL_EN_CHECKPOINT,
    &SMALL_CHECKPOINT,
    &MEDIUM_EN_CHECKPOINT,
    &MEDIUM_CHECKPOINT,
    &LARGE_V1_CHECKPOINT,
    &LARGE_V2_CHECKPOINT,
    &LARGE_V3_CHECKPOINT,
    &LARGE_V3_TURBO_CHECKPOINT,
];

/// `multilingual.tiktoken`: the vocabulary of every multilingual checkpoint.
/// Its last line is `= 50256`, base64 of nothing, and that empty token is
/// real; bunsen's parser reads it as such.
pub static MULTILINGUAL_VOCABULARY: StaticResourceMap<'static> = StaticResourceMap {
    name: "openai/multilingual.tiktoken",
    description: "the multilingual vocabulary: GPT-2's ranks plus one, as every multilingual checkpoint numbers them",
    license: Some("MIT"),
    origin: Some(OPENAI_ASSETS_ORIGIN),
    namespace: OPENAI_NAMESPACE,
    bases: &[StaticBase::Url(
        "https://raw.githubusercontent.com/openai/whisper/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets",
    )],
    resources: &[StaticResource {
        key: VOCABULARY,
        file: "multilingual.tiktoken",
        sha256: Some("b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"),
        kind: Some(TIKTOKEN),
        sources: &[],
    }],
};

/// `gpt2.tiktoken`: the vocabulary of the English-only (`*.en`) checkpoints,
/// GPT-2's ranks as they are.
pub static GPT2_VOCABULARY: StaticResourceMap<'static> = StaticResourceMap {
    name: "openai/gpt2.tiktoken",
    description: "the English-only vocabulary: GPT-2's ranks, as the `*.en` checkpoints number them",
    license: Some("MIT"),
    origin: Some(OPENAI_ASSETS_ORIGIN),
    namespace: OPENAI_NAMESPACE,
    bases: &[StaticBase::Url(
        "https://raw.githubusercontent.com/openai/whisper/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets",
    )],
    resources: &[StaticResource {
        key: VOCABULARY,
        file: "gpt2.tiktoken",
        sha256: Some("306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"),
        kind: Some(TIKTOKEN),
        sources: &[],
    }],
};

/// Both vocabulary maps: multilingual first.
pub static OPENAI_VOCABULARIES_MAPS: &[&StaticResourceMap<'static>] =
    &[&MULTILINGUAL_VOCABULARY, &GPT2_VOCABULARY];

/// Which of the two rank files a checkpoint decodes through:
/// `multilingual.tiktoken` for a multilingual layout, `gpt2.tiktoken` for
/// an English-only one.
///
/// [`for_layout`](Self::for_layout) is the rule a path model's vocabulary
/// is derived by, and a named row's is checked against; [`map`](Self::map)
/// is the file, as a resource map. The two files number their tokens
/// differently, and a checkpoint decoded through the wrong one produces
/// text that is wrong without being obviously so, so the rule takes the
/// token layout, which comes from the checkpoint's vocabulary size, and
/// the pairing stays out of the caller's hands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperVocabulary {
    /// `multilingual.tiktoken`: every multilingual checkpoint's.
    Multilingual,

    /// `gpt2.tiktoken`: the English-only (`*.en`) checkpoints'.
    Gpt2,
}

impl WhisperVocabulary {
    /// Both, multilingual first.
    pub const ALL: [Self; 2] = [Self::Multilingual, Self::Gpt2];

    /// The vocabulary a token layout selects.
    pub fn for_layout(ids: &WhisperSpecialIds) -> Self {
        if ids.is_multilingual() {
            Self::Multilingual
        } else {
            Self::Gpt2
        }
    }

    /// The rank file, as a resource map: [`MULTILINGUAL_VOCABULARY`] or
    /// [`GPT2_VOCABULARY`].
    pub fn map(self) -> &'static StaticResourceMap<'static> {
        match self {
            Self::Multilingual => &MULTILINGUAL_VOCABULARY,
            Self::Gpt2 => &GPT2_VOCABULARY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::Source,
        kits::speech::whisper::pretrained::WHISPER_PREFABS,
    };

    fn ids_of(prefab: &str) -> WhisperSpecialIds {
        let cfg = WHISPER_PREFABS.expect_lookup_prefab(prefab).to_config();
        *cfg.token_layout
            .policy_for_vocab(cfg.vocab_size)
            .unwrap()
            .ids()
    }

    /// Every checkpoint map is one pinned, labeled `checkpoint` under the
    /// `openai` namespace, found under upstream's root before upstream's
    /// URL, and that URL is addressed by the digest that pins it.
    #[test]
    fn test_checkpoint_maps_validate_and_pin_their_urls() {
        assert_eq!(OPENAI_CHECKPOINTS.len(), 12);
        let mut names: Vec<&str> = OPENAI_CHECKPOINTS.iter().map(|m| m.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 12, "a map name is repeated");

        for map in OPENAI_CHECKPOINTS {
            let owned = map.to_map();
            owned.validate().unwrap();
            assert_eq!(owned.keys(), [CHECKPOINT], "{}", map.name);
            let r = owned.get(CHECKPOINT).unwrap();
            assert_eq!(r.namespace, OPENAI_NAMESPACE, "{}", map.name);
            assert_eq!(r.kind.as_deref(), Some(PYTORCH_FP16), "{}", map.name);
            let sha256 = r.sha256.as_deref().expect("every checkpoint is pinned");
            assert_eq!(map.name, format!("openai/{}", r.file));
            assert_eq!(r.sources.len(), 2, "{}", map.name);
            assert!(
                matches!(&r.sources[0], Source::LocalDir { name, .. } if name == OPENAI_LOCAL_DIR),
                "{}: upstream's root comes first",
                map.name
            );
            assert_eq!(
                r.urls(),
                vec![format!(
                    "https://openaipublic.azureedge.net/main/whisper/models/{sha256}/{}",
                    r.file
                )],
                "{}",
                map.name
            );
        }
        assert!(openai_download_root().is_some_and(|d| d.ends_with("whisper")));
    }

    /// The vocabulary maps validate, are pinned to the commit, and the
    /// layout picks between them.
    #[test]
    fn test_vocabulary_maps_and_the_rule() {
        for map in OPENAI_VOCABULARIES_MAPS {
            let owned = map.to_map();
            owned.validate().unwrap();
            assert_eq!(owned.keys(), [VOCABULARY], "{}", map.name);
            let r = owned.get(VOCABULARY).unwrap();
            assert_eq!(r.namespace, OPENAI_NAMESPACE);
            assert_eq!(r.kind.as_deref(), Some(TIKTOKEN));
            assert!(r.is_pinned());
            assert_eq!(map.name, format!("openai/{}", r.file));
            let urls = r.urls();
            assert_eq!(urls.len(), 1);
            assert!(
                urls[0].contains("839639a223b92ad61851baae9ad8a695ccb41ce5"),
                "{}",
                urls[0]
            );
            assert!(urls[0].ends_with(&format!("/{}", r.file)), "{}", urls[0]);
        }

        let rule = |prefab: &str| WhisperVocabulary::for_layout(&ids_of(prefab));
        assert_eq!(rule("tiny"), WhisperVocabulary::Multilingual);
        assert_eq!(rule("tiny.en"), WhisperVocabulary::Gpt2);
        assert_eq!(rule("large-v3"), WhisperVocabulary::Multilingual);
        assert_eq!(rule("tiny").map().name, MULTILINGUAL_VOCABULARY.name);
        assert_eq!(rule("tiny.en").map().name, GPT2_VOCABULARY.name);
        assert_eq!(
            WhisperVocabulary::ALL.map(|v| v.map().name),
            OPENAI_VOCABULARIES_MAPS
                .iter()
                .map(|m| m.name)
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    /// The bundle's directory is a cache of exactly these maps: pointed at
    /// it, offline, the cache finds `openai/base` and both vocabularies
    /// cached, at paths under it, and writes nothing.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_the_bundle_is_a_cache_of_these_maps() {
        use crate::{
            data::pretrained::{
                CacheStatus,
                Fuse,
                Provenance,
            },
            kits::speech::whisper::pretrained::{
                WHISPER_KIT,
                testing::offline_cache,
            },
        };
        let root = bunsen_bundled_whisper::cache_dir();
        let cache = offline_cache();

        for map in [&BASE_CHECKPOINT, &MULTILINGUAL_VOCABULARY, &GPT2_VOCABULARY] {
            let owned = map.to_map();
            for (key, r) in &owned.resources {
                assert_eq!(cache.status(WHISPER_KIT, r), CacheStatus::Cached, "{key}");
                let resolved = cache.resolve(WHISPER_KIT, r).unwrap();
                assert_eq!(resolved.provenance, Provenance::Cached, "{key}");
                assert!(
                    resolved.path.starts_with(root),
                    "{}",
                    resolved.path.display()
                );
                assert!(resolved.path.is_file(), "{}", resolved.path.display());
            }
        }

        let base = BASE_CHECKPOINT
            .to_map()
            .fuse(MULTILINGUAL_VOCABULARY.to_map(), Fuse::Strict)
            .unwrap();
        let loaded = cache.load(WHISPER_KIT, &base).unwrap();
        assert_eq!(loaded.keys(), [CHECKPOINT, VOCABULARY]);
        for (_, part) in loaded.iter() {
            assert_eq!(part.provenance, Provenance::Cached);
        }
        assert_eq!(loaded.kind(CHECKPOINT), Some(PYTORCH_FP16));
        assert_eq!(loaded.kind(VOCABULARY), Some(TIKTOKEN));
        assert!(
            !root
                .join("pretrained")
                .join(WHISPER_KIT)
                .join("given")
                .exists(),
            "nothing was written"
        );
    }
}
