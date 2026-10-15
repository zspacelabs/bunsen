//! # Qwen3-ASR pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::Qwen3AsrConstruct;
use crate::{
    data::pretrained::{
        HfProvider,
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
    kits::speech::qwen3_asr::Qwen3AsrBundle,
};

/// The kit segment of a Qwen3-ASR resource's path in the cache:
/// `<cache>/pretrained/qwen3-asr/<namespace>/<sha256>/<file>`.
pub const QWEN3_ASR_KIT: &str = "qwen3-asr";

/// The key of the checkpoint in a Qwen3-ASR resource map.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the vocabulary in a Qwen3-ASR resource map, for a checkpoint
/// that does not carry its own.
pub const VOCABULARY: &str = "vocabulary";

/// The `kind` of a Phonon-1 archive: a zstd tar of a packed checkpoint, its
/// `packed_manifest.json`, and its vocabulary.
pub const PHONON_ARCHIVE: &str = "phonon archive";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a `transformers` `tokenizer.json` vocabulary.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// The prefab the 0.6B rows instantiate.
const PREFAB: &str = "qwen3-asr-0.6b";

const FERMION_ORIGIN: &str = "https://github.com/fermionresearch/phonon";

const QWEN_ORIGIN: &str = "https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf";

const QWEN_1_7B_ORIGIN: &str = "https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf";

/// One Phonon-1 archive, at a fixed revision of its repo.
macro_rules! phonon_1_archive {
    ($name:literal, $description:literal, $base:literal, $file:literal, $sha256:literal) => {
        StaticResourceMap {
            name: $name,
            description: $description,
            license: Some("Apache-2.0"),
            origin: Some(FERMION_ORIGIN),
            namespace: "fermion",
            bases: &[StaticBase::Url($base)],
            resources: &[StaticResource {
                key: CHECKPOINT,
                file: $file,
                sha256: Some($sha256),
                kind: Some(PHONON_ARCHIVE),
                sources: &[],
            }],
        }
    };
}

/// `FermionResearch/Phonon-1`'s archive.
pub static PHONON_1_ARCHIVE: StaticResourceMap<'static> = phonon_1_archive!(
    "fermion/phonon-audio6.bps.tar.zst",
    "Phonon-1: quint5 text decoder, 6-bit audio tower, 8-bit embedding, zstd tar",
    "https://huggingface.co/FermionResearch/Phonon-1/resolve/68f96e309cd19bf5b9b72c0ead80eed92a499eae",
    "phonon-audio6.bps.tar.zst",
    "214c3b45aa57257013811a53f99a905848466ad4ab21f2b2f8368f7ac79427b2"
);

/// `FermionResearch/Phonon-1-Micro`'s archive.
pub static PHONON_1_MICRO_ARCHIVE: StaticResourceMap<'static> = phonon_1_archive!(
    "fermion/phonon-micro.bps.tar.zst",
    "Phonon-1-Micro: quint5 text decoder, 4-bit audio tower and embedding, zstd tar",
    "https://huggingface.co/FermionResearch/Phonon-1-Micro/resolve/e916a4f8e30ed5c3ddfc1808429f049669bd8bf3",
    "phonon-micro.bps.tar.zst",
    "8a03e55d78e65c10f6ce22aff7e32359ca6a7aaa27973435b9ebc3cdbff13f9c"
);

/// `FermionResearch/Phonon-1-Big`'s archive.
pub static PHONON_1_BIG_ARCHIVE: StaticResourceMap<'static> = phonon_1_archive!(
    "fermion/phonon-parity.bps.tar.zst",
    "Phonon-1-Big: quint5 text decoder, bf16 audio tower and embedding, zstd tar",
    "https://huggingface.co/FermionResearch/Phonon-1-Big/resolve/74ec0333ac8a5ede5e855489c069ef763bc99edb",
    "phonon-parity.bps.tar.zst",
    "5dbd566d022a05bac045d1e84f0bf6188999494070e2a00dc5fd1bac6caffdb9"
);

/// `Qwen/Qwen3-ASR-0.6B-hf`'s checkpoint and tokenizer, at a fixed revision.
pub static QWEN3_ASR_0_6B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-asr-0.6b",
    description: "Qwen3-ASR-0.6B: model.safetensors (bf16) and tokenizer.json",
    license: Some("Apache-2.0"),
    origin: Some(QWEN_ORIGIN),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/resolve/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model.safetensors",
            sha256: Some("d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: VOCABULARY,
            file: "tokenizer.json",
            sha256: Some("fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
    ],
};

static PHONON_1: StaticPretrained<'static> = StaticPretrained {
    name: "phonon-1",
    aliases: &[],
    description: "Phonon-1: Qwen3-ASR-0.6B re-packed, 415 MB, 30 languages",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/FermionResearch/Phonon-1"),
    prefab: Some(PREFAB),
    maps: &[&PHONON_1_ARCHIVE],
};

static PHONON_1_MICRO: StaticPretrained<'static> = StaticPretrained {
    name: "phonon-1-micro",
    aliases: &[],
    description: "Phonon-1-Micro: Qwen3-ASR-0.6B re-packed, 285 MB, 30 languages",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/FermionResearch/Phonon-1-Micro"),
    prefab: Some(PREFAB),
    maps: &[&PHONON_1_MICRO_ARCHIVE],
};

static PHONON_1_BIG: StaticPretrained<'static> = StaticPretrained {
    name: "phonon-1-big",
    aliases: &[],
    description: "Phonon-1-Big: Qwen3-ASR-0.6B re-packed, 581 MB, 30 languages",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/FermionResearch/Phonon-1-Big"),
    prefab: Some(PREFAB),
    maps: &[&PHONON_1_BIG_ARCHIVE],
};

/// `Qwen/Qwen3-ASR-1.7B-hf`'s checkpoint and tokenizer, at a fixed revision.
pub static QWEN3_ASR_1_7B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-asr-1.7b",
    description: "Qwen3-ASR-1.7B: model.safetensors (bf16) and tokenizer.json",
    license: Some("Apache-2.0"),
    origin: Some(QWEN_1_7B_ORIGIN),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/resolve/bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model.safetensors",
            sha256: Some("2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: VOCABULARY,
            file: "tokenizer.json",
            sha256: Some("fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_ASR_1_7B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-asr-1.7b",
    aliases: &[],
    description: "Qwen3-ASR-1.7B: 2.0 B parameters, bf16, 30 languages",
    license: Some("Apache-2.0"),
    origin: Some(QWEN_1_7B_ORIGIN),
    prefab: Some("qwen3-asr-1.7b"),
    maps: &[&QWEN3_ASR_1_7B_FILES],
};

static QWEN3_ASR_0_6B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-asr-0.6b",
    aliases: &[],
    description: "Qwen3-ASR-0.6B: 0.8 B parameters, bf16, 30 languages",
    license: Some("Apache-2.0"),
    origin: Some(QWEN_ORIGIN),
    prefab: Some(PREFAB),
    maps: &[&QWEN3_ASR_0_6B_FILES],
};

/// `FermionResearch`'s Phonon-1 builds.
pub static FERMION: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "fermion",
    description: "FermionResearch's Phonon-1 builds of Qwen3-ASR-0.6B",
    license: Some("Apache-2.0"),
    origin: Some(FERMION_ORIGIN),
    items: &[&PHONON_1, &PHONON_1_MICRO, &PHONON_1_BIG],
};

/// Qwen's Qwen3-ASR models.
pub static QWEN: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "qwen",
    description: "Qwen's Qwen3-ASR speech recognizers, as transformers ships them",
    license: Some("Apache-2.0"),
    origin: Some(QWEN_ORIGIN),
    items: &[&QWEN3_ASR_0_6B, &QWEN3_ASR_1_7B],
};

/// The Qwen3-ASR checkpoints bunsen knows by name, behind the [`WELL_KNOWN`]
/// provider: `well-known:fermion/phonon-1`, `fermion/phonon-1`, or
/// `phonon-1`.
pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
    name: WELL_KNOWN,
    description: "the Qwen3-ASR checkpoints bunsen knows by name",
    groups: &[&FERMION, &QWEN],
};

/// The name [`default_qwen3_asr_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_ASR_MODEL: &str = "fermion/phonon-1";

/// Qwen3-ASR's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tokenizer.json` as the vocabulary).
pub fn default_qwen3_asr_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(WELL_KNOWN_TABLE.to_table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(VOCABULARY, "tokenizer.json", TOKENIZER_JSON),
        ),
    ]
}

/// Qwen3-ASR's factory over [`default_qwen3_asr_providers`], building
/// through [`Qwen3AsrConstruct`].
///
/// [`load_bundle`](PretrainedFactory::load_bundle) with `"[provider:]name"`
/// is the whole pathway from a name to a [`Qwen3AsrBundle`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_asr_factory() -> BunsenResult<PretrainedFactory<Qwen3AsrConstruct>> {
    PretrainedFactory::new().with_providers(default_qwen3_asr_providers())
}

impl PretrainedFactory<Qwen3AsrConstruct> {
    /// A name to a bundle: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_bundle(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<Qwen3AsrBundle>> {
        Ok(self.load(spec, cache, device)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::qwen3_asr::pretrained::QWEN3_ASR_PREFABS;

    /// Every row is pinned, names a prefab there is, and answers by its
    /// qualified and short names.
    #[test]
    fn test_rows_are_pinned_and_named() {
        for map in [
            &PHONON_1_ARCHIVE,
            &PHONON_1_MICRO_ARCHIVE,
            &PHONON_1_BIG_ARCHIVE,
            &QWEN3_ASR_0_6B_FILES,
            &QWEN3_ASR_1_7B_FILES,
        ] {
            let map = map.to_map();
            map.validate().unwrap();
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.is_pinned(), "{}", map.name);
            assert_eq!(checkpoint.urls().len(), 1, "{}", map.name);
        }

        let factory = default_qwen3_asr_factory().unwrap();
        assert_eq!(factory.kit(), QWEN3_ASR_KIT);
        assert_eq!(
            factory.ids(),
            [
                "well-known:fermion/phonon-1",
                "well-known:fermion/phonon-1-micro",
                "well-known:fermion/phonon-1-big",
                "well-known:qwen/qwen3-asr-0.6b",
                "well-known:qwen/qwen3-asr-1.7b",
            ]
        );
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["well-known", "hf"]
        );
        for spec in [
            DEFAULT_QWEN3_ASR_MODEL,
            "phonon-1",
            "well-known:fermion/phonon-1",
        ] {
            assert_eq!(
                factory.lookup(spec).unwrap().1.name,
                "fermion/phonon-1",
                "{spec}"
            );
        }
        for (spec, name) in [
            ("phonon-1-micro", "fermion/phonon-1-micro"),
            ("phonon-1-big", "fermion/phonon-1-big"),
            ("qwen3-asr-0.6b", "qwen/qwen3-asr-0.6b"),
            ("qwen3-asr-1.7b", "qwen/qwen3-asr-1.7b"),
        ] {
            let (_, row) = factory.lookup(spec).unwrap();
            assert_eq!(row.name, name);
            let prefab = row.prefab.as_deref().unwrap();
            QWEN3_ASR_PREFABS.try_lookup_prefab(prefab).unwrap();
        }
    }
}
