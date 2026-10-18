//! # Parakeet pretrained providers and factory
use alloc::sync::Arc;

use burn::tensor::Device;

use super::ParakeetConstruct;
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
    kits::speech::parakeet::ParakeetBundle,
};

/// The kit segment of a Parakeet resource's path in the cache:
/// `<cache>/pretrained/parakeet/<namespace>/<sha256>/<file>`.
pub const PARAKEET_KIT: &str = "parakeet";

/// The key of the checkpoint in a Parakeet resource map.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the vocabulary in a Parakeet resource map, for a checkpoint
/// that does not carry its own.
pub const VOCABULARY: &str = "vocabulary";

/// The `kind` of a Phonon archive: a zstd tar of a `.fermion` container and
/// its `NeMo` config, which carries the vocabulary.
pub const PHONON_ARCHIVE: &str = "phonon archive";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a `transformers` `tokenizer.json` vocabulary.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

const PHONON_ORIGIN: &str = "https://huggingface.co/FermionResearch/Phonon-2";

const NVIDIA_ORIGIN: &str = "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3";

/// `FermionResearch/Phonon-2`'s archive, at a fixed revision.
pub static PHONON_2_ARCHIVE: StaticResourceMap<'static> = StaticResourceMap {
    name: "fermion/phonon-2.bps.tar.zst",
    description: "Phonon-2: the five-value container and its NeMo config, zstd tar",
    license: Some("CC-BY-4.0"),
    origin: Some(PHONON_ORIGIN),
    namespace: "fermion",
    bases: &[StaticBase::Url(
        "https://huggingface.co/FermionResearch/Phonon-2/resolve/7e153e4054c0a10db6d47d6fff5c062d4c77c154",
    )],
    resources: &[StaticResource {
        key: CHECKPOINT,
        file: "phonon-2.bps.tar.zst",
        sha256: Some("98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e"),
        kind: Some(PHONON_ARCHIVE),
        sources: &[],
    }],
};

/// `nvidia/parakeet-tdt-0.6b-v3`'s `transformers` checkpoint and tokenizer,
/// at a fixed revision.
pub static PARAKEET_TDT_0_6B_V3_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nvidia/parakeet-tdt-0.6b-v3",
    description: "parakeet-tdt-0.6b-v3: model.safetensors (f32) and tokenizer.json",
    license: Some("CC-BY-4.0"),
    origin: Some(NVIDIA_ORIGIN),
    namespace: "nvidia",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model.safetensors",
            sha256: Some("3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: VOCABULARY,
            file: "tokenizer.json",
            sha256: Some("bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
    ],
};

static PHONON_2: StaticPretrained<'static> = StaticPretrained {
    name: "phonon-2",
    aliases: &["phonon"],
    description: "Phonon-2: parakeet-tdt-0.6b-v3 with a five-value encoder, 164 MB, English",
    license: Some("CC-BY-4.0"),
    origin: Some(PHONON_ORIGIN),
    prefab: None,
    maps: &[&PHONON_2_ARCHIVE],
};

static PARAKEET_TDT_0_6B_V3: StaticPretrained<'static> = StaticPretrained {
    name: "parakeet-tdt-0.6b-v3",
    aliases: &[],
    description: "parakeet-tdt-0.6b-v3: 600 M parameters, f32, 25 European languages",
    license: Some("CC-BY-4.0"),
    origin: Some(NVIDIA_ORIGIN),
    prefab: None,
    maps: &[&PARAKEET_TDT_0_6B_V3_FILES],
};

/// `FermionResearch`'s Phonon models.
pub static FERMION: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "fermion",
    description: "FermionResearch's Phonon speech recognizers",
    license: Some("CC-BY-4.0"),
    origin: Some("https://github.com/fermionresearch/phonon"),
    items: &[&PHONON_2],
};

/// NVIDIA's Parakeet TDT models.
pub static NVIDIA: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "nvidia",
    description: "NVIDIA's Parakeet TDT speech recognizers, as transformers ships them",
    license: Some("CC-BY-4.0"),
    origin: Some(NVIDIA_ORIGIN),
    items: &[&PARAKEET_TDT_0_6B_V3],
};

/// The Parakeet checkpoints bunsen knows by name, behind the [`WELL_KNOWN`]
/// provider: `well-known:fermion/phonon-2`, `fermion/phonon-2`, or
/// `phonon-2`.
pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
    name: WELL_KNOWN,
    description: "the Parakeet TDT checkpoints bunsen knows by name",
    groups: &[&FERMION, &NVIDIA],
};

/// The name [`default_parakeet_factory`] loads when a caller names none.
pub const DEFAULT_PARAKEET_MODEL: &str = "fermion/phonon-2";

/// Parakeet's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tokenizer.json` as the vocabulary).
pub fn default_parakeet_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(WELL_KNOWN_TABLE.to_table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(VOCABULARY, "tokenizer.json", TOKENIZER_JSON),
        ),
    ]
}

/// Parakeet's factory over [`default_parakeet_providers`], building through
/// [`ParakeetConstruct`].
///
/// [`load_bundle`](PretrainedFactory::load_bundle) with `"[provider:]name"`
/// is the whole pathway from a name to a [`ParakeetBundle`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_parakeet_factory() -> BunsenResult<PretrainedFactory<ParakeetConstruct>> {
    PretrainedFactory::new().with_providers(default_parakeet_providers())
}

impl PretrainedFactory<ParakeetConstruct> {
    /// A name to a bundle: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_bundle(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<ParakeetBundle>> {
        Ok(self.load(spec, cache, device)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is pinned, and answers by its qualified, short and alias
    /// names.
    #[test]
    fn test_rows_are_pinned_and_named() {
        for map in [&PHONON_2_ARCHIVE, &PARAKEET_TDT_0_6B_V3_FILES] {
            let map = map.to_map();
            map.validate().unwrap();
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.is_pinned(), "{}", map.name);
            assert_eq!(checkpoint.urls().len(), 1, "{}", map.name);
        }

        let factory = default_parakeet_factory().unwrap();
        assert_eq!(factory.kit(), PARAKEET_KIT);
        assert_eq!(
            factory.ids(),
            [
                "well-known:fermion/phonon-2",
                "well-known:nvidia/parakeet-tdt-0.6b-v3"
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
            DEFAULT_PARAKEET_MODEL,
            "phonon-2",
            "phonon",
            "well-known:fermion/phonon-2",
        ] {
            assert_eq!(
                factory.lookup(spec).unwrap().1.name,
                "fermion/phonon-2",
                "{spec}"
            );
        }
        assert_eq!(
            factory.lookup("parakeet-tdt-0.6b-v3").unwrap().1.name,
            "nvidia/parakeet-tdt-0.6b-v3"
        );
    }
}
