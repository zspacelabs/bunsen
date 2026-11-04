//! # Qwen3-ASR pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::Qwen3AsrConstructConfig;
use crate::{
    data::pretrained::{
        EmbeddedCatalog,
        HfProvider,
        PretrainedCache,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::{
        speech::qwen3_asr::Qwen3AsrConstruct,
        support::construct::PretrainedFactory,
    },
};

/// The kit segment of a Qwen3-ASR resource's path in the cache:
/// `<cache>/pretrained/qwen3-asr/<namespace>/<sha256>/<file>`.
pub const QWEN3_ASR_KIT: &str = "qwen3-asr";

/// The key of the checkpoint in a Qwen3-ASR resource bundle.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the vocabulary in a Qwen3-ASR resource bundle, for a checkpoint
/// that does not carry its own.
pub const VOCABULARY: &str = "vocabulary";

/// The `kind` of a Phonon-1 archive: a zstd tar of a packed checkpoint, its
/// `packed_manifest.json`, and its vocabulary.
pub const PHONON_ARCHIVE: &str = "phonon archive";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a `transformers` `tokenizer.json` vocabulary.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// The Qwen3-ASR checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider: in group
/// `fermion`, `FermionResearch`'s Phonon-1, -Micro and -Big archives, and in
/// group `qwen`, Qwen's Qwen3-ASR 0.6B and 1.7B.
///
/// Built from `validation/catalogs/qwen3-asr.yml` by `cargo validate catalog`.
pub static QWEN3_ASR_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("qwen3-asr", include_str!("catalog.json"));

/// The name [`default_qwen3_asr_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_ASR_MODEL: &str = "fermion/phonon-1";

/// Qwen3-ASR's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tokenizer.json` as the vocabulary).
pub fn default_qwen3_asr_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(QWEN3_ASR_CATALOG.table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(VOCABULARY, "tokenizer.json", TOKENIZER_JSON),
        ),
    ]
}

/// Qwen3-ASR's factory over [`default_qwen3_asr_providers`], building
/// through [`Qwen3AsrConstructConfig`].
///
/// [`load_construct`](PretrainedFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`Qwen3AsrConstruct`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_asr_factory() -> BunsenResult<PretrainedFactory<Qwen3AsrConstructConfig>> {
    PretrainedFactory::new().with_providers(default_qwen3_asr_providers())
}

impl PretrainedFactory<Qwen3AsrConstructConfig> {
    /// A name to a construct under `policy`: [`load`](Self::load), keeping
    /// the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_construct(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
        policy: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<Qwen3AsrConstruct>> {
        Ok(self.load(spec, cache, device, policy)?.handle)
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
        let prefabs: Vec<&str> = QWEN3_ASR_PREFABS.iter().map(|p| p.name).collect();
        QWEN3_ASR_CATALOG.catalog().validate(&prefabs).unwrap();
        for bundle in &QWEN3_ASR_CATALOG.catalog().bundles {
            let map = bundle.to_resource_bundle();
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
