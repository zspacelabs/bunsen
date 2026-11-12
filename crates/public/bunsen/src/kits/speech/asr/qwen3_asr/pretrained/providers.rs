//! # Qwen3-ASR pretrained providers and factory
use alloc::sync::Arc;

use super::Qwen3AsrConstructConfig;
use crate::{
    data::pretrained::{
        HfProvider,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::support::construct::PretrainedFactory,
};

/// The Qwen3-ASR kit's name: its rows' `kit` tag.
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

/// What the qwen3-asr kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const QWEN3_ASR_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `well-known:` rows are its builds.
pub const QWEN3_ASR_CATALOG_CONTRACT: &str = "qwen3-asr";

/// The name [`default_qwen3_asr_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_ASR_MODEL: &str = "fermion/phonon-1";

/// Qwen3-ASR's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tokenizer.json` as the vocabulary).
pub fn default_qwen3_asr_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(
            crate::data::pretrained::BuildCatalogProvider::expect_builtin(
                QWEN3_ASR_CATALOG_CONTRACT,
                QWEN3_ASR_KIT,
            ),
        ),
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
/// [`Qwen3AsrConstruct`](crate::kits::speech::asr::qwen3_asr::Qwen3AsrConstruct).
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_asr_factory() -> BunsenResult<PretrainedFactory<Qwen3AsrConstructConfig>> {
    PretrainedFactory::new().with_providers(default_qwen3_asr_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::asr::qwen3_asr::pretrained::QWEN3_ASR_PREFABS;

    /// Every row is pinned and answers by its qualified and short names; a
    /// row that states its geometry states a prefab's.
    #[test]
    fn test_rows_are_pinned_and_named() {
        let factory = default_qwen3_asr_factory().unwrap();
        for (_, row) in factory.list() {
            let map = &row.resources;
            map.validate().unwrap();
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.is_pinned(), "{}", map.name);
            assert_eq!(checkpoint.urls().len(), 1, "{}", map.name);
        }
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
            // Its geometry is stated, reads as a config, and is a prefab's.
            let geometry = &row.args[crate::data::pretrained::GEOMETRY];
            serde_json::from_value::<crate::kits::speech::asr::qwen3_asr::Qwen3AsrConfig>(
                geometry.clone(),
            )
            .unwrap();
            assert!(
                QWEN3_ASR_PREFABS
                    .iter()
                    .any(|p| serde_json::to_value(p.to_config()).unwrap() == *geometry),
                "{name}: its geometry is no prefab's"
            );
        }
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::{
            data::pretrained::CONFIG,
            kits::support::catalog_contracts::testing::contract_lines,
        };
        assert_eq!(
            contract_lines(QWEN3_ASR_CONTRACT),
            [
                "qwen3-asr: abstract".to_string(),
                format!("qwen3-asr: {CHECKPOINT} [{PHONON_ARCHIVE}, {SAFETENSORS}]"),
                format!("qwen3-asr: {VOCABULARY}? [{TOKENIZER_JSON}]"),
                format!("qwen3-asr/probed: {CONFIG} [json]"),
                "qwen3-asr/stated: geometry: \"geometry\"".to_string(),
            ]
        );
    }
}
