//! # Parakeet pretrained providers and factory
use alloc::sync::Arc;

use super::ParakeetConstructConfig;
use crate::{
    data::builds::{
        HfProvider,
        PretrainedFactory,
        PretrainedProvider,
    },
    errors::BunsenResult,
};

/// The Parakeet kit's name: its rows' `kit` tag.
pub const PARAKEET_KIT: &str = "parakeet";

/// The key of the checkpoint in a Parakeet resource bundle.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the vocabulary in a Parakeet resource bundle, for a checkpoint
/// that does not carry its own.
pub const VOCABULARY: &str = "vocabulary";

/// The `kind` of a Phonon archive: a zstd tar of a `.fermion` container and
/// its `NeMo` config, which carries the vocabulary.
pub const PHONON_ARCHIVE: &str = "phonon archive";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a `transformers` `tokenizer.json` vocabulary.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// What the parakeet kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const PARAKEET_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `well-known:` rows are its builds.
pub const PARAKEET_CATALOG_CONTRACT: &str = "parakeet-asr";

/// The name [`default_parakeet_factory`] loads when a caller names none.
pub const DEFAULT_PARAKEET_MODEL: &str = "fermion/phonon-2";

/// Parakeet's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tokenizer.json` as the vocabulary).
pub fn default_parakeet_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(crate::data::builds::BuildCatalogProvider::expect_builtin(
            PARAKEET_CATALOG_CONTRACT,
            PARAKEET_KIT,
        )),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(VOCABULARY, "tokenizer.json", TOKENIZER_JSON),
        ),
    ]
}

/// Parakeet's factory over [`default_parakeet_providers`], building through
/// [`ParakeetConstructConfig`].
///
/// [`load_construct`](PretrainedFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`ParakeetConstruct`](crate::kits::speech::asr::parakeet::ParakeetConstruct).
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_parakeet_factory() -> BunsenResult<PretrainedFactory<ParakeetConstructConfig>> {
    PretrainedFactory::new().with_providers(default_parakeet_providers())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is pinned, and answers by its qualified, short and alias
    /// names.
    #[test]
    fn test_rows_are_pinned_and_named() {
        let factory = default_parakeet_factory().unwrap();
        for (_, row) in factory.list() {
            let map = &row.resources;
            map.validate().unwrap();
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.is_pinned(), "{}", map.name);
            assert_eq!(checkpoint.urls().len(), 1, "{}", map.name);
        }
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

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::kits::support::catalog_contracts::testing::contract_lines;
        assert_eq!(
            contract_lines(PARAKEET_CONTRACT),
            [
                format!("parakeet-asr: {CHECKPOINT} [{PHONON_ARCHIVE}, {SAFETENSORS}]"),
                format!("parakeet-asr: {VOCABULARY}? [{TOKENIZER_JSON}]"),
            ]
        );
    }
}
