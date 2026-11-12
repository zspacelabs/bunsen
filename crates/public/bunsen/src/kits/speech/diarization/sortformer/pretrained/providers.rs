//! # Sortformer pretrained providers and factory
use alloc::sync::Arc;

use super::SortformerConstructConfig;
use crate::{
    data::builds::{
        HfProvider,
        PretrainedFactory,
        PretrainedProvider,
    },
    errors::BunsenResult,
};

/// The Sortformer kit's name: its rows' `kit` tag.
pub const SORTFORMER_KIT: &str = "sortformer";

/// The key of the checkpoint in a Sortformer resource set.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a `NeMo` `.nemo` checkpoint: its config and its weights in
/// one archive.
pub const NEMO: &str = "nemo";

/// What the sortformer kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const SORTFORMER_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `well-known:` rows are its builds.
pub const SORTFORMER_CATALOG_CONTRACT: &str = "sortformer-diarizer";

/// The name [`default_sortformer_factory`] loads when a caller names none.
pub const DEFAULT_SORTFORMER_MODEL: &str = "nvidia/sortformer/nemotron-3-diarization";

/// Sortformer's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, and its `config.json`), or one
/// holding a `NeMo` `.nemo` alone.
pub fn default_sortformer_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(crate::data::builds::BuildCatalogProvider::expect_builtin(
            SORTFORMER_CATALOG_CONTRACT,
            SORTFORMER_KIT,
        )),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_archive_checkpoint(".nemo", NEMO),
        ),
    ]
}

/// Sortformer's factory over [`default_sortformer_providers`], building
/// through [`SortformerConstructConfig`].
///
/// [`load_construct`](PretrainedFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`SortformerConstruct`](crate::kits::speech::diarization::sortformer::SortformerConstruct).
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_sortformer_factory() -> BunsenResult<PretrainedFactory<SortformerConstructConfig>> {
    PretrainedFactory::new().with_providers(default_sortformer_providers())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is pinned, and answers by its qualified and short names.
    #[test]
    fn test_rows_are_pinned_and_named() {
        for (name, kind) in [
            ("nvidia/nemotron-3-diarization", SAFETENSORS),
            ("nvidia/diar_streaming_sortformer_4spk-v2", NEMO),
            ("nvidia/diar_streaming_sortformer_4spk-v2.1", NEMO),
        ] {
            let map = crate::data::builds::builtin_build_bundle(name).unwrap();
            map.validate().unwrap();
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.is_pinned(), "{name}");
            assert_eq!(checkpoint.urls().len(), 1, "{name}");
            assert!(checkpoint.urls()[0].ends_with(&checkpoint.file));
            assert_eq!(checkpoint.format.as_deref(), Some(kind), "{name}");
        }

        let factory = default_sortformer_factory().unwrap();
        assert_eq!(factory.kit(), SORTFORMER_KIT);
        assert_eq!(
            factory.ids(),
            [
                "well-known:nvidia/sortformer/nemotron-3-diarization",
                "well-known:nvidia/sortformer/diar_streaming_sortformer_4spk-v2",
                "well-known:nvidia/sortformer/diar_streaming_sortformer_4spk-v2.1",
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
        for (spec, name) in [
            (
                DEFAULT_SORTFORMER_MODEL,
                "nvidia/sortformer/nemotron-3-diarization",
            ),
            (
                "nemotron-3-diarization",
                "nvidia/sortformer/nemotron-3-diarization",
            ),
            (
                "well-known:nvidia/nemotron-3-diarization",
                "nvidia/sortformer/nemotron-3-diarization",
            ),
            (
                "nvidia/diar_streaming_sortformer_4spk-v2",
                "nvidia/sortformer/diar_streaming_sortformer_4spk-v2",
            ),
            (
                "sortformer-v2",
                "nvidia/sortformer/diar_streaming_sortformer_4spk-v2",
            ),
            (
                "diar_streaming_sortformer_4spk-v2.1",
                "nvidia/sortformer/diar_streaming_sortformer_4spk-v2.1",
            ),
            (
                "sortformer-v2.1",
                "nvidia/sortformer/diar_streaming_sortformer_4spk-v2.1",
            ),
        ] {
            assert_eq!(factory.lookup(spec).unwrap().1.name, name, "{spec}");
        }
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::kits::support::catalog_contracts::testing::contract_lines;
        assert_eq!(
            contract_lines(SORTFORMER_CONTRACT),
            [format!(
                "sortformer-diarizer: {CHECKPOINT} [{NEMO}, {SAFETENSORS}]"
            )]
        );
    }
}
