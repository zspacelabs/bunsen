//! # Sortformer pretrained providers and factory
use alloc::sync::Arc;

use super::SortformerConstructConfig;
use crate::{
    data::pretrained::{
        EmbeddedCatalog,
        HfProvider,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::support::construct::PretrainedFactory,
};

/// The kit segment of a Sortformer resource's path in the cache:
/// `<cache>/pretrained/sortformer/<namespace>/<sha256>/<file>`.
pub const SORTFORMER_KIT: &str = "sortformer";

/// The key of the checkpoint in a Sortformer resource bundle.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a `NeMo` `.nemo` checkpoint: its config and its weights in
/// one archive.
pub const NEMO: &str = "nemo";

/// The Sortformer checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider, in group
/// `nvidia`: Nemotron-3-Diarization, and the streaming Sortformer v2 and v2.1.
///
/// Built from `validation/catalogs/sortformer.yml` by `cargo validate catalog`.
pub static SORTFORMER_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("sortformer", include_str!("catalog.json"));

/// The name [`default_sortformer_factory`] loads when a caller names none.
pub const DEFAULT_SORTFORMER_MODEL: &str = "nvidia/nemotron-3-diarization";

/// Sortformer's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, and its `config.json`), or one
/// holding a `NeMo` `.nemo` alone.
pub fn default_sortformer_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(SORTFORMER_CATALOG.table()),
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
        SORTFORMER_CATALOG.catalog().validate(&[]).unwrap();
        for (name, kind) in [
            ("nvidia/nemotron-3-diarization", SAFETENSORS),
            ("nvidia/diar_streaming_sortformer_4spk-v2", NEMO),
            ("nvidia/diar_streaming_sortformer_4spk-v2.1", NEMO),
        ] {
            let map = SORTFORMER_CATALOG.resource_bundle(name);
            map.validate().unwrap();
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.is_pinned(), "{name}");
            assert_eq!(checkpoint.urls().len(), 1, "{name}");
            assert!(checkpoint.urls()[0].ends_with(&checkpoint.file));
            assert_eq!(checkpoint.kind.as_deref(), Some(kind), "{name}");
        }

        let factory = default_sortformer_factory().unwrap();
        assert_eq!(factory.kit(), SORTFORMER_KIT);
        assert_eq!(
            factory.ids(),
            [
                "well-known:nvidia/nemotron-3-diarization",
                "well-known:nvidia/diar_streaming_sortformer_4spk-v2",
                "well-known:nvidia/diar_streaming_sortformer_4spk-v2.1",
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
            (DEFAULT_SORTFORMER_MODEL, "nvidia/nemotron-3-diarization"),
            ("nemotron-3-diarization", "nvidia/nemotron-3-diarization"),
            (
                "well-known:nvidia/nemotron-3-diarization",
                "nvidia/nemotron-3-diarization",
            ),
            (
                "nvidia/diar_streaming_sortformer_4spk-v2",
                "nvidia/diar_streaming_sortformer_4spk-v2",
            ),
            ("sortformer-v2", "nvidia/diar_streaming_sortformer_4spk-v2"),
            (
                "diar_streaming_sortformer_4spk-v2.1",
                "nvidia/diar_streaming_sortformer_4spk-v2.1",
            ),
            (
                "sortformer-v2.1",
                "nvidia/diar_streaming_sortformer_4spk-v2.1",
            ),
        ] {
            assert_eq!(factory.lookup(spec).unwrap().1.name, name, "{spec}");
        }
    }
}
