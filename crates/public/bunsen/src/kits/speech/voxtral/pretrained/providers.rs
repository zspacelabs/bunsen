//! # Voxtral Realtime pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::VoxtralConstructConfig;
use crate::{
    data::pretrained::{
        EmbeddedCatalog,
        HfProvider,
        PretrainedCache,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::{
        speech::voxtral::VoxtralConstruct,
        support::construct::PretrainedFactory,
    },
};

/// The kit segment of a Voxtral resource's path in the cache:
/// `<cache>/pretrained/voxtral/<namespace>/<sha256>/<file>`.
pub const VOXTRAL_KIT: &str = "voxtral";

/// The key of the checkpoint in a Voxtral resource bundle.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the vocabulary, `tekken.json`, in a Voxtral resource bundle.
pub const VOCABULARY: &str = "vocabulary";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a Mistral `tekken.json` vocabulary.
pub const TEKKEN_JSON: &str = "tekken.json";

/// The Voxtral Realtime checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider, in group
/// `mistralai`: Voxtral-Mini-4B-Realtime-2602.
///
/// Built from `validation/catalogs/voxtral.yml` by `cargo validate catalog`.
pub static VOXTRAL_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("voxtral", include_str!("catalog.json"));

/// The name [`default_voxtral_factory`] loads when a caller names none.
pub const DEFAULT_VOXTRAL_MODEL: &str = "mistralai/voxtral-mini-4b-realtime-2602";

/// Voxtral's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in the published layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tekken.json` as the vocabulary).
pub fn default_voxtral_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(VOXTRAL_CATALOG.table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(VOCABULARY, "tekken.json", TEKKEN_JSON),
        ),
    ]
}

/// Voxtral's factory over [`default_voxtral_providers`], building through
/// [`VoxtralConstructConfig`].
///
/// [`load_construct`](PretrainedFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`VoxtralConstruct`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_voxtral_factory() -> BunsenResult<PretrainedFactory<VoxtralConstructConfig>> {
    PretrainedFactory::new().with_providers(default_voxtral_providers())
}

impl PretrainedFactory<VoxtralConstructConfig> {
    /// A name to a construct: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_construct(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<VoxtralConstruct>> {
        Ok(self.load(spec, cache, device)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::voxtral::pretrained::VOXTRAL_PREFABS;

    /// The row is pinned, names a prefab there is, and answers by its
    /// qualified, short and aliased names.
    #[test]
    fn test_rows_are_pinned_and_named() {
        let prefabs: Vec<&str> = VOXTRAL_PREFABS.iter().map(|p| p.name).collect();
        VOXTRAL_CATALOG.catalog().validate(&prefabs).unwrap();
        let map = VOXTRAL_CATALOG.resource_bundle("mistralai/voxtral-mini-4b-realtime-2602");
        map.validate().unwrap();
        for key in [CHECKPOINT, VOCABULARY] {
            let resource = map.get(key).unwrap();
            assert!(resource.is_pinned(), "{key}");
            assert_eq!(resource.urls().len(), 1, "{key}");
        }

        let factory = default_voxtral_factory().unwrap();
        assert_eq!(factory.kit(), VOXTRAL_KIT);
        assert_eq!(
            factory.ids(),
            ["well-known:mistralai/voxtral-mini-4b-realtime-2602"]
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
            DEFAULT_VOXTRAL_MODEL,
            "voxtral-mini-4b-realtime-2602",
            "voxtral-mini-4b-realtime",
            "well-known:mistralai/voxtral-mini-4b-realtime-2602",
        ] {
            let (_, row) = factory.lookup(spec).unwrap();
            assert_eq!(row.name, DEFAULT_VOXTRAL_MODEL, "{spec}");
            VOXTRAL_PREFABS
                .try_lookup_prefab(row.prefab.as_deref().unwrap())
                .unwrap();
        }
    }
}
