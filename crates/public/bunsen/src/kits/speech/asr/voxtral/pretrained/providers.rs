//! # Voxtral Realtime pretrained providers and factory
use alloc::sync::Arc;

use super::VoxtralConstructConfig;
use crate::{
    data::builds::{
        BuildFactory,
        BuildProvider,
        HfProvider,
    },
    errors::BunsenResult,
};

/// The Voxtral kit's name: its rows' `kit` tag.
pub const VOXTRAL_KIT: &str = "voxtral";

/// The key of the checkpoint in a Voxtral resource set.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the vocabulary, `tekken.json`, in a Voxtral resource set.
pub const VOCABULARY: &str = "vocabulary";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a Mistral `tekken.json` vocabulary.
pub const TEKKEN_JSON: &str = "tekken.json";

/// What the voxtral kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const VOXTRAL_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const VOXTRAL_CATALOG_CONTRACT: &str = "voxtral-asr";

/// The name [`default_voxtral_factory`] loads when a caller names none.
pub const DEFAULT_VOXTRAL_MODEL: &str = "mistralai/voxtral/voxtral-mini-4b-realtime-2602";

/// Voxtral's compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a Hugging Face repo in the published layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tekken.json` as the vocabulary).
pub fn default_voxtral_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            VOXTRAL_CATALOG_CONTRACT,
        )),
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
/// [`load_construct`](BuildFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`VoxtralConstruct`](crate::kits::speech::asr::voxtral::VoxtralConstruct).
///
/// # Errors
/// As [`BuildFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_voxtral_factory() -> BunsenResult<BuildFactory<VoxtralConstructConfig>> {
    BuildFactory::new().with_providers(default_voxtral_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::Prefabs,
        kits::speech::asr::voxtral::VoxtralRealtimeConfig,
    };

    /// The row is pinned, states a prefab's geometry, and answers by its
    /// qualified, short and aliased names.
    #[test]
    fn test_rows_are_pinned_and_named() {
        let map = crate::data::catalog::BuiltinCatalog::get()
            .unwrap()
            .build("mistralai/voxtral-mini-4b-realtime-2602")
            .unwrap()
            .resource_set();
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
            ["catalog:mistralai/voxtral/voxtral-mini-4b-realtime-2602"]
        );
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["catalog", "hf"]
        );
        for spec in [
            DEFAULT_VOXTRAL_MODEL,
            "voxtral-mini-4b-realtime-2602",
            "voxtral-mini-4b-realtime",
            "catalog:mistralai/voxtral-mini-4b-realtime-2602",
        ] {
            let (_, row) = factory.lookup(spec).unwrap();
            assert_eq!(row.name, DEFAULT_VOXTRAL_MODEL, "{spec}");
            // Its geometry is stated, reads as a config, and is a prefab's.
            let geometry = &row.args[crate::data::builds::GEOMETRY].value;
            serde_json::from_value::<crate::kits::speech::asr::voxtral::VoxtralRealtimeConfig>(
                geometry.clone(),
            )
            .unwrap();
            assert!(
                VoxtralRealtimeConfig::PREFABS
                    .iter()
                    .any(|(_, prefab)| serde_json::to_value(prefab()).unwrap() == *geometry),
                "{spec}: its geometry is no prefab's"
            );
        }
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::{
            data::builds::CONFIG,
            kits::support::catalog_contracts::testing::contract_lines,
        };
        assert_eq!(
            contract_lines(VOXTRAL_CONTRACT),
            [
                "voxtral-asr: abstract".to_string(),
                format!("voxtral-asr: {CHECKPOINT} [{SAFETENSORS}]"),
                format!("voxtral-asr: {VOCABULARY} [{TEKKEN_JSON}]"),
                format!("voxtral-asr/probed: {CONFIG} [json]"),
                "voxtral-asr/stated: geometry: \"geometry\"".to_string(),
            ]
        );
    }
}
