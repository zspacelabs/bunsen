//! # Voxtral Realtime pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::VoxtralConstruct;
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
    kits::speech::voxtral::VoxtralBundle,
};

/// The kit segment of a Voxtral resource's path in the cache:
/// `<cache>/pretrained/voxtral/<namespace>/<sha256>/<file>`.
pub const VOXTRAL_KIT: &str = "voxtral";

/// The key of the checkpoint in a Voxtral resource map.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the vocabulary, `tekken.json`, in a Voxtral resource map.
pub const VOCABULARY: &str = "vocabulary";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a Mistral `tekken.json` vocabulary.
pub const TEKKEN_JSON: &str = "tekken.json";

/// The prefab the published row instantiates.
const PREFAB: &str = "voxtral-mini-4b-realtime-2602";

const MISTRAL_ORIGIN: &str = "https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602";

/// `mistralai/Voxtral-Mini-4B-Realtime-2602`'s checkpoint and vocabulary, at
/// a fixed revision.
pub static VOXTRAL_MINI_4B_REALTIME_2602_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "mistralai/voxtral-mini-4b-realtime-2602",
    description: "Voxtral-Mini-4B-Realtime-2602: model.safetensors (bf16) and tekken.json",
    license: Some("Apache-2.0"),
    origin: Some(MISTRAL_ORIGIN),
    namespace: "mistralai",
    bases: &[StaticBase::Url(
        "https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/resolve/2769294da9567371363522aac9bbcfdd19447add",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model.safetensors",
            sha256: Some("e745e4902df6a4c48f29f2f8dc1f6d0fb4cc73c7156bc45923451a5bcdfcd1d6"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: VOCABULARY,
            file: "tekken.json",
            sha256: Some("8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44"),
            kind: Some(TEKKEN_JSON),
            sources: &[],
        },
    ],
};

static VOXTRAL_MINI_4B_REALTIME_2602: StaticPretrained<'static> = StaticPretrained {
    name: "voxtral-mini-4b-realtime-2602",
    aliases: &["voxtral-mini-4b-realtime"],
    description: "Voxtral-Mini-4B-Realtime-2602: 4.4 B parameters, bf16, 13 languages, streaming",
    license: Some("Apache-2.0"),
    origin: Some(MISTRAL_ORIGIN),
    prefab: Some(PREFAB),
    maps: &[&VOXTRAL_MINI_4B_REALTIME_2602_FILES],
};

/// Mistral's Voxtral Realtime models.
pub static MISTRALAI: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "mistralai",
    description: "Mistral's Voxtral Realtime speech recognizers",
    license: Some("Apache-2.0"),
    origin: Some(MISTRAL_ORIGIN),
    items: &[&VOXTRAL_MINI_4B_REALTIME_2602],
};

/// The Voxtral checkpoints bunsen knows by name, behind the [`WELL_KNOWN`]
/// provider: `well-known:mistralai/voxtral-mini-4b-realtime-2602`,
/// `mistralai/voxtral-mini-4b-realtime-2602`, or `voxtral-mini-4b-realtime`.
pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
    name: WELL_KNOWN,
    description: "the Voxtral Realtime checkpoints bunsen knows by name",
    groups: &[&MISTRALAI],
};

/// The name [`default_voxtral_factory`] loads when a caller names none.
pub const DEFAULT_VOXTRAL_MODEL: &str = "mistralai/voxtral-mini-4b-realtime-2602";

/// Voxtral's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in the published layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tekken.json` as the vocabulary).
pub fn default_voxtral_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(WELL_KNOWN_TABLE.to_table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(VOCABULARY, "tekken.json", TEKKEN_JSON),
        ),
    ]
}

/// Voxtral's factory over [`default_voxtral_providers`], building through
/// [`VoxtralConstruct`].
///
/// [`load_bundle`](PretrainedFactory::load_bundle) with `"[provider:]name"`
/// is the whole pathway from a name to a [`VoxtralBundle`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_voxtral_factory() -> BunsenResult<PretrainedFactory<VoxtralConstruct>> {
    PretrainedFactory::new().with_providers(default_voxtral_providers())
}

impl PretrainedFactory<VoxtralConstruct> {
    /// A name to a bundle: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_bundle(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<VoxtralBundle>> {
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
        let map = VOXTRAL_MINI_4B_REALTIME_2602_FILES.to_map();
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
