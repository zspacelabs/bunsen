//! # Constructing a Voxtral Realtime pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    SAFETENSORS,
    TEKKEN_JSON,
    VOCABULARY,
    VOXTRAL_KIT,
};
use crate::{
    data::{
        builds::{
            BuildRef,
            CONFIG,
            ConstructInit,
            GEOMETRY,
        },
        catalog::presets::ResolvedPresetSet,
        resources::{
            LoadedResourceSet,
            ResourceSet,
            SAFETENSORS_INDEX,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::asr::voxtral::{
            VoxtralConstruct,
            VoxtralPolicy,
            VoxtralRealtimeConfig,
            interop::{
                mistral_common::voxtral_token_layout_config,
                transformers::{
                    load_voxtral_realtime,
                    voxtral_realtime_config_from_path,
                },
            },
        },
        support::{
            interop::{
                json_mirror_from_path,
                mistral_common::TekkenMirror,
            },
            model_config::resolve_ref_presets,
        },
    },
};

/// How a Voxtral Realtime pretrained is built: its safetensors checkpoint
/// read into the geometry its `config.json` gives
/// ([`voxtral_realtime_config_from_path`]), or else the prefab its row names
/// (`voxtral-mini-4b-realtime-2602` when it names none), in the device's
/// float dtype; its `tekken.json` read as the vocabulary
/// ([`TekkenVocabulary`](crate::kits::support::tokens::TekkenVocabulary)) and
/// as the stream's layout ([`voxtral_token_layout_config`]: the
/// delay it names, 480 ms for the published model); behind a front end of the
/// model's mel count and the default decoder.
///
/// The Voxtral kit's construct config: its [`ConstructInit`] hook.
/// [`default_voxtral_factory`](super::default_voxtral_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug)]
pub struct VoxtralConstructConfig;

impl ConstructInit for VoxtralConstructConfig {
    type Construct = VoxtralConstruct;

    const KIT: &'static str = VOXTRAL_KIT;
    const RESOLVES_PRESETS: bool = true;

    fn applied_presets(
        construct: &Self::Construct
    ) -> Option<crate::data::catalog::presets::PresetSelection> {
        Some(construct.presets.clone())
    }

    /// The hook, for a safetensors checkpoint (one file or shards) with a
    /// `tekken.json` vocabulary.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        if !matches!(
            checkpoint.format.as_deref(),
            Some(SAFETENSORS | SAFETENSORS_INDEX)
        ) {
            return Err(BunsenError::unsupported(format!(
                "no Voxtral reader for a checkpoint of kind {:?}",
                checkpoint.format
            )));
        }
        let vocab = map.try_get(VOCABULARY)?;
        if vocab.format.as_deref() != Some(TEKKEN_JSON) {
            return Err(BunsenError::unsupported(format!(
                "no Voxtral vocabulary reader for kind {:?}",
                vocab.format
            )));
        }
        Ok(Self)
    }

    fn try_init(
        &self,
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<VoxtralConstruct>> {
        let tekken = loaded.expect(VOCABULARY)?;
        let mirror: TekkenMirror = json_mirror_from_path(tekken)?;
        let at = |e: BunsenError| e.context(tekken.display());
        let vocab = mirror.tekken_vocabulary().map_err(at)?;
        let layout = voxtral_token_layout_config(&mirror)
            .and_then(|c| c.try_init())
            .map_err(at)?;
        drop(mirror);

        let config = match loaded.get(CONFIG) {
            Some(part) => voxtral_realtime_config_from_path(&part.path)?,
            None => model
                .stated(GEOMETRY)?
                .unwrap_or_else(VoxtralRealtimeConfig::voxtral_mini_4b_realtime),
        };
        let voxtral = load_voxtral_realtime(loaded, &config, device, load)?;
        let policy = resolve_ref_presets(
            &VoxtralPolicy::default(),
            model,
            &ResolvedPresetSet::default(),
        )?;
        Ok(Arc::new(
            VoxtralConstruct::new(voxtral, vocab, layout)?
                .with_policy(policy.config.clone(), policy.applied())?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_hook_takes_safetensors_and_tekken() {
        VoxtralConstructConfig::for_resource_set(
            &crate::data::catalog::BuiltinCatalog::get()
                .unwrap()
                .build("mistralai/voxtral-mini-4b-realtime-2602")
                .unwrap()
                .resource_set(),
        )
        .unwrap();
    }

    /// Every row resolves to the default policy, which has no named presets;
    /// the lookup's patch sets its fields, and a box finds no preset.
    #[test]
    fn test_the_policy() {
        use crate::kits::speech::asr::voxtral::{
            VoxtralPolicy,
            pretrained::default_voxtral_factory,
        };

        let factory = default_voxtral_factory().unwrap();
        let row = factory.ids()[0].clone();
        let policy = |spec: &str| factory.resolve_presets(spec, &VoxtralPolicy::default());
        let default = policy(&row).unwrap();
        assert!(default.presets.is_empty());
        assert_eq!(default.config, VoxtralPolicy::default());
        assert_eq!(default.config.delay_ms, None);
        assert_eq!(
            policy(&format!("{row}{{delay_ms: 960}}"))
                .unwrap()
                .config
                .delay_ms,
            Some(960)
        );
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Lookup)
            .message_contains("no preset")
            .assert_err(&policy(&format!("{row}[fast]")));
    }
}
