//! # Constructing a Voxtral Realtime pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    SAFETENSORS,
    TEKKEN_JSON,
    VOCABULARY,
    VOXTRAL_KIT,
    VOXTRAL_PREFABS,
};
use crate::{
    burner::module::ModuleInit,
    data::pretrained::{
        CONFIG,
        GEOMETRY,
        LoadedResources,
        PretrainedRef,
        ResolvedPresetSet,
        ResourceBundle,
        SAFETENSORS_INDEX,
        SafetensorsCheckpoint,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        speech::asr::voxtral::{
            VoxtralConstruct,
            VoxtralPolicy,
            VoxtralRealtime,
            VoxtralRealtimeConfig,
            VoxtralTokenLayoutConfig,
            weights::{
                load_transformers_tensors_as,
                voxtral_config_from_transformers,
            },
        },
        support::{
            construct::ConstructInit,
            model_config::resolve_ref_presets,
            tokens::TekkenVocabulary,
        },
    },
};

/// How a Voxtral Realtime pretrained is built: its safetensors checkpoint
/// read into the geometry its `config.json` gives
/// ([`voxtral_config_from_transformers`]), or else the prefab its row names
/// (`voxtral-mini-4b-realtime-2602` when it names none), in the device's
/// float dtype; its `tekken.json` read as the vocabulary
/// ([`TekkenVocabulary`]) and as the stream's layout
/// ([`VoxtralTokenLayoutConfig::from_tekken_json`]: the delay it names, 480
/// ms for the published model); behind a front end of the model's mel count
/// and the default decoder.
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

    /// The hook, for a safetensors checkpoint (one file or shards) with a
    /// `tekken.json` vocabulary.
    fn for_resource_bundle(map: &ResourceBundle) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        if !matches!(
            checkpoint.kind.as_deref(),
            Some(SAFETENSORS | SAFETENSORS_INDEX)
        ) {
            return Err(BunsenError::unsupported(format!(
                "no Voxtral reader for a checkpoint of kind {:?}",
                checkpoint.kind
            )));
        }
        let vocab = map.try_get(VOCABULARY)?;
        if vocab.kind.as_deref() != Some(TEKKEN_JSON) {
            return Err(BunsenError::unsupported(format!(
                "no Voxtral vocabulary reader for kind {:?}",
                vocab.kind
            )));
        }
        Ok(Self)
    }

    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        load: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<VoxtralConstruct>> {
        let tekken = loaded.expect(VOCABULARY)?;
        let json = std::fs::read_to_string(tekken).map_err(sys_at("read", tekken))?;
        let vocab =
            TekkenVocabulary::from_tekken_json(&json).map_err(|e| e.context(tekken.display()))?;
        let layout = VoxtralTokenLayoutConfig::from_tekken_json(&json)
            .and_then(|c| c.try_init())
            .map_err(|e| e.context(tekken.display()))?;
        drop(json);

        let config = match loaded.get(CONFIG) {
            Some(part) => {
                let json =
                    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))?;
                voxtral_config_from_transformers(&json)
                    .map_err(|e| e.context(part.path.display()))?
            }
            None => match model.stated(GEOMETRY)? {
                Some(config) => config,
                None => model
                    .prefab(&VOXTRAL_PREFABS)
                    .map_or_else(VoxtralRealtimeConfig::voxtral_mini_4b_realtime, |p| {
                        p.to_config()
                    }),
            },
        };
        let mut voxtral: VoxtralRealtime = config.try_init(device)?;
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        // The stored dtype (bf16), where the policy and the device allow it.
        let stored = crate::data::pretrained::stored_float_dtype(&tensors)
            .unwrap_or(burn::tensor::FloatDType::F32);
        let dtype = load.resolve(stored, device, |_| true)?;
        load_transformers_tensors_as(&mut voxtral, tensors, dtype)?;
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
        VoxtralConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle(
                "mistralai/voxtral-mini-4b-realtime-2602",
            )
            .unwrap(),
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
