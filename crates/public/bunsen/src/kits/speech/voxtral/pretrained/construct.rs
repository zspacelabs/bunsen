//! # Constructing a Voxtral Realtime pretrained
use std::sync::Arc;

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
        Construct,
        LoadedResources,
        PretrainedRef,
        ResourceMap,
        SAFETENSORS_INDEX,
        SafetensorsCheckpoint,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        speech::voxtral::{
            VoxtralBundle,
            VoxtralRealtime,
            VoxtralRealtimeConfig,
            VoxtralTokenLayoutConfig,
            weights::{
                load_hf_tensors,
                voxtral_config_from_hf,
            },
        },
        tokens::TekkenVocabulary,
    },
};

/// How a Voxtral Realtime pretrained is built: its safetensors checkpoint
/// read into the geometry its `config.json` gives
/// ([`voxtral_config_from_hf`]), or else the prefab its row names
/// (`voxtral-mini-4b-realtime-2602` when it names none), in the device's
/// float dtype; its `tekken.json` read as the vocabulary
/// ([`TekkenVocabulary`]) and as the stream's layout
/// ([`VoxtralTokenLayoutConfig::from_tekken_json`]: the delay it names, 480
/// ms for the published model); behind a front end of the model's mel count
/// and the default decoder.
///
/// The [`Construct`] hook for the Voxtral kit.
/// [`default_voxtral_factory`](super::default_voxtral_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug)]
pub struct VoxtralConstruct;

impl Construct for VoxtralConstruct {
    type Built = VoxtralBundle;

    const KIT: &'static str = VOXTRAL_KIT;

    /// The hook, for a safetensors checkpoint (one file or shards) with a
    /// `tekken.json` vocabulary.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
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

    fn construct(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<VoxtralBundle>> {
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
                voxtral_config_from_hf(&json).map_err(|e| e.context(part.path.display()))?
            }
            None => model
                .prefab(&VOXTRAL_PREFABS)
                .map_or_else(VoxtralRealtimeConfig::voxtral_mini_4b_realtime, |p| {
                    p.to_config()
                }),
        };
        let mut voxtral: VoxtralRealtime = config.try_init(device)?;
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        load_hf_tensors(&mut voxtral, tensors)?;
        Ok(Arc::new(VoxtralBundle::new(voxtral, vocab, layout)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::voxtral::pretrained::VOXTRAL_MINI_4B_REALTIME_2602_FILES;

    #[test]
    fn test_the_hook_takes_safetensors_and_tekken() {
        VoxtralConstruct::for_map(&VOXTRAL_MINI_4B_REALTIME_2602_FILES.to_map()).unwrap();
    }
}
