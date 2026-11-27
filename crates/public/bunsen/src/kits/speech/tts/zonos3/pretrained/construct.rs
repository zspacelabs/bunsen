//! # Constructing a ZONOS3 pretrained

use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    CONFIG,
    SAFETENSORS,
    TOKENIZER,
    VOICE_INFO,
    VOICES,
    ZONOS3_KIT,
};
use crate::{
    data::{
        builds::{
            BuildRef,
            ConstructInit,
        },
        formats::LoadPolicy,
        resources::{
            LoadedResourceSet,
            ResourceSet,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::tts::zonos3::{
            ZONOS3_SPEAKER_ENCODER,
            Zonos3Construct,
            Zonos3SpeakerPart,
            interop::zonos::{
                Zonos3Mirror,
                load_zonos3_lm_and_codec,
                load_zonos3_speaker_encoder,
                load_zonos3_voices,
            },
        },
        support::interop::json_mirror_from_path,
    },
};

/// How a ZONOS3 pretrained is built: `config.json`'s `lm` and `codec`
/// sections give the geometries; the checkpoint's `lm.*` and `codec.*`
/// tensors load into them, each in the dtype the load policy resolves its
/// stored dtype to (the language model is bf16 in the file, the codec
/// f32); with a `speaker_encoder` section, its `speaker_encoder.*` tensors
/// load into the speaker encoder, in f32 whatever the policy; the stock voices
/// are read, with their descriptions where the row has them; the tokenizer is
/// handed back as a path. Together, a [`Zonos3Construct`].
///
/// The ZONOS3 kit's construct config: its [`ConstructInit`] hook.
/// [`default_zonos3_factory`](super::default_zonos3_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct Zonos3ConstructConfig;

impl ConstructInit for Zonos3ConstructConfig {
    type Construct = Zonos3Construct;

    const KIT: &'static str = ZONOS3_KIT;

    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// resource set without a safetensors checkpoint, a config, a tokenizer
    /// and safetensors voices, or with voice descriptions not in JSON.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        for (key, format) in [
            (CHECKPOINT, Some(SAFETENSORS)),
            (VOICES, Some(SAFETENSORS)),
            (CONFIG, None),
            (TOKENIZER, None),
        ] {
            let Some(resource) = map.get(key) else {
                return Err(BunsenError::unsupported(format!(
                    "a ZONOS3 resource set needs a `{key}`"
                )));
            };
            if let (Some(want), Some(got)) = (format, resource.format.as_deref())
                && want != got
            {
                return Err(BunsenError::unsupported(format!(
                    "no ZONOS3 reader for a `{key}` of kind {got:?}"
                )));
            }
        }
        if let Some(got) = map.get(VOICE_INFO).and_then(|r| r.format.as_deref())
            && got != "json"
        {
            return Err(BunsenError::unsupported(format!(
                "no ZONOS3 reader for a `{VOICE_INFO}` of kind {got:?}"
            )));
        }
        Ok(Self)
    }

    fn try_init(
        &self,
        _model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<Arc<Zonos3Construct>> {
        let config_path = loaded.expect(CONFIG)?;
        let config: Zonos3Mirror = json_mirror_from_path(config_path)?;
        let at = |e: crate::errors::BunsenError| e.context(config_path.display());
        let lm_config = config.lm.lm_config().map_err(at)?;
        let codec_config = config.codec.codec_config().map_err(at)?;
        let (lm, codec) =
            load_zonos3_lm_and_codec(loaded, &lm_config, &codec_config, device, policy)?;
        let speaker = match &config.speaker_encoder {
            Some(section) => {
                let speaker_config = section.speaker_config().map_err(at)?;
                let encoder = load_zonos3_speaker_encoder(loaded, &speaker_config, device)?;
                let checkpoint = match loaded.map.get(CHECKPOINT).and_then(|r| r.sha256.clone()) {
                    Some(pin) => pin,
                    None => crate::data::cache::sha256_of(loaded.expect(CHECKPOINT)?)?,
                };
                Some(Zonos3SpeakerPart {
                    encoder,
                    config: speaker_config,
                    build: format!("{ZONOS3_SPEAKER_ENCODER}@sha256:{checkpoint}"),
                })
            }
            None => None,
        };
        Ok(Arc::new(Zonos3Construct {
            prompt: config.prompt_config(&lm_config),
            conditions: config.lm.conditions().map_err(at)?,
            engine_config: config.generation.engine_config(),
            language_tags: config.generation.language_tags().map_err(at)?,
            lm,
            lm_config,
            codec,
            codec_config,
            sample_rate: config.sampling_rate,
            voices: load_zonos3_voices(loaded)?,
            tokenizer: loaded.expect(TOKENIZER)?.to_path_buf(),
            speaker,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::testing::ErrorMatcher;

    /// The hook takes the catalog row's resource set, and refuses one that
    /// lacks a part.
    #[test]
    fn test_the_hook_follows_the_map() {
        let map = crate::data::catalog::BuiltinCatalog::get()
            .unwrap()
            .build("zonos3-mini")
            .unwrap()
            .resource_set();
        Zonos3ConstructConfig::for_resource_set(&map).unwrap();
        let mut partial = ResourceSet::new("no voices");
        for key in [CHECKPOINT, CONFIG, TOKENIZER] {
            partial.insert(map.get(key).unwrap().clone());
        }
        ErrorMatcher::unsupported().assert_err(&Zonos3ConstructConfig::for_resource_set(&partial));
    }
}
