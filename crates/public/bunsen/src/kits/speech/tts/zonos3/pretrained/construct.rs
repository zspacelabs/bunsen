//! # Constructing a ZONOS3 pretrained

use alloc::sync::Arc;

use burn::tensor::{
    Device,
    FloatDType,
};

use super::{
    CHECKPOINT,
    CONFIG,
    SAFETENSORS_FORMAT,
    TOKENIZER,
    VOICES,
    ZONOS3_KIT,
};
use crate::{
    burner::module::ModuleInit,
    data::{
        builds::{
            BuildRef,
            ConstructInit,
        },
        formats::{
            LoadPolicy,
            SafetensorsCheckpoint,
            stored_float_dtype,
        },
        resources::{
            LoadedResourceSet,
            ResourceSet,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::speech::tts::zonos3::{
        Zonos3Construct,
        Zonos3Json,
        codec::{
            Zonos3Codec,
            Zonos3CodecConfig,
            weights::{
                ZONOS3_CODEC_PREFIX,
                load_zonos3_codec_tensors,
            },
        },
        lm::{
            Zonos3Lm,
            Zonos3LmConfig,
            weights::{
                ZONOS3_LM_PREFIX,
                load_zonos3_lm_tensors,
            },
        },
        prompt::Zonos3PromptConfig,
        voices::zonos3_voices,
    },
};

/// How a ZONOS3 pretrained is built: `config.json`'s `lm` and `codec`
/// sections give the geometries; the checkpoint's `lm.*` and `codec.*`
/// tensors load into them, each in the dtype the load policy resolves its
/// stored dtype to (the language model is bf16 in the file, the codec
/// f32); the stock voices are read; the tokenizer is handed back as a
/// path. Together, a [`Zonos3Construct`].
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
    /// and safetensors voices.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        for (key, format) in [
            (CHECKPOINT, Some(SAFETENSORS_FORMAT)),
            (VOICES, Some(SAFETENSORS_FORMAT)),
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
        let json: Zonos3Json = serde_json::from_str(
            &std::fs::read_to_string(config_path).map_err(sys_at("read", config_path))?,
        )
        .map_err(|e| BunsenError::invalid_resource(format!("{}: {e}", config_path.display())))?;
        let lm_config = Zonos3LmConfig::from_json(&json.lm)?;
        let codec_config = Zonos3CodecConfig::from_json(&json.codec)?;

        let tensors = SafetensorsCheckpoint::single(loaded.expect(CHECKPOINT)?).tensors()?;
        let stored = |prefix: &str| {
            let part: alloc::vec::Vec<_> = tensors
                .iter()
                .filter(|t| t.name.starts_with(prefix))
                .cloned()
                .collect();
            stored_float_dtype(&part).unwrap_or(FloatDType::F32)
        };
        let lm_dtype = policy.resolve(stored(ZONOS3_LM_PREFIX), device, |_| true)?;
        let codec_dtype = policy.resolve(stored(ZONOS3_CODEC_PREFIX), device, |_| true)?;

        let mut lm: Zonos3Lm = lm_config.try_init(device)?;
        load_zonos3_lm_tensors(&mut lm, &lm_config, tensors.clone(), Some(lm_dtype))?;
        let mut codec: Zonos3Codec = codec_config.try_init(device)?;
        load_zonos3_codec_tensors(&mut codec, &codec_config, tensors, Some(codec_dtype))?;

        let voices =
            zonos3_voices(SafetensorsCheckpoint::single(loaded.expect(VOICES)?).tensors()?)?;
        let prompt = Zonos3PromptConfig::new(
            &lm_config,
            json.lm.text_vocab as u32,
            json.generation.silence_frame.clone(),
            json.lm.reference_max_frames,
        );
        Ok(Arc::new(Zonos3Construct {
            lm,
            lm_config,
            codec,
            codec_config,
            prompt,
            generation: json.generation,
            sample_rate: json.sampling_rate,
            voices,
            tokenizer: loaded.expect(TOKENIZER)?.to_path_buf(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(Zonos3ConstructConfig::for_resource_set(&partial).is_err());
    }
}
