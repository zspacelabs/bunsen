//! # Constructing a Qwen3-TTS pretrained

use alloc::{
    format,
    sync::Arc,
};
use std::path::Path;

use burn::tensor::{
    Device,
    FloatDType,
};
use serde::Deserialize;

use super::{
    CHECKPOINT,
    CODEC,
    CODEC_CONFIG,
    CONFIG,
    GENERATION_CONFIG,
    MERGES,
    QWEN3_TTS_KIT,
    SAFETENSORS_FORMAT,
    TOKENIZER_CONFIG,
    VOCAB,
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
    kits::{
        speech::tts::qwen3_tts::{
            Qwen3TtsConstruct,
            Qwen3TtsModel,
            Qwen3TtsModelConfig,
            Qwen3TtsTakeConfig,
            Qwen3TtsTokenizerFiles,
            codec::{
                Qwen3TtsCodecDecoder,
                Qwen3TtsCodecDecoderConfig,
                Qwen3TtsCodecDecoderJson,
                weights::load_qwen3_tts_codec_tensors,
            },
            prompt::{
                Qwen3TtsPromptConfig,
                Qwen3TtsPromptJson,
            },
            sampler::Qwen3TtsGenerationJson,
            weights::load_qwen3_tts_model_tensors,
        },
        support::model_config::TransformersConfigJson,
    },
};

/// The codec's `config.json`, as far as the decoder reads it.
#[derive(Clone, Debug, Deserialize)]
struct CodecJson {
    output_sample_rate: u32,
    decoder_config: Qwen3TtsCodecDecoderJson,
}

/// `path`'s contents.
fn read(path: &Path) -> BunsenResult<String> {
    std::fs::read_to_string(path).map_err(sys_at("read", path))
}

/// `path`'s JSON, as `T`.
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> BunsenResult<T> {
    serde_json::from_str(&read(path)?)
        .map_err(|e| BunsenError::invalid_resource(format!("{}: {e}", path.display())))
}

/// How a Qwen3-TTS pretrained is built: the model's `config.json` gives the
/// talker's and code predictor's geometries and the prompt's ids, the
/// codec's gives its decoder's; the checkpoints load into them, each in the
/// dtype the load policy resolves its stored dtype to (the model is bf16 in
/// its file, the codec f32); `generation_config.json` gives the release's
/// take; the tokenizer's files are handed back unread. Together, a
/// [`Qwen3TtsConstruct`].
///
/// The Qwen3-TTS kit's construct config: its [`ConstructInit`] hook.
/// [`default_qwen3_tts_factory`](super::default_qwen3_tts_factory) attaches
/// it; a caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct Qwen3TtsConstructConfig;

impl ConstructInit for Qwen3TtsConstructConfig {
    type Construct = Qwen3TtsConstruct;

    const KIT: &'static str = QWEN3_TTS_KIT;

    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// resource set without every part, or with a checkpoint that is not
    /// safetensors.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        for (key, format) in [
            (CHECKPOINT, Some(SAFETENSORS_FORMAT)),
            (CODEC, Some(SAFETENSORS_FORMAT)),
            (CONFIG, None),
            (CODEC_CONFIG, None),
            (GENERATION_CONFIG, None),
            (VOCAB, None),
            (MERGES, None),
            (TOKENIZER_CONFIG, None),
        ] {
            let Some(resource) = map.get(key) else {
                return Err(BunsenError::unsupported(format!(
                    "a Qwen3-TTS resource set needs a `{key}`"
                )));
            };
            if let (Some(want), Some(got)) = (format, resource.format.as_deref())
                && want != got
            {
                return Err(BunsenError::unsupported(format!(
                    "no Qwen3-TTS reader for a `{key}` of kind {got:?}"
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
    ) -> BunsenResult<Arc<Qwen3TtsConstruct>> {
        let config_path = loaded.expect(CONFIG)?;
        let config_text = read(config_path)?;
        let model_config = Qwen3TtsModelConfig::from_transformers(
            &TransformersConfigJson::parse(&config_text)
                .map_err(|e| e.context(config_path.display()))?,
        )?;
        let prompt_json: Qwen3TtsPromptJson = serde_json::from_str(&config_text).map_err(|e| {
            BunsenError::invalid_resource(format!("{}: {e}", config_path.display()))
        })?;
        let prompt = Qwen3TtsPromptConfig::from_json(&prompt_json)?;
        let codec_json: CodecJson = read_json(loaded.expect(CODEC_CONFIG)?)?;
        let codec_config = Qwen3TtsCodecDecoderConfig::from_json(&codec_json.decoder_config)?;
        let generation: Qwen3TtsGenerationJson = read_json(loaded.expect(GENERATION_CONFIG)?)?;

        let dtype = |tensors: &[burn::store::burn_pack::Tensor]| {
            policy.resolve(
                stored_float_dtype(tensors).unwrap_or(FloatDType::F32),
                device,
                |_| true,
            )
        };
        let tensors = SafetensorsCheckpoint::single(loaded.expect(CHECKPOINT)?).tensors()?;
        let model_dtype = dtype(&tensors)?;
        let mut model: Qwen3TtsModel = model_config.try_init(device)?;
        load_qwen3_tts_model_tensors(&mut model, tensors, Some(model_dtype))?;

        let tensors = SafetensorsCheckpoint::single(loaded.expect(CODEC)?).tensors()?;
        let codec_dtype = dtype(&tensors)?;
        let mut codec: Qwen3TtsCodecDecoder = codec_config.try_init(device)?;
        load_qwen3_tts_codec_tensors(&mut codec, &codec_config, tensors, Some(codec_dtype))?;

        Ok(Arc::new(Qwen3TtsConstruct {
            model,
            model_config,
            codec,
            codec_config,
            prompt,
            take: Qwen3TtsTakeConfig::from_generation(&generation),
            generation,
            sample_rate: codec_json.output_sample_rate,
            tokenizer: Qwen3TtsTokenizerFiles {
                vocab: loaded.expect(VOCAB)?.to_path_buf(),
                merges: loaded.expect(MERGES)?.to_path_buf(),
                tokenizer_config: loaded.expect(TOKENIZER_CONFIG)?.to_path_buf(),
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hook takes every catalog row's resource set, and refuses one
    /// that lacks a part.
    #[test]
    fn test_the_hook_follows_the_map() {
        let catalog = crate::data::catalog::BuiltinCatalog::get().unwrap();
        for name in [
            "qwen3-tts-12hz-0.6b-customvoice",
            "qwen3-tts-12hz-1.7b-customvoice",
            "qwen3-tts-12hz-1.7b-voicedesign",
        ] {
            let map = catalog.build(name).unwrap().resource_set();
            Qwen3TtsConstructConfig::for_resource_set(&map).unwrap();
        }
        let map = catalog
            .build("qwen3-tts-12hz-0.6b-customvoice")
            .unwrap()
            .resource_set();
        let mut partial = ResourceSet::new("no codec");
        for key in [
            CHECKPOINT,
            CONFIG,
            GENERATION_CONFIG,
            VOCAB,
            MERGES,
            TOKENIZER_CONFIG,
        ] {
            partial.insert(map.get(key).unwrap().clone());
        }
        assert!(Qwen3TtsConstructConfig::for_resource_set(&partial).is_err());
    }
}
