//! # Constructing a Qwen3-TTS pretrained

use alloc::{
    collections::BTreeMap,
    format,
    sync::Arc,
};

use burn::tensor::Device;

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
        speech::tts::qwen3_tts::{
            Qwen3TtsConstruct,
            Qwen3TtsTokenizerFiles,
            codec::Qwen3TtsDecodeConfig,
            interop::qwen_tts::{
                Qwen3TtsCodecMirror,
                Qwen3TtsGenerationMirror,
                Qwen3TtsModelTypeMirror,
                Qwen3TtsPromptMirror,
                load_codec_decoder,
                load_model,
                model_config,
            },
        },
        support::{
            interop::{
                json_mirror_from_path,
                transformers::config_from_path,
            },
            text::BunsenLanguageTag,
        },
    },
};

/// How a Qwen3-TTS pretrained is built, through
/// [`interop::qwen_tts`](crate::kits::speech::tts::qwen3_tts::interop::qwen_tts):
/// the model's `config.json` gives the talker's and code predictor's
/// geometries, the prompt's ids and the model's kind; the codec's gives its
/// decoder's; the checkpoints load into them, each in the dtype the load
/// policy resolves its stored dtype to (the model is bf16 in its file, the
/// codec f32); `generation_config.json` gives the release's take; the
/// catalog row's tags give whether it takes an instruction and its
/// languages' tags; the tokenizer's files are handed back unread. Together,
/// a [`Qwen3TtsConstruct`].
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
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<Arc<Qwen3TtsConstruct>> {
        // One parse of `config.json`: the geometry, the prompt's ids, the kind.
        let (model_config, prompt, model_type) = config_from_path(loaded.expect(CONFIG)?, |t| {
            Ok((
                model_config(t)?,
                t.mirror::<Qwen3TtsPromptMirror>()?.prompt_config()?,
                t.mirror::<Qwen3TtsModelTypeMirror>()?,
            ))
        })?;
        let codec: Qwen3TtsCodecMirror = json_mirror_from_path(loaded.expect(CODEC_CONFIG)?)?;
        let codec_config = codec.codec_decoder_config()?;
        let generation: Qwen3TtsGenerationMirror =
            json_mirror_from_path(loaded.expect(GENERATION_CONFIG)?)?;

        // The catalog row's facts, where it states them.
        let build = match model {
            BuildRef::Named { build, .. } => Some(build.as_ref()),
            BuildRef::Given(_) => None,
        };
        let instruction = match build.and_then(|b| b.tag("instruction")) {
            Some(value) => value.as_bool().ok_or_else(|| {
                BunsenError::invalid_resource(format!("the `instruction` tag is {value}"))
            })?,
            None => model_type.takes_instruction()?,
        };
        let mut language_tags = BTreeMap::new();
        if let Some(value) = build.and_then(|b| b.tag("language_keys")) {
            let map = value.as_object().ok_or_else(|| {
                BunsenError::invalid_resource(format!("the `language_keys` tag is {value}"))
            })?;
            for (key, tag) in map {
                let tag = tag.as_str().ok_or_else(|| {
                    BunsenError::invalid_resource(format!("`language_keys.{key}` is {tag}"))
                })?;
                language_tags.insert(key.clone(), BunsenLanguageTag::parse(tag)?);
            }
        }

        Ok(Arc::new(Qwen3TtsConstruct {
            model: load_model(loaded, &model_config, device, policy)?,
            model_config,
            codec: load_codec_decoder(loaded, &codec_config, device, policy)?,
            codec_config,
            decode: Qwen3TtsDecodeConfig::new(),
            prompt,
            take: generation.take_config(),
            sample_rate: codec.output_sample_rate,
            tokenizer: Qwen3TtsTokenizerFiles {
                vocab: loaded.expect(VOCAB)?.to_path_buf(),
                merges: loaded.expect(MERGES)?.to_path_buf(),
                tokenizer_config: loaded.expect(TOKENIZER_CONFIG)?.to_path_buf(),
            },
            kind: model_type.model_kind()?,
            instruction,
            language_tags,
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
