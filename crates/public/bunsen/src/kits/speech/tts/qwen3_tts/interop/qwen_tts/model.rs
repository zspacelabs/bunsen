//! The model's `config.json` and `generation_config.json`.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
};
use std::path::Path;

use serde::Deserialize;

use super::{
    code_predictor_config,
    talker_config,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::tts::qwen3_tts::{
            Qwen3TtsModelConfig,
            Qwen3TtsModelKind,
            Qwen3TtsTakeConfig,
            prompt::Qwen3TtsPromptConfig,
            sampler::Qwen3TtsSamplerConfig,
        },
        support::interop::transformers::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

/// The [`Qwen3TtsModelConfig`] of a model `config.json`: its `talker_config`,
/// and that section's `code_predictor_config`.
///
/// # Errors
/// As [`talker_config`] and
/// [`code_predictor_config`].
pub fn model_config(json: &TransformersConfigJson) -> BunsenResult<Qwen3TtsModelConfig> {
    let section = json.section("talker_config")?;
    let talker = talker_config(&section)?;
    let code_predictor = code_predictor_config(
        &section.section("code_predictor_config")?,
        talker.lm.d_model,
    )?;
    Ok(Qwen3TtsModelConfig::new(talker, code_predictor))
}

/// [`model_config`] of a `config.json`'s text.
///
/// # Errors
/// As [`config_from_str`] and [`model_config`].
pub fn model_config_from_str(json: &str) -> BunsenResult<Qwen3TtsModelConfig> {
    config_from_str(json, model_config)
}

/// [`model_config`] of the `config.json` at `path`; every error names the
/// file.
///
/// # Errors
/// As [`config_from_path`] and [`model_config`].
pub fn model_config_from_path(path: &Path) -> BunsenResult<Qwen3TtsModelConfig> {
    config_from_path(path, model_config)
}

/// `config.json`'s model kind and size, under upstream's names.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3TtsModelTypeMirror {
    /// The kind: `custom_voice`, `voice_design` or `base`.
    pub tts_model_type: String,
    /// The size: `0b6`, `1b7`.
    #[serde(default)]
    pub tts_model_size: Option<String>,
}

impl Qwen3TtsModelTypeMirror {
    /// The kind `tts_model_type` names.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the value, for a kind this kit does not know.
    pub fn model_kind(&self) -> BunsenResult<Qwen3TtsModelKind> {
        match self.tts_model_type.as_str() {
            "custom_voice" => Ok(Qwen3TtsModelKind::CustomVoice),
            "voice_design" => Ok(Qwen3TtsModelKind::VoiceDesign),
            "base" => Ok(Qwen3TtsModelKind::Base),
            other => Err(BunsenError::unsupported(format!(
                "a `tts_model_type` of {other:?}"
            ))),
        }
    }

    /// Whether upstream gives the model a style instruction: a `CustomVoice`
    /// model larger than 0.6B.
    ///
    /// # Errors
    /// As [`model_kind`](Self::model_kind).
    pub fn takes_instruction(&self) -> BunsenResult<bool> {
        Ok(self.model_kind()? == Qwen3TtsModelKind::CustomVoice
            && self.tts_model_size.as_deref() != Some("0b6"))
    }
}

/// The ids a model `config.json` gives its prompt, under upstream's names:
/// the top level's text ids, and `talker_config`'s codec ids, languages and
/// speakers.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3TtsPromptMirror {
    /// The text pad.
    pub tts_pad_token_id: u32,
    /// The text BOS.
    pub tts_bos_token_id: u32,
    /// The text EOS.
    pub tts_eos_token_id: u32,
    /// The codec ids, languages and speakers.
    pub talker_config: Qwen3TtsTalkerIdsMirror,
}

impl Qwen3TtsPromptMirror {
    /// The [`Qwen3TtsPromptConfig`] of these ids.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for a speaker whose dialect is neither `false` nor a language the
    /// config lists.
    pub fn prompt_config(&self) -> BunsenResult<Qwen3TtsPromptConfig> {
        let t = &self.talker_config;
        let mut speaker_languages = BTreeMap::new();
        for (speaker, dialect) in &t.spk_is_dialect {
            match dialect {
                serde_json::Value::Bool(false) => {}
                serde_json::Value::String(key) if t.codec_language_id.contains_key(key) => {
                    speaker_languages.insert(speaker.clone(), key.clone());
                }
                other => {
                    return Err(BunsenError::invalid_resource(format!(
                        "config.self: `talker_config.spk_is_dialect.{speaker}` is {other}, \
                         neither false nor a language of `codec_language_id`"
                    )));
                }
            }
        }
        Ok(Qwen3TtsPromptConfig::new(
            self.tts_pad_token_id,
            self.tts_bos_token_id,
            self.tts_eos_token_id,
            t.codec_pad_id,
            t.codec_bos_id,
            t.codec_eos_token_id,
            t.codec_think_id,
            t.codec_nothink_id,
            t.codec_think_bos_id,
            t.codec_think_eos_id,
            t.codec_language_id.clone(),
            t.spk_id.clone(),
            speaker_languages,
        ))
    }
}

/// `talker_config`'s codec ids, languages and speakers.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3TtsTalkerIdsMirror {
    /// The codec pad.
    pub codec_pad_id: u32,
    /// The codec BOS.
    pub codec_bos_id: u32,
    /// The codec EOS: the first codebook's end of the take.
    pub codec_eos_token_id: u32,
    /// The think tag.
    pub codec_think_id: u32,
    /// The no-think tag.
    pub codec_nothink_id: u32,
    /// The think block's open.
    pub codec_think_bos_id: u32,
    /// The think block's close.
    pub codec_think_eos_id: u32,
    /// Each language's codec row, by the model's key.
    #[serde(default)]
    pub codec_language_id: BTreeMap<String, u32>,
    /// Each stock speaker's codec row.
    #[serde(default)]
    pub spk_id: BTreeMap<String, u32>,
    /// Each stock speaker's dialect, a language key, or `false`.
    #[serde(default)]
    pub spk_is_dialect: BTreeMap<String, serde_json::Value>,
}

/// A model's `generation_config.json`, under upstream's names: the talker's
/// settings, and the code predictor's (`subtalker_*`).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3TtsGenerationMirror {
    /// Whether the talker samples; else it takes the argmax.
    pub do_sample: bool,
    /// The talker's repetition penalty.
    pub repetition_penalty: f64,
    /// The talker's temperature.
    pub temperature: f64,
    /// The talker's top-p.
    pub top_p: f64,
    /// The talker's top-k.
    pub top_k: usize,
    /// Whether the code predictor samples.
    pub subtalker_dosample: bool,
    /// Its temperature.
    pub subtalker_temperature: f64,
    /// Its top-p.
    pub subtalker_top_p: f64,
    /// Its top-k.
    pub subtalker_top_k: usize,
    /// The most codebook-0 tokens a take generates.
    pub max_new_tokens: usize,
}

impl Qwen3TtsGenerationMirror {
    /// The release's take
    /// ([`Qwen3TtsModel::generate_codes`](crate::kits::speech::tts::qwen3_tts::Qwen3TtsModel::generate_codes)):
    /// the talker's and the code predictor's samplers, and `max_new_tokens`
    /// of codebook-0 tokens, one frame fewer.
    pub fn take_config(&self) -> Qwen3TtsTakeConfig {
        Qwen3TtsTakeConfig::new()
            .with_talker(self.talker_sampler_config())
            .with_code_predictor(self.code_predictor_sampler_config())
            .with_max_frames(self.max_new_tokens.saturating_sub(1))
    }

    /// The talker's sampler settings.
    pub fn talker_sampler_config(&self) -> Qwen3TtsSamplerConfig {
        Qwen3TtsSamplerConfig::new()
            .with_sample(self.do_sample)
            .with_temperature(self.temperature)
            .with_top_k(self.top_k)
            .with_top_p(self.top_p)
            .with_repetition_penalty(self.repetition_penalty)
    }

    /// The code predictor's sampler settings: its own sampling, no
    /// repetition penalty (upstream passes it none).
    pub fn code_predictor_sampler_config(&self) -> Qwen3TtsSamplerConfig {
        Qwen3TtsSamplerConfig::new()
            .with_sample(self.subtalker_dosample)
            .with_temperature(self.subtalker_temperature)
            .with_top_k(self.subtalker_top_k)
            .with_top_p(self.subtalker_top_p)
            .with_repetition_penalty(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::BunsenErrorKind;

    fn release() -> Qwen3TtsGenerationMirror {
        serde_json::from_str(
            r#"{"do_sample": true, "repetition_penalty": 1.05, "temperature": 0.9,
                "top_p": 1.0, "top_k": 50, "subtalker_dosample": true,
                "subtalker_temperature": 0.9, "subtalker_top_p": 1.0,
                "subtalker_top_k": 50, "max_new_tokens": 8192}"#,
        )
        .expect("parse")
    }

    /// The release's settings: the config's defaults for the talker; the
    /// predictor's without a penalty.
    #[test]
    fn test_release_settings() {
        assert_eq!(release().talker_sampler_config().repetition_penalty, 1.05);
        assert_eq!(
            release().talker_sampler_config(),
            Qwen3TtsSamplerConfig::new().with_repetition_penalty(1.05)
        );
        let predictor = release().code_predictor_sampler_config();
        assert_eq!(
            (
                predictor.temperature,
                predictor.top_k,
                predictor.repetition_penalty
            ),
            (0.9, 50, 1.0)
        );
    }

    #[test]
    fn test_bad_dialect_refused() {
        let mut json: Qwen3TtsPromptMirror = serde_json::from_str(
            r#"{"tts_pad_token_id": 1, "tts_bos_token_id": 2, "tts_eos_token_id": 3,
                "talker_config": {"codec_pad_id": 4, "codec_bos_id": 5,
                "codec_eos_token_id": 6, "codec_think_id": 7, "codec_nothink_id": 8,
                "codec_think_bos_id": 9, "codec_think_eos_id": 10}}"#,
        )
        .expect("parse");
        json.talker_config
            .spk_is_dialect
            .insert("eric".into(), serde_json::Value::String("martian".into()));
        assert_eq!(
            json.prompt_config().unwrap_err().kind(),
            BunsenErrorKind::InvalidResource
        );
    }

    /// The kinds upstream names, and an unknown one refused.
    #[test]
    fn test_model_kind() {
        let mirror = |kind: &str, size: Option<&str>| Qwen3TtsModelTypeMirror {
            tts_model_type: kind.into(),
            tts_model_size: size.map(Into::into),
        };
        assert!(
            mirror("custom_voice", Some("1b7"))
                .takes_instruction()
                .unwrap()
        );
        assert!(
            !mirror("custom_voice", Some("0b6"))
                .takes_instruction()
                .unwrap()
        );
        assert!(
            !mirror("voice_design", Some("1b7"))
                .takes_instruction()
                .unwrap()
        );
        assert_eq!(
            mirror("base", None).model_kind().unwrap(),
            Qwen3TtsModelKind::Base
        );
        assert_eq!(
            mirror("cloned", None).model_kind().unwrap_err().kind(),
            BunsenErrorKind::Unsupported
        );
    }
}
