//! The config conversion.

use std::path::Path;

use crate::{
    blocks::transformers::models::language_decoder::LmDecoderMeta,
    errors::BunsenResult,
    kits::{
        language::generation::qwen3::interop::transformers::qwen3_config,
        speech::asr::qwen3_asr::{
            Qwen3AsrConfig,
            blocks::{
                Qwen3AsrAudioEncoderConfig,
                Qwen3AsrAudioEncoderMeta,
            },
        },
        support::transformers::interop::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

/// The [`Qwen3AsrConfig`] of a `transformers`
/// `Qwen3ASRForConditionalGeneration` `config.json`, parsed: what an `hf:`
/// repo's geometry is, `Qwen3-ASR-0.6B`'s or `Qwen3-ASR-1.7B`'s.
///
/// The audio tower's and the text model's sizes, the attention windows, the
/// norm epsilon, the rotary base and the audio placeholder come from the
/// file. What bunsen's model fixes is checked: a GELU audio tower with as
/// many key/value heads as query heads, whose output is the text model's
/// width; and a text model as
/// [`qwen3_config`] reads it, with its head tied: the
/// section or the file may say so or say nothing, as Qwen3-ASR ships no head
/// of its own.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Qwen3-ASR is not.
pub fn qwen3_asr_config(c: &TransformersConfigJson) -> BunsenResult<Qwen3AsrConfig> {
    c.expect_str("model_type", "qwen3_asr")?;

    let a = c.section("audio_config")?;
    a.expect_str("activation_function", "gelu")?;
    let heads = a.usize("encoder_attention_heads")?;
    if let Ok(kv) = a.usize("num_key_value_heads") {
        a.expect_eq("num_key_value_heads", kv, heads)?;
    }
    a.expect_eq(
        "scale_embedding",
        a.bool_or("scale_embedding", false)?,
        false,
    )?;
    let audio = Qwen3AsrAudioEncoderConfig::new(
        a.usize("num_mel_bins")?,
        a.usize("d_model")?,
        heads,
        a.usize("encoder_ffn_dim")?,
        a.usize("encoder_layers")?,
        a.usize("downsample_hidden_size")?,
    )
    .with_n_window(a.usize("n_window")?)
    .with_n_window_infer(a.usize("n_window_infer")?);
    if let Ok(positions) = a.usize("max_position_embeddings") {
        a.expect_eq("max_position_embeddings", positions, audio.chunk_out())?;
    }

    let text = qwen3_config(
        &c.section("text_config")?,
        c.bool_or("tie_word_embeddings", true)?,
    )?;
    c.expect_eq("tie_word_embeddings", text.tied_head, true)?;
    a.expect_eq("output_dim", a.usize("output_dim")?, text.d_model())?;

    Ok(Qwen3AsrConfig::new(audio, text).with_audio_token(c.usize("audio_token_id")?))
}

/// [`qwen3_asr_config`] of a `config.json`'s text.
///
/// # Errors
/// As [`config_from_str`] and [`qwen3_asr_config`].
pub fn qwen3_asr_config_from_str(json: &str) -> BunsenResult<Qwen3AsrConfig> {
    config_from_str(json, qwen3_asr_config)
}

/// [`qwen3_asr_config`] of the `config.json` at `path`; every error names the
/// file.
///
/// # Errors
/// As [`config_from_path`] and [`qwen3_asr_config`].
pub fn qwen3_asr_config_from_path(path: &Path) -> BunsenResult<Qwen3AsrConfig> {
    config_from_path(path, qwen3_asr_config)
}
