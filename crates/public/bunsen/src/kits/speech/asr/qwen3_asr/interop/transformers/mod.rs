//! # Qwen3-ASR in `transformers`' names
//!
//! A `Qwen3ASRForConditionalGeneration` checkpoint as bunsen's
//! [`Qwen3Asr`](crate::kits::speech::asr::qwen3_asr::Qwen3Asr):
//!
//! - **Config:** [`qwen3_asr_config`] of a parsed `config.json`, and
//!   [`qwen3_asr_config_from_str`] / [`qwen3_asr_config_from_path`].
//! - **Names:** [`qwen3_asr_rekey`]; its text model's layers are Qwen3's
//!   ([`text_layer_rekey`]).
//! - **Loading** (feature `store`): `load_qwen3_asr_tensors`; with `cache`,
//!   `load_qwen3_asr`, a resource set's checkpoint and `tokenizer.json`
//!   vocabulary.

use std::path::Path;

use crate::{
    blocks::{
        conv::ConvSeq2d,
        transformers::models::language_decoder::LmDecoderMeta,
    },
    burner::module::Rekey,
    errors::BunsenResult,
    kits::{
        language::qwen3::interop::transformers::{
            qwen3_config,
            text_layer_rekey,
        },
        speech::asr::qwen3_asr::{
            Qwen3AsrConfig,
            blocks::{
                Qwen3AsrAudioEncoderConfig,
                Qwen3AsrAudioEncoderMeta,
            },
        },
        support::interop::transformers::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// `transformers`' `Qwen3ASRForConditionalGeneration` state-dict names onto
/// a [`Qwen3Asr`](crate::kits::speech::asr::qwen3_asr::Qwen3Asr)'s module
/// paths: under `model.`, the audio tower's 1-based `conv2d{n}` onto its stem's
/// 0-based `ConvSeq2d` blocks ([`ConvSeq2d::one_based_rekey`]), its layers by
/// `transformers`' Whisper layer names
/// ([`whisper_encoder_layer_rekey`](crate::kits::support::interop::transformers::whisper_encoder_layer_rekey)); the text
/// model onto its
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel)
/// (`language_model.lm.`): its layers by [`text_layer_rekey`], its embedding
/// and final norm; and everything else as it is.
/// Anything outside `model.` (the tied `lm_head.weight`) is dropped.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`Rekey::load`] resolves to `gamma` / `beta`.
pub fn qwen3_asr_rekey() -> Rekey {
    Rekey::drop_rest().mount(
        "model.",
        "",
        Rekey::keep_rest()
            .mount(
                "audio_tower.conv2d",
                "audio_tower.stem.",
                ConvSeq2d::one_based_rekey(),
            )
            .mount_indexed(
                "audio_tower.layers.",
                "audio_tower.layers.",
                crate::kits::support::interop::transformers::whisper_encoder_layer_rekey(),
            )
            .mount_indexed(
                "language_model.layers.",
                "language_model.lm.stack.layers.",
                text_layer_rekey(),
            )
            .rename(
                "language_model.embed_tokens.",
                "language_model.lm.embedding.",
            )
            .rename("language_model.norm.", "language_model.lm.stack.norm."),
    )
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_asr_rekey() {
        let rekey = qwen3_asr_rekey();
        for (hf, bunsen) in [
            (
                "model.audio_tower.conv2d1.weight",
                Some("audio_tower.stem.blocks.0.conv.weight"),
            ),
            (
                "model.audio_tower.conv2d3.bias",
                Some("audio_tower.stem.blocks.2.conv.bias"),
            ),
            (
                "model.audio_tower.conv_out.weight",
                Some("audio_tower.conv_out.weight"),
            ),
            (
                "model.audio_tower.layers.3.self_attn.k_proj.bias",
                Some("audio_tower.layers.3.attn.key.bias"),
            ),
            (
                "model.audio_tower.layers.3.self_attn.out_proj.weight",
                Some("audio_tower.layers.3.attn.output.weight"),
            ),
            (
                "model.audio_tower.layers.3.final_layer_norm.weight",
                Some("audio_tower.layers.3.mlp_norm.weight"),
            ),
            (
                "model.audio_tower.layers.3.fc2.bias",
                Some("audio_tower.layers.3.mlp.linear2.bias"),
            ),
            (
                "model.audio_tower.ln_post.weight",
                Some("audio_tower.ln_post.weight"),
            ),
            (
                "model.multi_modal_projector.linear_1.weight",
                Some("multi_modal_projector.linear_1.weight"),
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                Some("language_model.lm.stack.layers.0.attn.c_proj.weight"),
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                Some("language_model.lm.stack.layers.0.attn.q_norm.weight"),
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                Some("language_model.lm.stack.layers.0.mlp.gate_proj.weight"),
            ),
            (
                "model.language_model.norm.weight",
                Some("language_model.lm.stack.norm.weight"),
            ),
            (
                "model.language_model.embed_tokens.weight",
                Some("language_model.lm.embedding.weight"),
            ),
            ("lm_head.weight", None),
        ] {
            assert_eq!(rekey.apply(hf).as_deref(), bunsen, "{hf}");
        }
    }

    /// `Qwen3-ASR-0.6B-hf`'s `config.json` is the built-in geometry;
    /// `Qwen3-ASR-1.7B-hf`'s is the larger model's; a model bunsen's is not
    /// is refused by the field that says so.
    #[test]
    fn test_qwen3_asr_config_from_str() {
        use crate::kits::speech::asr::qwen3_asr::blocks::Qwen3AsrMeta;

        let small = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/qwen3_asr/qwen3-asr-0.6b-hf.config.json"
        ));
        let config = qwen3_asr_config_from_str(small).unwrap();
        assert_eq!(
            format!("{config:?}"),
            format!("{:?}", Qwen3AsrConfig::qwen3_asr_0_6b())
        );

        let large = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/qwen3_asr/qwen3-asr-1.7b-hf.config.json"
        ));
        let config = qwen3_asr_config_from_str(large).unwrap();
        assert_eq!(
            format!("{config:?}"),
            format!("{:?}", Qwen3AsrConfig::qwen3_asr_1_7b())
        );
        assert_eq!(config.d_audio(), 1024);
        assert_eq!(config.audio.n_layers(), 24);
        assert_eq!(config.d_text(), 2048);
        assert_eq!(config.text.d_ff, 6144);
        assert_eq!(config.n_vocab(), 151_936);

        let untied = small.replace(
            "\"tie_word_embeddings\": true",
            "\"tie_word_embeddings\": false",
        );
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("tie_word_embeddings")
            .assert_err(&qwen3_asr_config_from_str(&untied));
    }
}
