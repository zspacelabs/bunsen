//! The code predictor's `config.json` section and state-dict names.

#[cfg(doc)]
use super::QWEN3_TTS_CODE_PREDICTOR_PREFIX;
use crate::{
    burner::module::Rekey,
    errors::BunsenResult,
    kits::{
        language::generation::qwen3::{
            Qwen3Config,
            interop::transformers::text_layer_rekey,
        },
        speech::tts::qwen3_tts::blocks::Qwen3TtsCodePredictorConfig,
        support::transformers::interop::{
            TransformersConfigJson,
            TransformersDecoderSection,
        },
    },
};

/// The [`Qwen3TtsCodePredictorConfig`] of a model `config.json`'s
/// `talker_config.code_predictor_config` section, for a talker of width
/// `talker_width`; it predicts codebooks `1..num_code_groups`.
///
/// What the predictor fixes is checked: `model_type`
/// `qwen3_tts_talker_code_predictor`, `SiLU`, no attention bias, no
/// sliding window, the default rotary type.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the section lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
/// the field and its value, for a predictor this one is not.
pub fn code_predictor_config(
    t: &TransformersConfigJson,
    talker_width: usize,
) -> BunsenResult<Qwen3TtsCodePredictorConfig> {
    t.expect_str("model_type", "qwen3_tts_talker_code_predictor")?;
    let d = TransformersDecoderSection::read(t)?;
    t.expect_eq("hidden_act", d.hidden_act.as_str(), "silu")?;
    t.expect_eq("attention_bias", d.attention_bias, false)?;
    t.expect_eq("sliding_window", d.sliding_window, None)?;
    t.expect_eq("rope_scaling.rope_type", d.rope_type.as_str(), "default")?;
    let lm = Qwen3Config::new(
        t.usize("vocab_size")?,
        d.hidden_size,
        d.num_hidden_layers,
        d.num_attention_heads,
        d.num_key_value_heads,
        d.head_dim,
        d.intermediate_size,
    )
    .with_rms_eps(d.rms_norm_eps)
    .with_rope_theta(d.rope_theta);
    Ok(Qwen3TtsCodePredictorConfig::new(
        lm,
        talker_width,
        t.usize("num_code_groups")? - 1,
    ))
}

/// The predictor's names, without
/// [`QWEN3_TTS_CODE_PREDICTOR_PREFIX`], onto a
/// [`DepthDecoder`](crate::blocks::transformers::decoders::depth::DepthDecoder)'
/// s canonical ones: the layers by Qwen3's [`text_layer_rekey`], the final
/// norm, the tables, the heads and the projection.
pub fn code_predictor_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount(
            "model.",
            "",
            Rekey::drop_rest()
                .mount_indexed("layers.", "stack.layers.", text_layer_rekey())
                .rename("norm.", "stack.norm.")
                .rename("codec_embedding.", "tables."),
        )
        .rename("lm_head.", "heads.")
        .rename("small_to_mtp_projection.", "input_proj.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::transformers::decoders::depth::DepthDecoderMeta;

    /// The release's `code_predictor_config` (both sizes'), under 1.7B's
    /// talker: 15 codebooks of 2048 codes, a projection from 2048 to 1024.
    #[test]
    fn test_release_section() {
        let section = TransformersConfigJson::parse(
            r#"{
                "model_type": "qwen3_tts_talker_code_predictor", "hidden_act": "silu",
                "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 5,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                "rms_norm_eps": 1e-6, "rope_theta": 1000000, "rope_scaling": null,
                "vocab_size": 2048, "num_code_groups": 16,
                "use_sliding_window": false, "sliding_window": null
            }"#,
        )
        .expect("parse");
        let config = code_predictor_config(&section, 2048).expect("the release's");
        assert_eq!(
            [
                config.input_width(),
                config.width(),
                config.codebooks(),
                config.vocab()
            ],
            [2048, 1024, 15, 2048]
        );
        assert_eq!(config.lm.n_layers, 5);
    }
}
