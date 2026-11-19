//! The code predictor's config.

use crate::{
    blocks::transformers::decoders::depth::DepthDecoderConfig,
    errors::BunsenResult,
    kits::{
        language::qwen3::Qwen3Config,
        support::model_config::{
            TransformersConfigJson,
            TransformersDecoderSection,
        },
    },
};

/// The [`DepthDecoderConfig`] of a model `config.json`'s
/// `talker_config.code_predictor_config`, for a talker of width
/// `talker_width`: a Qwen3 stack (QK-norm before the rotary, a `SwiGLU`, RMS
/// norms), predicting codebooks `1..num_code_groups`, each of
/// `vocab_size` codes; its tables at the talker's width, and a projection
/// into its own when the widths differ (1.7B's `small_to_mtp_projection`).
///
/// What the predictor fixes is checked: `model_type`
/// `qwen3_tts_talker_code_predictor`, `SiLU`, no attention bias, no sliding
/// window, the default rotary type.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the section lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a predictor this one is not.
pub fn qwen3_tts_code_predictor_config(
    t: &TransformersConfigJson,
    talker_width: usize,
) -> BunsenResult<DepthDecoderConfig> {
    t.expect_str("model_type", "qwen3_tts_talker_code_predictor")?;
    let d = TransformersDecoderSection::read(t)?;
    t.expect_eq("hidden_act", d.hidden_act.as_str(), "silu")?;
    t.expect_eq("attention_bias", d.attention_bias, false)?;
    t.expect_eq("sliding_window", d.sliding_window, None)?;
    t.expect_eq("rope_scaling.rope_type", d.rope_type.as_str(), "default")?;
    let vocab = t.usize("vocab_size")?;
    let stack = Qwen3Config::new(
        vocab,
        d.hidden_size,
        d.num_hidden_layers,
        d.num_attention_heads,
        d.num_key_value_heads,
        d.head_dim,
        d.intermediate_size,
    )
    .with_rms_eps(d.rms_norm_eps)
    .with_rope_theta(d.rope_theta as usize)
    .stack();
    Ok(DepthDecoderConfig::new(
        talker_width,
        t.usize("num_code_groups")? - 1,
        vocab,
        stack,
    ))
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
        let config = qwen3_tts_code_predictor_config(&section, 2048).expect("the release's");
        assert_eq!(
            [
                config.input_width(),
                config.width(),
                config.codebooks(),
                config.vocab()
            ],
            [2048, 1024, 15, 2048]
        );
        assert_eq!(config.stack.n_layers, 5);
    }
}
