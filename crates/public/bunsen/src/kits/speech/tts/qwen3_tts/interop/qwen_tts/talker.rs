//! The talker's `config.json` section and state-dict names.

use alloc::format;

use crate::{
    burner::module::Rekey,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::generation::qwen3::{
            Qwen3Config,
            interop::transformers::text_layer_rekey,
        },
        speech::tts::qwen3_tts::blocks::{
            Qwen3TtsTalkerConfig,
            release_text_projection,
        },
        support::transformers::interop::{
            TransformersConfigJson,
            TransformersDecoderSection,
        },
    },
};

/// The prefix the talker's tensors sit under in a model file.
pub const QWEN3_TTS_TALKER_PREFIX: &str = "talker.";

/// The prefix of the code predictor's tensors, inside the talker's.
pub const QWEN3_TTS_CODE_PREDICTOR_PREFIX: &str = "talker.code_predictor.";

/// The [`Qwen3TtsTalkerConfig`] of a model `config.json`'s `talker_config`
/// section.
///
/// The stack's geometry, the norm epsilon and the rotary base come from
/// the section; what the talker fixes is checked: `model_type`
/// `qwen3_tts_talker`, `SiLU`, no attention bias, no sliding window, the
/// default rotary type. The rotation is the 1-D one, which bunsen's stack
/// applies: a text-to-speech prompt gives the M-RoPE's three position
/// streams the same positions, so every section rotates alike. The
/// sections (`rope_scaling.mrope_section`) must still cover the head's
/// frequencies, `sum == head_dim / 2`, as upstream's split of them
/// needs.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the section lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
/// the field and its value, for a talker this one is not.
pub fn talker_config(t: &TransformersConfigJson) -> BunsenResult<Qwen3TtsTalkerConfig> {
    t.expect_str("model_type", "qwen3_tts_talker")?;
    let d = TransformersDecoderSection::read(t)?;
    t.expect_eq("hidden_act", d.hidden_act.as_str(), "silu")?;
    t.expect_eq("attention_bias", d.attention_bias, false)?;
    t.expect_eq("sliding_window", d.sliding_window, None)?;
    t.expect_eq("rope_scaling.rope_type", d.rope_type.as_str(), "default")?;
    if let Some(rope) = t.section_opt("rope_scaling")?
        && let Some(sections) = rope.usizes("mrope_section").ok()
    {
        let covered: usize = sections.iter().sum();
        if covered != d.head_dim / 2 {
            return Err(BunsenError::unsupported(format!(
                "config.json: `rope_scaling.mrope_section` {sections:?} covers {covered} \
                 frequencies of a head of {}",
                d.head_dim
            )));
        }
    }
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
    .with_rope_theta(d.rope_theta)
    .with_tied_head(false);
    let text_width = t.usize("text_hidden_size")?;
    Ok(Qwen3TtsTalkerConfig::new(
        lm,
        t.usize("text_vocab_size")?,
        text_width,
        release_text_projection(text_width, d.hidden_size),
        t.usize("num_code_groups")?,
    ))
}

/// The talker's names, without [`QWEN3_TTS_TALKER_PREFIX`], onto a
/// [`Qwen3TtsTalker`](crate::kits::speech::tts::qwen3_tts::blocks::Qwen3TtsTalker)'s canonical ones: the decoder's layers by Qwen3's
/// [`text_layer_rekey`], its final norm, the codec table as the decoder's
/// embedding and `codec_head` as its head; the text table as it is, and its
/// projection's `linear_fc1` / `linear_fc2` as its
/// [`BunsenMlp`](crate::blocks::mlp::BunsenMlp)'s `linear1` / `linear2`.
pub fn talker_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount(
            "model.",
            "",
            Rekey::drop_rest()
                .mount_indexed("layers.", "lm.stack.layers.", text_layer_rekey())
                .rename("norm.", "lm.stack.norm.")
                .rename("codec_embedding.", "lm.embedding.")
                .rename("text_embedding.", "text_embedding."),
        )
        .rename("codec_head.", "lm.head.proj.")
        .mount(
            "text_projection.",
            "text_projection.",
            Rekey::keep_rest()
                .rename("linear_fc1.", "linear1.")
                .rename("linear_fc2.", "linear2."),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::testing::assert_rekeys,
        errors::testing::ErrorMatcher,
    };

    fn section(extra: &str) -> TransformersConfigJson {
        TransformersConfigJson::parse(&format!(
            r#"{{
                "model_type": "qwen3_tts_talker", "hidden_act": "silu",
                "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                "rms_norm_eps": 1e-6, "rope_theta": 1000000, "vocab_size": 3072,
                "text_vocab_size": 151936, "text_hidden_size": 2048, "num_code_groups": 16,
                "use_sliding_window": false, "sliding_window": null{extra}
            }}"#
        ))
        .expect("parse")
    }

    /// 0.6B's `talker_config`: the 1-D rotary of an M-RoPE whose sections
    /// cover the head.
    #[test]
    fn test_release_section() {
        let config = talker_config(&section(
            r#", "rope_scaling": {"interleaved": true, "mrope_section": [24, 20, 20],
                 "rope_type": "default", "type": "default"}"#,
        ))
        .expect("the release's section");
        assert_eq!(
            (config.lm.n_vocab, config.lm.d_model, config.lm.head_dim),
            (3072, 1024, 128)
        );
        assert_eq!(
            (config.text_vocab, config.text_width, config.code_groups),
            (151_936, 2048, 16)
        );
        assert!(!config.lm.tied_head);
        assert_eq!(config.lm.rope_theta, 1_000_000.0);
    }

    /// The text projection's names land on its MLP's.
    #[test]
    fn test_text_projection_rekey() {
        let rekey = talker_rekey();
        assert_rekeys(
            &rekey,
            [
                (
                    "text_projection.linear_fc1.weight",
                    "text_projection.linear1.weight",
                ),
                (
                    "text_projection.linear_fc2.bias",
                    "text_projection.linear2.bias",
                ),
                ("model.text_embedding.weight", "text_embedding.weight"),
            ],
        );
    }

    /// M-RoPE sections that do not cover the head's frequencies are refused.
    #[test]
    fn test_partial_mrope_refused() {
        let error = talker_config(&section(
            r#", "rope_scaling": {"mrope_section": [16, 16, 16], "rope_type": "default"}"#,
        ))
        .unwrap_err();
        ErrorMatcher::unsupported()
            .display_contains("mrope_section")
            .assert(&error);
    }
}
