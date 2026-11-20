//! The code predictor's config.

use burn::config::Config;

use crate::{
    blocks::transformers::decoders::depth::{
        DepthDecoderConfig,
        DepthDecoderMeta,
    },
    burner::module::ToStructureConfig,
    errors::BunsenResult,
    kits::{
        language::qwen3::Qwen3Config,
        support::interop::transformers::{
            TransformersConfigJson,
            TransformersDecoderSection,
        },
    },
};

/// Config for Qwen3-TTS's code predictor: a Qwen3 stack (QK-norm before the
/// rotary, a `SwiGLU`, RMS norms) predicting a frame's codebooks after the
/// first, each of [`lm`](Self::lm)'s `n_vocab` codes; its tables at the
/// talker's width, and a projection into its own when the widths differ
/// (1.7B's `small_to_mtp_projection`).
///
/// Lowers to the [`DepthDecoderConfig`] that builds the predictor
/// ([`ToStructureConfig`]).
#[derive(Config, Debug)]
pub struct Qwen3TtsCodePredictorConfig {
    /// The stack's geometry; its `n_vocab` is the codes per codebook. Only
    /// the stack is built from it: the tables and heads are the depth
    /// decoder's own.
    pub lm: Qwen3Config,

    /// The talker's width: the rows the predictor is given, and its tables'.
    pub talker_width: usize,

    /// The codebooks it predicts: a frame's after the first.
    pub codebooks: usize,
}

impl Qwen3TtsCodePredictorConfig {
    /// The config of a model `config.json`'s
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
    pub fn from_transformers_section(
        t: &TransformersConfigJson,
        talker_width: usize,
    ) -> BunsenResult<Self> {
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
        Ok(Self::new(lm, talker_width, t.usize("num_code_groups")? - 1))
    }
}

impl DepthDecoderMeta for Qwen3TtsCodePredictorConfig {
    fn input_width(&self) -> usize {
        self.talker_width
    }

    fn width(&self) -> usize {
        self.lm.d_model
    }

    fn codebooks(&self) -> usize {
        self.codebooks
    }

    fn vocab(&self) -> usize {
        self.lm.n_vocab
    }
}

impl ToStructureConfig for Qwen3TtsCodePredictorConfig {
    type Structure = DepthDecoderConfig;

    fn try_to_structure(&self) -> BunsenResult<DepthDecoderConfig> {
        Ok(DepthDecoderConfig::new(
            self.talker_width,
            self.codebooks,
            self.lm.n_vocab,
            self.lm.stack(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::decoders::depth::DepthDecoder,
        burner::module::ModuleInit,
        support::testing::cpu_device,
    };

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
        let config = Qwen3TtsCodePredictorConfig::from_transformers_section(&section, 2048)
            .expect("the release's");
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

    /// The policy, its lowering and the predictor built from either answer
    /// the meta alike, with a projection when the widths differ.
    #[test]
    fn test_lowering_agrees() {
        let device = cpu_device();
        let config =
            Qwen3TtsCodePredictorConfig::new(Qwen3Config::new(11, 8, 2, 2, 1, 4, 16), 12, 3);
        let structure = config.to_structure();
        let direct: DepthDecoder = config.init(&device);
        let lowered: DepthDecoder = structure.init(&device);
        for meta in [
            &config as &dyn DepthDecoderMeta,
            &structure,
            &direct,
            &lowered,
        ] {
            assert_eq!(
                [
                    meta.input_width(),
                    meta.width(),
                    meta.codebooks(),
                    meta.vocab()
                ],
                [12, 8, 3, 11]
            );
        }
        assert_eq!(structure.stack.n_layers, 2);
        assert!(direct.input_proj.is_some() && lowered.input_proj.is_some());
    }
}
