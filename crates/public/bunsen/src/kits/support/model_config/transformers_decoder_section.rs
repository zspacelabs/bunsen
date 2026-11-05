//! The decoder fields a `transformers` config section shares.

use crate::{
    errors::BunsenResult,
    kits::support::model_config::HfConfigJson,
};

/// What a `transformers` decoder stack's config section says, in the
/// spellings its families use: a whole `config.json` of a causal LM, or the
/// `text_config` / `audio_config` section of a model with one.
///
/// A kit reads its sections through [`read`](Self::read), then checks what
/// its model fixes (`model_type`, the activation, biases, a window, the
/// rotary type) with [`HfConfigJson::expect_eq`] on the same section, which
/// names the field in its error.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformersDecoderSection {
    /// `hidden_size`.
    pub hidden_size: usize,

    /// `num_hidden_layers`.
    pub num_hidden_layers: usize,

    /// `num_attention_heads`.
    pub num_attention_heads: usize,

    /// `num_key_value_heads`; `num_attention_heads` when absent, as
    /// `transformers` defaults it.
    pub num_key_value_heads: usize,

    /// `head_dim`; `hidden_size / num_attention_heads` when absent, as
    /// `transformers` defaults it.
    pub head_dim: usize,

    /// `intermediate_size`.
    pub intermediate_size: usize,

    /// `rms_norm_eps`.
    pub rms_norm_eps: f64,

    /// The rotary base: `rope_parameters.rope_theta`, else `rope_theta`.
    pub rope_theta: f64,

    /// The rotary type: `rope_parameters.rope_type` or `rope_scaling`'s
    /// (`rope_type`, or the older `type`), the first of them that is not
    /// `"default"`; `"default"` when none says otherwise.
    pub rope_type: String,

    /// `hidden_act`.
    pub hidden_act: String,

    /// `attention_bias`; `false` when absent.
    pub attention_bias: bool,

    /// The attention window: `sliding_window`, unless `use_sliding_window`
    /// is `false`; `None` when absent.
    pub sliding_window: Option<usize>,

    /// `layer_types`, each layer's attention kind, when the section lists
    /// them.
    pub layer_types: Option<Vec<String>>,
}

impl TransformersDecoderSection {
    /// Reads `section`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
    /// naming the field, when the section lacks a required one or one does
    /// not parse.
    pub fn read(section: &HfConfigJson) -> BunsenResult<Self> {
        let hidden_size = section.usize("hidden_size")?;
        let num_attention_heads = section.usize("num_attention_heads")?;
        let rope = section.section("rope_parameters").ok();
        let rope_theta = rope
            .as_ref()
            .and_then(|rope| rope.f64("rope_theta").ok())
            .map_or_else(|| section.f64("rope_theta"), Ok)?;
        let mut rope_types = Vec::new();
        if let Some(rope) = &rope {
            rope_types.push(rope.str("rope_type").ok().map(str::to_string));
        }
        if let Ok(scaling) = section.section("rope_scaling") {
            rope_types.push(
                scaling
                    .str("rope_type")
                    .or_else(|_| scaling.str("type"))
                    .ok()
                    .map(str::to_string),
            );
        }
        let rope_type = rope_types
            .into_iter()
            .flatten()
            .find(|t| t != "default")
            .unwrap_or_else(|| "default".to_string());
        let sliding_window = if section.bool_or("use_sliding_window", true)? {
            section.usize_opt("sliding_window")?
        } else {
            None
        };
        Ok(Self {
            hidden_size,
            num_hidden_layers: section.usize("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: section
                .usize_opt("num_key_value_heads")?
                .unwrap_or(num_attention_heads),
            head_dim: section
                .usize_opt("head_dim")?
                .unwrap_or(hidden_size / num_attention_heads.max(1)),
            intermediate_size: section.usize("intermediate_size")?,
            rms_norm_eps: section.f64("rms_norm_eps")?,
            rope_theta,
            rope_type,
            hidden_act: section.str("hidden_act")?.to_string(),
            attention_bias: section.bool_or("attention_bias", false)?,
            sliding_window,
            layer_types: section.strs_opt("layer_types")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    fn read(json: &str) -> BunsenResult<TransformersDecoderSection> {
        TransformersDecoderSection::read(&HfConfigJson::parse(json).unwrap())
    }

    const QWEN3: &str = r#"{
        "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000, "rope_scaling": null,
        "hidden_act": "silu", "attention_bias": false,
        "use_sliding_window": false, "sliding_window": null, "max_window_layers": 28
    }"#;

    /// A Qwen3 `config.json`: every field as written, no window.
    #[test]
    fn test_read_qwen3() {
        assert_eq!(
            read(QWEN3).unwrap(),
            TransformersDecoderSection {
                hidden_size: 1024,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                intermediate_size: 3072,
                rms_norm_eps: 1e-6,
                rope_theta: 1e6,
                rope_type: "default".to_string(),
                hidden_act: "silu".to_string(),
                attention_bias: false,
                sliding_window: None,
                layer_types: None,
            }
        );
    }

    /// The newer spellings: the base in `rope_parameters`, a window with no
    /// `use_sliding_window`, `layer_types`; and `transformers`' defaults for
    /// the key/value heads and the head width.
    #[test]
    fn test_read_newer_spellings_and_defaults() {
        let d = read(
            r#"{
            "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 64, "rms_norm_eps": 1e-5, "hidden_act": "silu",
            "rope_parameters": {"rope_theta": 1e6, "rope_type": "default"},
            "sliding_window": 5, "layer_types": ["sliding_attention", "full_attention"]
        }"#,
        )
        .unwrap();
        assert_eq!(d.num_key_value_heads, 4);
        assert_eq!(d.head_dim, 8);
        assert_eq!(d.rope_theta, 1e6);
        assert_eq!(d.rope_type, "default");
        assert_eq!(d.sliding_window, Some(5));
        assert_eq!(
            d.layer_types.as_deref(),
            Some(
                &[
                    "sliding_attention".to_string(),
                    "full_attention".to_string()
                ][..]
            )
        );
    }

    /// A rotary type other than the default, in either section, is the
    /// section's; `use_sliding_window: false` drops a window it states.
    #[test]
    fn test_read_rope_type_and_window_switch() {
        let base = |extra: &str| {
            format!(
                r#"{{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                "intermediate_size": 16, "rms_norm_eps": 1e-6, "hidden_act": "silu",
                "rope_theta": 10000, {extra}}}"#
            )
        };
        let d = read(&base(r#""rope_scaling": {"type": "yarn", "factor": 4.0}"#)).unwrap();
        assert_eq!(d.rope_type, "yarn");
        let d = read(&base(
            r#""rope_parameters": {"rope_type": "default"},
               "rope_scaling": {"rope_type": "linear"}"#,
        ))
        .unwrap();
        assert_eq!(d.rope_type, "linear");
        let d = read(&base(
            r#""use_sliding_window": false, "sliding_window": 4096"#,
        ))
        .unwrap();
        assert_eq!(d.sliding_window, None);
    }

    /// A missing size names its field.
    #[test]
    fn test_read_missing_field() {
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("intermediate_size")
            .assert_err(&read(
                r#"{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                    "rms_norm_eps": 1e-6, "hidden_act": "silu", "rope_theta": 10000}"#,
            ));
    }
}
