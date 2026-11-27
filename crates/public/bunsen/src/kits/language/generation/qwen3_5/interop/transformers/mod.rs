//! # Qwen3.5 in `transformers`' names
//!
//! A dense `transformers` Qwen3.5 checkpoint (a
//! `Qwen3_5ForConditionalGeneration`'s text model, its vision tower and
//! multi-token-prediction head dropped, or a bare `Qwen3_5ForCausalLM`) as
//! bunsen's [`Qwen3_5Model`](crate::kits::language::generation::qwen3_5::Qwen3_5Model):
//!
//! - **Config:** [`qwen3_5_config`] of a parsed `config.json`, and
//!   [`qwen3_5_config_from_str`] / [`qwen3_5_config_from_path`].
//! - **Names:** [`qwen3_5_rekey`], its layers by [`text_layer_rekey`].
//! - **Loading** (feature `store`): `load_qwen3_5_tensors`; with `cache`,
//!   `load_qwen3_5`, a resource set's checkpoint.

use std::path::Path;

use crate::{
    blocks::transformers::{
        attention::csa::AttentionOutputGate,
        decoders::causal::CausalDecoderBlock,
    },
    burner::module::{
        Rekey,
        Select,
        TensorLoadTransform,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::generation::qwen3_5::{
            Qwen3_5Config,
            Qwen3_5LayerKind,
        },
        support::interop::transformers::{
            TransformersConfigJson,
            TransformersDecoderSection,
            config_from_path,
            config_from_str,
            llama_layer_keys,
        },
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// The prefixes a Qwen3.5's text model sits under: a
/// `Qwen3_5ForConditionalGeneration`'s (every published checkpoint), and a
/// `Qwen3_5ForCausalLM`'s, in the order they are tried.
pub const QWEN3_5_TEXT_PREFIXES: [&str; 2] = ["model.language_model.", "model."];

/// The `1 + w` norms, by their paths under a layer of the canonical scheme:
/// the two sublayers' and the attention's QK-norms. The `DeltaNet`'s gated norm
/// (`attn.norm.norm`) applies its gain as stored.
const ZERO_CENTERED_LAYER_NORMS: [&str; 4] = [
    "attn_norm.weight",
    "mlp_norm.weight",
    "attn.q_norm.weight",
    "attn.k_norm.weight",
];

/// `transformers`' Qwen3.5 decoder layer names onto a [`CausalDecoderBlock`]'s,
/// relative to it: Llama's for the norms (`input_layernorm`,
/// `post_attention_layernorm`) and a full layer's `self_attn`, as
/// [`CausalDecoderBlock::layer_rekey`] maps them; a linear layer's
/// `linear_attn` the mixer's fields under their own names, but `A_log`,
/// which is `a_log`, and the gated norm's gain, which is its norm's
/// (`norm.norm.weight`); and the `SwiGLU`'s projections as they are.
pub fn text_layer_rekey() -> Rekey {
    CausalDecoderBlock::layer_rekey(&llama_layer_keys(
        "input_layernorm",
        "post_attention_layernorm",
    ))
    .rename("linear_attn.A_log", "attn.a_log")
    .rename("linear_attn.norm.", "attn.norm.norm.")
    .rename("linear_attn.", "attn.")
}

/// `transformers`' Qwen3.5 state-dict names onto the module paths of a
/// [`Qwen3_5Model`](crate::kits::language::generation::qwen3_5::Qwen3_5Model)'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
/// under either of [`QWEN3_5_TEXT_PREFIXES`], the layers by
/// [`text_layer_rekey`], the embedding (`embed_tokens`, the model's
/// `embedding`) and the final norm; `lm_head.weight`, the head's projection;
/// when `config`'s [`zero_centered_norms`](Qwen3_5Config::zero_centered_norms)
/// says so, every `1 + w` norm's gain, stored as `w`, loaded as `1 + w`
/// ([`TensorLoadTransform::zero_centered_gain`]). Anything else is dropped: the
/// vision tower, and the multi-token-prediction head (`mtp.`).
///
/// A tied model has no `lm_head` to load: a checkpoint's `lm_head.weight`
/// is then unused, as `transformers` ties over it.
pub fn qwen3_5_rekey(config: &Qwen3_5Config) -> Rekey {
    let zero_centered: &[&str] = if config.zero_centered_norms {
        &ZERO_CENTERED_LAYER_NORMS
    } else {
        &[]
    };
    let layer = zero_centered
        .iter()
        .fold(text_layer_rekey(), |rekey, norm| {
            rekey.transform(
                Select::suffix(*norm),
                TensorLoadTransform::zero_centered_gain(),
            )
        });
    let text = Rekey::keep_rest()
        .mount("visual.", "", Rekey::drop_rest())
        .mount_indexed("layers.", "stack.layers.", layer)
        .rename("embed_tokens.", "embedding.")
        .rename("norm.", "stack.norm.");
    let text = if config.zero_centered_norms {
        text.transform(
            Select::suffix("stack.norm.weight"),
            TensorLoadTransform::zero_centered_gain(),
        )
    } else {
        text
    };
    QWEN3_5_TEXT_PREFIXES
        .iter()
        .fold(Rekey::drop_rest(), |rekey, prefix| {
            rekey.mount(*prefix, "", text.clone())
        })
        .exact("lm_head.weight", "head.proj.weight")
}

/// The [`Qwen3_5Config`] of a `transformers` dense Qwen3.5 `config.json`: a
/// `Qwen3_5ForConditionalGeneration`'s (`model_type` `qwen3_5`, read from its
/// `text_config`; its vision tower is not part of bunsen's model), or a bare
/// text model's (`qwen3_5_text`).
///
/// The layer kinds come from `layer_types`, or without them from
/// `full_attention_interval` (4 by default) as `transformers` derives them;
/// the rotary share from `rope_parameters.partial_rotary_factor` (0.25 by
/// default); the head is tied as `tie_word_embeddings` says, in the text
/// section or at the top, and untied when neither does.
///
/// What bunsen's model fixes is checked: `SiLU`, no attention bias, no
/// sliding window, the default rotary embedding, an `mrope_section` (text
/// positions are the same on its three axes, so it reduces to one table) that
/// covers the rotated dimensions, no `mlp_only_layers`, a `mamba_ssm_dtype`
/// (the delta rule's state) of `float32`, and `architectures`, when stated,
/// its model type's class (whose layout [`QWEN3_5_TEXT_PREFIXES`] reads).
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Qwen3.5 is not (a mixture of
/// experts, `qwen3_5_moe`, among them).
pub fn qwen3_5_config(c: &TransformersConfigJson) -> BunsenResult<Qwen3_5Config> {
    let root = c.clone();
    let tie_default = root.bool_or("tie_word_embeddings", false)?;
    let (t, architecture) = match root.str("model_type")? {
        "qwen3_5" => {
            let text = root.section("text_config")?;
            text.expect_str("model_type", "qwen3_5_text")?;
            (text, "Qwen3_5ForConditionalGeneration")
        }
        "qwen3_5_text" => (root.clone(), "Qwen3_5ForCausalLM"),
        other => {
            return Err(BunsenError::unsupported(format!(
                "config.json: `model_type` is {other:?}, not a dense Qwen3.5"
            )));
        }
    };
    // The tensors sit under the prefix the class lays them out at
    // (`QWEN3_5_TEXT_PREFIXES`); a config whose class is not its model type's
    // lays them out otherwise.
    if let Some(architectures) = root.strs_opt("architectures")? {
        root.expect_eq(
            "architectures",
            architectures,
            vec![architecture.to_string()],
        )?;
    }
    // The delta rule's state is f32; a model that keeps it narrower would
    // decode otherwise.
    if let Some(dtype) = t.str_opt("mamba_ssm_dtype")? {
        t.expect_eq("mamba_ssm_dtype", dtype, "float32")?;
    }
    let d = TransformersDecoderSection::read(&t)?;
    t.expect_eq("hidden_act", d.hidden_act.as_str(), "silu")?;
    t.expect_eq("attention_bias", d.attention_bias, false)?;
    t.expect_eq("sliding_window", d.sliding_window, None)?;
    t.expect_eq("rope_parameters.rope_type", d.rope_type.as_str(), "default")?;
    let mlp_only = t.usizes("mlp_only_layers").unwrap_or_default();
    t.expect_eq("mlp_only_layers", mlp_only, Vec::new())?;

    let rope = t.section_opt("rope_parameters")?;
    let factor = match &rope {
        Some(rope) => rope.f64_opt("partial_rotary_factor")?,
        None => None,
    };
    let factor = match factor {
        Some(f) => f,
        None => t.f64_opt("partial_rotary_factor")?.unwrap_or(0.25),
    };

    let n_layers = d.num_hidden_layers;
    let config = Qwen3_5Config::new(
        t.usize("vocab_size")?,
        d.hidden_size,
        n_layers,
        d.num_attention_heads,
        d.num_key_value_heads,
        d.head_dim,
        d.intermediate_size,
        t.usize("linear_num_key_heads")?,
        t.usize("linear_num_value_heads")?,
        t.usize("linear_key_head_dim")?,
        t.usize("linear_value_head_dim")?,
    )
    .with_linear_conv_kernel(t.usize_opt("linear_conv_kernel_dim")?.unwrap_or(4))
    .with_rms_eps(d.rms_norm_eps)
    .with_rope_theta(d.rope_theta)
    .with_partial_rotary_factor(factor)
    .with_attn_output_gate(
        t.bool_or("attn_output_gate", true)?
            .then_some(AttentionOutputGate::Elementwise),
    )
    .with_tied_head(t.bool_or("tie_word_embeddings", tie_default)?);
    let interval = t
        .usize_opt("full_attention_interval")?
        .unwrap_or(config.full_attention_interval);
    let config = config.with_full_attention_interval(interval);
    let config = match &d.layer_types {
        Some(types) => {
            t.expect_eq("layer_types.len", types.len(), n_layers)?;
            let kinds = types
                .iter()
                .map(|kind| match kind.as_str() {
                    "linear_attention" => Ok(Qwen3_5LayerKind::Linear),
                    "full_attention" => Ok(Qwen3_5LayerKind::Full),
                    other => Err(BunsenError::unsupported(format!(
                        "config.json: a layer type of {other:?}"
                    ))),
                })
                .collect::<BunsenResult<_>>()?;
            config.with_layer_kinds(kinds)
        }
        None => config.with_every_nth_full(),
    };

    if let Some(rope) = &rope
        && let Ok(section) = rope.usizes("mrope_section")
    {
        rope.expect_eq(
            "mrope_section.sum",
            section.iter().sum::<usize>(),
            config.rotary_dim() / 2,
        )?;
    }
    Ok(config)
}

/// [`qwen3_5_config`] of a `config.json`'s text.
///
/// # Errors
/// As [`config_from_str`] and [`qwen3_5_config`].
pub fn qwen3_5_config_from_str(json: &str) -> BunsenResult<Qwen3_5Config> {
    config_from_str(json, qwen3_5_config)
}

/// [`qwen3_5_config`] of the `config.json` at `path`; every error names the
/// file.
///
/// # Errors
/// As [`config_from_path`] and [`qwen3_5_config`].
pub fn qwen3_5_config_from_path(path: &Path) -> BunsenResult<Qwen3_5Config> {
    config_from_path(path, qwen3_5_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::testing::assert_rekeys,
        errors::testing::ErrorMatcher,
    };

    /// The checkpoints' names land on the model's canonical names, the
    /// vision tower and the prediction head dropped.
    #[test]
    fn test_qwen3_5_rekey() {
        let rekey = qwen3_5_rekey(&Qwen3_5Config::qwen3_5_0_8b());
        assert_rekeys(
            &rekey,
            [
                (
                    "model.language_model.embed_tokens.weight",
                    Some("embedding.weight"),
                ),
                ("model.embed_tokens.weight", Some("embedding.weight")),
                (
                    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                    Some("stack.layers.0.attn.in_proj_qkv.weight"),
                ),
                (
                    "model.language_model.layers.0.linear_attn.A_log",
                    Some("stack.layers.0.attn.a_log"),
                ),
                (
                    "model.language_model.layers.0.linear_attn.conv1d.weight",
                    Some("stack.layers.0.attn.conv1d.weight"),
                ),
                (
                    "model.language_model.layers.0.linear_attn.norm.weight",
                    Some("stack.layers.0.attn.norm.norm.weight"),
                ),
                (
                    "model.language_model.layers.3.self_attn.q_proj.weight",
                    Some("stack.layers.3.attn.c_q.weight"),
                ),
                (
                    "model.language_model.layers.3.self_attn.q_norm.weight",
                    Some("stack.layers.3.attn.q_norm.weight"),
                ),
                (
                    "model.language_model.layers.3.input_layernorm.weight",
                    Some("stack.layers.3.attn_norm.weight"),
                ),
                (
                    "model.language_model.layers.3.mlp.down_proj.weight",
                    Some("stack.layers.3.mlp.down_proj.weight"),
                ),
                (
                    "model.language_model.norm.weight",
                    Some("stack.norm.weight"),
                ),
                ("lm_head.weight", Some("head.proj.weight")),
                ("model.visual.patch_embed.proj.weight", None),
                ("mtp.fc.weight", None),
            ],
        );
    }

    /// The `1 + w` norms are loaded as `1 + w`; the `DeltaNet`'s gated norm,
    /// and everything else, as stored.
    #[test]
    fn test_zero_centered_norms() {
        let rekey = qwen3_5_rekey(&Qwen3_5Config::qwen3_5_0_8b());
        let transformed = |name: &str| {
            rekey
                .target(name)
                .is_some_and(|target| !target.transform_names.is_empty())
        };
        for name in [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.language_model.layers.3.self_attn.q_norm.weight",
            "model.language_model.layers.3.self_attn.k_norm.weight",
            "model.language_model.norm.weight",
        ] {
            assert!(transformed(name), "{name}: `1 + w`");
        }
        for name in [
            "model.language_model.layers.0.linear_attn.norm.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.3.self_attn.q_proj.weight",
        ] {
            assert!(!transformed(name), "{name}: as stored");
        }

        let as_stored =
            qwen3_5_rekey(&Qwen3_5Config::qwen3_5_0_8b().with_zero_centered_norms(false));
        for name in [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.3.self_attn.q_norm.weight",
            "model.language_model.norm.weight",
        ] {
            assert!(
                as_stored
                    .target(name)
                    .is_some_and(|target| target.transform_names.is_empty()),
                "{name}: as stored without zero-centered norms"
            );
        }
    }

    const QWEN3_5_0_8B: &str = r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5", "tie_word_embeddings": true,
        "text_config": {
            "model_type": "qwen3_5_text", "attention_bias": false,
            "attn_output_gate": true, "full_attention_interval": 4, "head_dim": 256,
            "hidden_act": "silu", "hidden_size": 1024, "intermediate_size": 3584,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 16,
            "linear_value_head_dim": 128, "max_position_embeddings": 262144,
            "mlp_only_layers": [], "num_attention_heads": 8, "num_hidden_layers": 24,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "tie_word_embeddings": true,
            "vocab_size": 248320,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "mrope_interleaved": true, "mrope_section": [11, 11, 10],
                "rope_type": "default", "rope_theta": 10000000,
                "partial_rotary_factor": 0.25
            }
        },
        "vision_config": {"depth": 12, "hidden_size": 768}
    }"#;

    /// `Qwen3.5-0.8B`'s `config.json`, as the Hub serves it.
    #[test]
    fn test_reads_the_0_8b() {
        let config = qwen3_5_config_from_str(QWEN3_5_0_8B).unwrap();
        assert_eq!(
            (
                config.n_vocab,
                config.d_model,
                config.n_layers,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.d_ff
            ),
            (248_320, 1024, 24, 8, 2, 256, 3584)
        );
        assert_eq!(
            (
                config.linear_key_heads,
                config.linear_value_heads,
                config.linear_key_dim,
                config.linear_value_dim,
                config.linear_conv_kernel
            ),
            (16, 16, 128, 128, 4)
        );
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert_eq!(config.rotary_dim(), 64);
        assert!(
            config.tied_head && config.attn_output_gate == Some(AttentionOutputGate::Elementwise)
        );
        assert_eq!(config.layer_kinds, Qwen3_5Config::every_nth_full(24, 4));
    }

    /// A file that omits the layer types, the tie and the rotary share takes
    /// `transformers`' defaults: every `full_attention_interval`-th layer
    /// full, an untied head, a quarter of each head rotated.
    #[test]
    fn test_defaults() {
        let json = QWEN3_5_0_8B
            .replace(r#""tie_word_embeddings": true,"#, "")
            .replace(
                r#""full_attention_interval": 4,"#,
                r#""full_attention_interval": 3,"#,
            )
            .replace(r#", "partial_rotary_factor": 0.25"#, "");
        let json = json[..json.find(r#""layer_types""#).unwrap()].to_string()
            + &json[json.find(r#""rope_parameters""#).unwrap()..];
        let config = qwen3_5_config_from_str(&json).unwrap();
        assert!(!config.tied_head);
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.layer_kinds, Qwen3_5Config::every_nth_full(24, 3));
    }

    /// An f32 delta-rule state, as the release's files state it, is read.
    #[test]
    fn test_f32_state_accepted() {
        let json = QWEN3_5_0_8B.replace(
            r#""model_type": "qwen3_5_text", "attention_bias": false,"#,
            r#""model_type": "qwen3_5_text", "attention_bias": false,
               "mamba_ssm_dtype": "float32","#,
        );
        assert_ne!(json, QWEN3_5_0_8B);
        qwen3_5_config_from_str(&json).unwrap();
    }

    /// What bunsen's Qwen3.5 is not is refused, naming the field.
    #[test]
    fn test_refusals() {
        for (from, to, field) in [
            (
                r#""model_type": "qwen3_5","#,
                r#""model_type": "qwen3_5_moe","#,
                "model_type",
            ),
            (
                r#""hidden_act": "silu""#,
                r#""hidden_act": "gelu""#,
                "hidden_act",
            ),
            (
                r#""mrope_section": [11, 11, 10]"#,
                r#""mrope_section": [11, 11, 11]"#,
                "mrope_section",
            ),
            (
                r#""mlp_only_layers": []"#,
                r#""mlp_only_layers": [2]"#,
                "mlp_only_layers",
            ),
            (
                r#""rope_type": "default""#,
                r#""rope_type": "yarn""#,
                "rope_type",
            ),
            (
                r#"["Qwen3_5ForConditionalGeneration"]"#,
                r#"["Qwen3_5ForCausalLM"]"#,
                "architectures",
            ),
            (
                r#""model_type": "qwen3_5_text", "attention_bias": false,"#,
                r#""model_type": "qwen3_5_text", "attention_bias": false,
                   "mamba_ssm_dtype": "bfloat16","#,
                "mamba_ssm_dtype",
            ),
        ] {
            let json = QWEN3_5_0_8B.replace(from, to);
            assert_ne!(json, QWEN3_5_0_8B, "{from}");
            ErrorMatcher::unsupported()
                .message_contains(field)
                .assert_err(&qwen3_5_config_from_str(&json));
        }
    }
}
