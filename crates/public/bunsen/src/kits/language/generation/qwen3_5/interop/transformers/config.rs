//! The config conversion.

use std::path::Path;

use crate::{
    blocks::transformers::attention::csa::AttentionOutputGate,
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
        },
    },
};

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
/// its model type's class (whose layout [`QWEN3_5_TEXT_PREFIXES`](super::QWEN3_5_TEXT_PREFIXES) reads).
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
