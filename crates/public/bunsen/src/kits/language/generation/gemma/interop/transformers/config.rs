//! The config conversion.

use std::path::Path;

use crate::{
    blocks::transformers::embedding::RopeScaling,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::generation::gemma::{
            GemmaConfig,
            GemmaLayerKind,
        },
        support::interop::transformers::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

/// A Gemma generation, as a `config.json`'s `model_type` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Generation {
    Gemma,
    Gemma2,
    Gemma3,
}

/// The [`GemmaConfig`] of a `transformers` Gemma `config.json`: `model_type`
/// `gemma`, `gemma2`, `gemma3_text`, or `gemma3` (a multimodal model, read
/// from its `text_config`; its vision tower is not part of bunsen's model,
/// and its text path applies no final soft-cap).
///
/// Read in both spellings `transformers` has written: the rotary bases as
/// `rope_parameters` (per layer kind, for Gemma 3) or as `rope_theta`,
/// `rope_local_base_freq` and `rope_scaling`; the layer kinds as
/// `layer_types` or, without them, `transformers`' pattern (Gemma 2:
/// alternate layers sliding, from the first; Gemma 3: every
/// `sliding_window_pattern`-th layer full, 6 by default).
///
/// What bunsen's model fixes is checked: the tanh `GELU` (Gemma 1's
/// `"gelu"`, which `transformers` reads as it), no attention bias, a tied
/// head, causal attention, and only a linear rotary scaling (on Gemma 3's
/// global table).
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Gemma is not.
pub fn gemma_config(c: &TransformersConfigJson) -> BunsenResult<GemmaConfig> {
    let root = c.clone();
    let tie_default = root.bool_or("tie_word_embeddings", true)?;
    let (t, generation, multimodal) = match root.str("model_type")? {
        "gemma" => (root, Generation::Gemma, false),
        "gemma2" => (root, Generation::Gemma2, false),
        "gemma3_text" => (root, Generation::Gemma3, false),
        "gemma3" => {
            let text = root.section("text_config")?;
            text.expect_str("model_type", "gemma3_text")?;
            (text, Generation::Gemma3, true)
        }
        other => {
            return Err(BunsenError::unsupported(format!(
                "config.json: `model_type` is {other:?}, not a Gemma 1, 2 or 3"
            )));
        }
    };

    let hidden = t.usize("hidden_size")?;
    let heads = t.usize("num_attention_heads")?;
    let (field, activation) = match t.str_opt("hidden_activation")? {
        Some(act) => ("hidden_activation", act),
        None => ("hidden_act", t.str("hidden_act")?),
    };
    if !(activation == "gelu_pytorch_tanh"
        || (generation == Generation::Gemma && activation == "gelu"))
    {
        t.expect_eq(field, activation, "gelu_pytorch_tanh")?;
    }
    t.expect_eq("attention_bias", t.bool_or("attention_bias", false)?, false)?;
    t.expect_eq(
        "tie_word_embeddings",
        t.bool_or("tie_word_embeddings", tie_default)?,
        true,
    )?;
    t.expect_eq(
        "use_bidirectional_attention",
        t.bool_or("use_bidirectional_attention", false)?,
        false,
    )?;

    let n_layers = t.usize("num_hidden_layers")?;
    let mut config = GemmaConfig::new(
        t.usize("vocab_size")?,
        hidden,
        n_layers,
        heads,
        t.usize_opt("num_key_value_heads")?.unwrap_or(heads),
        t.usize_opt("head_dim")?.unwrap_or(hidden / heads),
        t.usize("intermediate_size")?,
    )
    .with_rms_eps(t.f64_opt("rms_norm_eps")?.unwrap_or(1e-6));

    if generation == Generation::Gemma {
        let (theta, scaling) = rope_of(&t, None)?;
        t.expect_eq("rope_scaling", scaling, RopeScaling::None)?;
        return Ok(config.with_rope_theta(theta));
    }

    config = config
        .with_post_norms(true)
        .with_query_pre_attn_scalar(t.f64_opt("query_pre_attn_scalar")?)
        .with_sliding_window(t.usize_opt("sliding_window")?)
        .with_layer_kinds(layer_kinds(&t, generation, n_layers)?);
    match generation {
        Generation::Gemma2 => {
            let (theta, scaling) = rope_of(&t, None)?;
            t.expect_eq("rope_scaling", scaling, RopeScaling::None)?;
            Ok(config
                .with_rope_theta(theta)
                .with_attn_softcap(t.f64_opt("attn_logit_softcapping")?)
                .with_final_softcap(t.f64_opt("final_logit_softcapping")?))
        }
        _ => {
            let (theta, scaling) = rope_of(&t, Some("full_attention"))?;
            let local = match t.section_opt("rope_parameters")? {
                Some(rope) if rope.section_opt("sliding_attention")?.is_some() => {
                    let (theta, scaling) = rope_of(&t, Some("sliding_attention"))?;
                    rope.section("sliding_attention")?.expect_eq(
                        "rope_type",
                        scaling,
                        RopeScaling::None,
                    )?;
                    theta
                }
                _ => t.f64_opt("rope_local_base_freq")?.unwrap_or(10_000.0),
            };
            Ok(config
                .with_qk_norm(true)
                .with_rope_theta(theta)
                .with_rope_scaling(scaling)
                .with_rope_local_theta(Some(local))
                .with_final_softcap(if multimodal {
                    None
                } else {
                    t.f64_opt("final_logit_softcapping")?
                }))
        }
    }
}

/// A rotary table's base and scaling: from `rope_parameters` (its `kind`
/// entry, for a per-kind table), else `rope_theta` and `rope_scaling`.
fn rope_of(
    t: &TransformersConfigJson,
    kind: Option<&str>,
) -> BunsenResult<(f64, RopeScaling)> {
    let params = match (t.section_opt("rope_parameters")?, kind) {
        (Some(rope), Some(kind)) if rope.section_opt(kind)?.is_some() => Some(rope.section(kind)?),
        (Some(rope), None) => Some(rope),
        (Some(rope), Some(_)) if rope.f64_opt("rope_theta")?.is_some() => Some(rope),
        _ => None,
    };
    let (theta, scaling) = match &params {
        Some(p) => (p.f64("rope_theta")?, Some(p.clone())),
        None => (t.f64("rope_theta")?, t.section_opt("rope_scaling")?),
    };
    let scaling = match scaling {
        None => RopeScaling::None,
        Some(s) => {
            let kind = match s.str_opt("rope_type")? {
                Some(kind) => kind.to_string(),
                None => s.str_opt("type")?.unwrap_or("default").to_string(),
            };
            match kind.as_str() {
                "default" => RopeScaling::None,
                "linear" => RopeScaling::Linear {
                    factor: s.f64("factor")?,
                },
                other => {
                    return Err(BunsenError::unsupported(format!(
                        "config.json: a rope type of {other:?}; bunsen's Gemma reads \
                         `default` and `linear`"
                    )));
                }
            }
        }
    };
    Ok((theta, scaling))
}

/// Each layer's kind: `layer_types`, else `transformers`' pattern for the
/// generation.
fn layer_kinds(
    t: &TransformersConfigJson,
    generation: Generation,
    n_layers: usize,
) -> BunsenResult<Vec<GemmaLayerKind>> {
    if let Some(types) = t.strs_opt("layer_types")? {
        t.expect_eq("layer_types.len", types.len(), n_layers)?;
        return types
            .iter()
            .map(|kind| match kind.as_str() {
                "sliding_attention" => Ok(GemmaLayerKind::Sliding),
                "full_attention" => Ok(GemmaLayerKind::Full),
                other => Err(BunsenError::unsupported(format!(
                    "config.json: a layer type of {other:?}"
                ))),
            })
            .collect();
    }
    let full_every = match generation {
        Generation::Gemma2 => 2,
        _ => match t.usize_opt("sliding_window_pattern")? {
            Some(n) => n,
            None => t.usize_opt("_sliding_window_pattern")?.unwrap_or(6),
        },
    };
    Ok(GemmaConfig::every_nth_full(n_layers, full_every))
}

/// [`gemma_config`] of a `config.json`'s text.
///
/// # Errors
/// As [`config_from_str`] and [`gemma_config`].
pub fn gemma_config_from_str(json: &str) -> BunsenResult<GemmaConfig> {
    config_from_str(json, gemma_config)
}

/// [`gemma_config`] of the `config.json` at `path`; every error names the
/// file.
///
/// # Errors
/// As [`config_from_path`] and [`gemma_config`].
pub fn gemma_config_from_path(path: &Path) -> BunsenResult<GemmaConfig> {
    config_from_path(path, gemma_config)
}
