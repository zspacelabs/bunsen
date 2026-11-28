//! The config conversion.

use std::path::Path;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::encoders::modernbert::{
            ModernBertContractConfig,
            ModernBertLayerKind,
        },
        support::transformers::interop::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

/// The [`ModernBertContractConfig`] of a parsed `transformers` `ModernBERT`
/// `config.json` (`model_type` `modernbert`).
///
/// The layer kinds are read as `layer_types` or, without them, as
/// `global_attn_every_n_layers` (3 by default: every third layer global,
/// from the first); the rotary bases as `rope_parameters` (per layer kind)
/// or as `global_rope_theta` and `local_rope_theta`.
///
/// What bunsen's model fixes is checked: the exact `GELU`, no biases on the
/// projections or the norms, and only the default rotary type.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's `ModernBERT` is not.
pub fn modernbert_config(t: &TransformersConfigJson) -> BunsenResult<ModernBertContractConfig> {
    t.expect_str("model_type", "modernbert")?;
    t.expect_eq(
        "hidden_activation",
        t.str_opt("hidden_activation")?.unwrap_or("gelu"),
        "gelu",
    )?;
    for field in ["attention_bias", "mlp_bias", "norm_bias"] {
        t.expect_eq(field, t.bool_or(field, false)?, false)?;
    }

    let n_layers = t.usize("num_hidden_layers")?;
    let layer_kinds = match t.strs_opt("layer_types")? {
        Some(types) => types
            .iter()
            .enumerate()
            .map(|(i, kind)| match kind.as_str() {
                "full_attention" => Ok(ModernBertLayerKind::Global),
                "sliding_attention" => Ok(ModernBertLayerKind::Local),
                other => Err(BunsenError::unsupported(format!(
                    "config.json: `layer_types[{i}]` is {other:?}"
                ))),
            })
            .collect::<BunsenResult<Vec<_>>>()?,
        None => {
            let every = t.usize_opt("global_attn_every_n_layers")?.unwrap_or(3);
            (0..n_layers)
                .map(|i| {
                    if i.is_multiple_of(every) {
                        ModernBertLayerKind::Global
                    } else {
                        ModernBertLayerKind::Local
                    }
                })
                .collect()
        }
    };

    let (global_theta, local_theta) = match t.section_opt("rope_parameters")? {
        Some(rope) => {
            let theta = |kind: &str| -> BunsenResult<f64> {
                let params = rope.section(kind)?;
                params.expect_eq(
                    "rope_type",
                    params.str_opt("rope_type")?.unwrap_or("default"),
                    "default",
                )?;
                params.f64("rope_theta")
            };
            (theta("full_attention")?, theta("sliding_attention")?)
        }
        None => (
            t.f64_opt("global_rope_theta")?.unwrap_or(160_000.0),
            t.f64_opt("local_rope_theta")?.unwrap_or(10_000.0),
        ),
    };

    Ok(ModernBertContractConfig::new(
        t.usize("vocab_size")?,
        t.usize("hidden_size")?,
        n_layers,
        t.usize("num_attention_heads")?,
        t.usize("intermediate_size")?,
    )
    .with_norm_eps(t.f64_opt("norm_eps")?.unwrap_or(1e-5))
    .with_layer_kinds(layer_kinds)
    .with_local_radius(t.usize_opt("local_attention")?.unwrap_or(128) / 2)
    .with_global_rope_theta(global_theta)
    .with_local_rope_theta(local_theta))
}

/// [`modernbert_config`] of a `config.json`'s text.
///
/// # Errors
/// As [`config_from_str`] and [`modernbert_config`].
pub fn modernbert_config_from_str(json: &str) -> BunsenResult<ModernBertContractConfig> {
    config_from_str(json, modernbert_config)
}

/// [`modernbert_config`] of the `config.json` at `path`; every error names
/// the file.
///
/// # Errors
/// As [`config_from_path`] and [`modernbert_config`].
pub fn modernbert_config_from_path(path: &Path) -> BunsenResult<ModernBertContractConfig> {
    config_from_path(path, modernbert_config)
}
