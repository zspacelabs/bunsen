//! Loading `transformers`-named tensors into a [`ModernBertModel`], and
//! reading its `config.json`.
use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    ModernBertConfig,
    ModernBertLayerKind,
    ModernBertModel,
};
use crate::{
    blocks::transformers::decoders::causal::{
        CausalDecoderBlock,
        DecoderLayerKeys,
    },
    burner::module::{
        Rekey,
        TensorLoadRowSplit,
    },
    data::pretrained::{
        NamedTensorsApplied,
        load_rekeyed,
        split_rows,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::model_config::TransformersConfigJson,
};

/// The fused projections of a `transformers` `ModernBERT` layer, split by
/// rows into the parts the block holds apart, named in the checkpoint's
/// terms: `attn.Wqkv` into `attn.Wq`, `attn.Wk` and `attn.Wv`; `mlp.Wi`
/// into `mlp.Wi_gate`, its first half (upstream's `input`, the half the
/// activation takes), and `mlp.Wi_up`, its second (upstream's `gate`, the
/// half it multiplies).
pub fn transformers_row_splits() -> [TensorLoadRowSplit; 2] {
    [
        TensorLoadRowSplit::equal(
            ".attn.Wqkv.weight",
            [".attn.Wq.weight", ".attn.Wk.weight", ".attn.Wv.weight"],
        ),
        TensorLoadRowSplit::equal(
            ".mlp.Wi.weight",
            [".mlp.Wi_gate.weight", ".mlp.Wi_up.weight"],
        ),
    ]
}

/// The names of a `transformers` `ModernBERT` layer's parts, once
/// [`transformers_row_splits`] has split its projections: `attn_norm` and
/// `mlp_norm`, and `attn`'s `Wq`, `Wk`, `Wv` and `Wo`.
pub fn transformers_layer_keys() -> DecoderLayerKeys {
    DecoderLayerKeys {
        attn: "attn".to_string(),
        attn_projs: ["Wq", "Wk", "Wv", "Wo"].map(String::from),
        ..DecoderLayerKeys::llama("attn_norm", "mlp_norm")
    }
}

/// `transformers`' `ModernBertModel` state-dict names, after
/// [`transformers_row_splits`], onto a [`ModernBertModel`]'s module paths,
/// relative to the encoder: the embedding's table and norm, each layer by
/// [`transformers_layer_keys`] and its `GeGLU`'s parts (`Wi_gate`, `Wi_up`,
/// `Wo`), and the final norm. Anything else is dropped (a `ModernBertFor*`'s
/// head).
///
/// Normalization parameters keep their `weight` spellings, which
/// [`load_rekeyed`] resolves to `gamma`.
pub fn transformers_rekey() -> Rekey {
    let layer = CausalDecoderBlock::layer_rekey(&transformers_layer_keys())
        .rename("mlp.Wi_gate.", "mlp.gate_proj.")
        .rename("mlp.Wi_up.", "mlp.up_proj.")
        .rename("mlp.Wo.", "mlp.down_proj.");
    Rekey::drop_rest()
        .rename("embeddings.tok_embeddings.", "embeddings.")
        .rename("embeddings.norm.", "embeddings.norm.")
        .mount_indexed("layers.", "stack.layers.", layer)
        .rename("final_norm.", "stack.norm.")
}

/// The prefix a checkpoint's encoder sits under: `model.` for a
/// `ModernBertFor*`'s (a masked LM's, a classifier's), none for a bare
/// `ModernBertModel`'s.
pub fn transformers_prefix(tensors: &[PackTensor]) -> &'static str {
    if tensors.iter().any(|t| t.name.starts_with("model.layers.")) {
        "model."
    } else {
        ""
    }
}

/// Loads tensors named as a `transformers` `ModernBERT` state dict (a bare
/// model's, or under [`transformers_prefix`]) into `model`: the fused
/// projections split ([`transformers_row_splits`]), then [`load_rekeyed`]
/// under [`transformers_rekey`]; in `dtype` (a load policy's resolution),
/// or, when `None`, in the dtype `model` was built in.
///
/// # Errors
/// As [`split_rows`] and [`load_rekeyed`].
pub fn load_transformers_tensors(
    model: &mut ModernBertModel,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<burn::tensor::FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors.into_iter().collect();
    let prefix = transformers_prefix(&tensors);
    let tensors = split_rows(tensors, &transformers_row_splits())?;
    let rekey = Rekey::drop_rest().mount(prefix, "", transformers_rekey());
    load_rekeyed(model, tensors, &rekey, dtype)
}

/// The [`ModernBertConfig`] of a `transformers` `ModernBERT` `config.json`
/// (`model_type` `modernbert`).
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
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's `ModernBERT` is not.
pub fn modernbert_config_from_transformers(json: &str) -> BunsenResult<ModernBertConfig> {
    let t = TransformersConfigJson::parse(json)?;
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

    Ok(ModernBertConfig::new(
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer's names, once split, land on the block's: the projections,
    /// the norms, the `GeGLU`'s parts.
    #[test]
    fn test_transformers_rekey() {
        let rekey = Rekey::drop_rest().mount("model.", "", transformers_rekey());
        for (stored, bunsen) in [
            (
                "model.embeddings.tok_embeddings.weight",
                "embeddings.weight",
            ),
            ("model.embeddings.norm.weight", "embeddings.norm.weight"),
            (
                "model.layers.1.attn_norm.weight",
                "stack.layers.1.attn_norm.weight",
            ),
            (
                "model.layers.0.attn.Wq.weight",
                "stack.layers.0.attn.c_q.weight",
            ),
            (
                "model.layers.0.attn.Wk.weight",
                "stack.layers.0.attn.c_k.weight",
            ),
            (
                "model.layers.0.attn.Wv.weight",
                "stack.layers.0.attn.c_v.weight",
            ),
            (
                "model.layers.0.attn.Wo.weight",
                "stack.layers.0.attn.c_proj.weight",
            ),
            (
                "model.layers.0.mlp_norm.weight",
                "stack.layers.0.mlp_norm.weight",
            ),
            (
                "model.layers.2.mlp.Wi_gate.weight",
                "stack.layers.2.mlp.gate_proj.weight",
            ),
            (
                "model.layers.2.mlp.Wi_up.weight",
                "stack.layers.2.mlp.up_proj.weight",
            ),
            (
                "model.layers.2.mlp.Wo.weight",
                "stack.layers.2.mlp.down_proj.weight",
            ),
            ("model.final_norm.weight", "stack.norm.weight"),
        ] {
            assert_eq!(rekey.apply(stored).as_deref(), Some(bunsen), "{stored}");
        }
        assert_eq!(rekey.apply("head.dense.weight"), None);
        assert_eq!(rekey.apply("model.head.norm.weight"), None);
    }

    /// Laya's encoder config: 28 layers, every third global from the
    /// first, its `rope_parameters` per kind.
    #[test]
    fn test_config_from_transformers() {
        let json = r#"{
            "model_type": "modernbert", "vocab_size": 50368, "hidden_size": 1024,
            "num_hidden_layers": 6, "num_attention_heads": 16, "intermediate_size": 2624,
            "hidden_activation": "gelu", "norm_eps": 1e-05, "norm_bias": false,
            "attention_bias": false, "mlp_bias": false, "local_attention": 128,
            "layer_types": ["full_attention", "sliding_attention", "sliding_attention",
                            "full_attention", "sliding_attention", "sliding_attention"],
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }"#;
        let config = modernbert_config_from_transformers(json).unwrap();
        assert_eq!(
            (
                config.n_vocab,
                config.d_model,
                config.n_layers,
                config.n_heads,
                config.d_ff
            ),
            (50368, 1024, 6, 16, 2624)
        );
        assert_eq!(config.local_radius, 64);
        assert_eq!(
            (config.global_rope_theta, config.local_rope_theta),
            (160000.0, 10000.0)
        );
        use ModernBertLayerKind::*;
        assert_eq!(
            config.layer_kinds,
            [Global, Local, Local, Global, Local, Local]
        );

        // The older spelling: a pattern and two thetas.
        let older = r#"{
            "model_type": "modernbert", "vocab_size": 8, "hidden_size": 8,
            "num_hidden_layers": 4, "num_attention_heads": 2, "intermediate_size": 12,
            "global_attn_every_n_layers": 2, "global_rope_theta": 1000.0,
            "local_rope_theta": 100.0, "local_attention": 8
        }"#;
        let config = modernbert_config_from_transformers(older).unwrap();
        assert_eq!(config.layer_kinds, [Global, Local, Global, Local]);
        assert_eq!(
            (
                config.global_rope_theta,
                config.local_rope_theta,
                config.local_radius
            ),
            (1000.0, 100.0, 4)
        );

        let biased = json.replace(r#""mlp_bias": false"#, r#""mlp_bias": true"#);
        assert!(modernbert_config_from_transformers(&biased).is_err());
    }
}
