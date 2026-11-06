//! Loading `transformers`-named tensors into a [`GemmaModel`], and reading
//! its `config.json`.
use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    GemmaConfig,
    GemmaLayerKind,
    GemmaModel,
};
use crate::{
    blocks::transformers::{
        decoders::causal::{
            CausalDecoderBlock,
            DecoderLayerKeys,
        },
        embedding::RopeScaling,
    },
    burner::module::{
        CanonicalKeys,
        Rekey,
        Select,
        TensorLoadTransform,
    },
    data::pretrained::{
        NamedTensorsApplied,
        load_rekeyed,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::model_config::TransformersConfigJson,
};

/// The prefixes a Gemma's text model sits under: `transformers` 5's
/// `Gemma3ForConditionalGeneration`, its 4.x spelling, and a causal LM's
/// (`GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`), in the
/// order they are tried.
pub const GEMMA_TEXT_PREFIXES: [&str; 3] =
    ["model.language_model.", "language_model.model.", "model."];

/// `transformers`' Gemma decoder layer names onto a [`CausalDecoderBlock`]'s,
/// relative to it. With `post_norms` (Gemma 2 and 3), the four norms of the
/// sandwich: `input_layernorm` and `post_attention_layernorm` the
/// attention's, `pre_feedforward_layernorm` and `post_feedforward_layernorm`
/// the feed-forward's. Without (Gemma 1), Llama's two. `self_attn`'s
/// projections and QK-norms, and the `GeGLU`'s projections, as
/// [`CausalDecoderBlock::layer_rekey`] maps them.
pub fn text_layer_rekey(post_norms: bool) -> Rekey {
    if post_norms {
        CausalDecoderBlock::layer_rekey(
            &DecoderLayerKeys::llama("input_layernorm", "pre_feedforward_layernorm")
                .with_post_norms("post_attention_layernorm", "post_feedforward_layernorm"),
        )
    } else {
        CausalDecoderBlock::layer_rekey(&DecoderLayerKeys::llama(
            "input_layernorm",
            "post_attention_layernorm",
        ))
    }
}

/// `transformers`' Gemma state-dict names onto a [`GemmaModel`]'s module
/// paths: under any of [`GEMMA_TEXT_PREFIXES`], the layers by
/// [`text_layer_rekey`], the embedding and the final norm; every norm's
/// gain, stored as `w` for `1 + w`, loaded as `1 + w`
/// ([`TensorLoadTransform::zero_centered_gain`]). Anything else is dropped: a
/// tied head's copy, a vision tower and its projector.
pub fn transformers_rekey(post_norms: bool) -> Rekey {
    let text = || {
        Rekey::keep_rest()
            .mount_indexed("layers.", "stack.layers.", text_layer_rekey(post_norms))
            .rename("norm.", "stack.norm.")
    };
    GEMMA_TEXT_PREFIXES
        .iter()
        .fold(Rekey::drop_rest(), |rekey, prefix| {
            rekey.mount(*prefix, "", text())
        })
        .transform(
            Select::suffix("norm.weight"),
            TensorLoadTransform::zero_centered_gain(),
        )
}

/// Loads tensors named as a `transformers` Gemma state dict into `model`:
/// [`load_rekeyed`] under [`transformers_rekey`] for `model`'s generation, in
/// the dtype `model` was built in.
///
/// # Errors
/// As [`load_rekeyed`].
pub fn load_transformers_tensors(
    model: &mut GemmaModel,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let rekey = transformers_rekey(model.stack.layers[0].attn_post_norm.is_some());
    let rekey = rekey.then(model.canonical_rekey());
    load_rekeyed(model, tensors, &rekey, None)
}

/// [`load_transformers_tensors`], every floating tensor placed in `dtype` (a
/// load policy's resolution) rather than the dtype the model was built in.
///
/// # Errors
/// As [`load_rekeyed`].
pub fn load_transformers_tensors_as(
    model: &mut GemmaModel,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: burn::tensor::FloatDType,
) -> BunsenResult<NamedTensorsApplied> {
    let rekey = transformers_rekey(model.stack.layers[0].attn_post_norm.is_some());
    let rekey = rekey.then(model.canonical_rekey());
    load_rekeyed(model, tensors, &rekey, Some(dtype))
}

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
pub fn gemma_config_from_transformers(json: &str) -> BunsenResult<GemmaConfig> {
    let root = TransformersConfigJson::parse(json)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    #[test]
    fn test_transformers_rekey() {
        for (post_norms, hf, bunsen) in [
            (
                true,
                "model.embed_tokens.weight",
                Some("embed_tokens.weight"),
            ),
            (
                true,
                "model.layers.3.post_attention_layernorm.weight",
                Some("stack.layers.3.attn_post_norm.weight"),
            ),
            (
                true,
                "model.layers.3.pre_feedforward_layernorm.weight",
                Some("stack.layers.3.mlp_norm.weight"),
            ),
            (
                true,
                "language_model.model.layers.0.self_attn.q_norm.weight",
                Some("stack.layers.0.attn.q_norm.weight"),
            ),
            (
                true,
                "model.language_model.layers.1.mlp.gate_proj.weight",
                Some("stack.layers.1.mlp.gate_proj.weight"),
            ),
            (true, "model.norm.weight", Some("stack.norm.weight")),
            (
                true,
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                None,
            ),
            (
                true,
                "multi_modal_projector.mm_input_projection_weight",
                None,
            ),
            (true, "lm_head.weight", None),
            (
                false,
                "model.layers.0.post_attention_layernorm.weight",
                Some("stack.layers.0.mlp_norm.weight"),
            ),
        ] {
            assert_eq!(
                transformers_rekey(post_norms).apply(hf).as_deref(),
                bunsen,
                "{hf}"
            );
        }
    }

    /// Every norm loads as `1 + w`; nothing else is transformed.
    #[test]
    fn test_transformers_rekey_transforms_the_norms() {
        let plan = transformers_rekey(true).plan([
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
        ]);
        let transformed: Vec<bool> = plan
            .targets
            .iter()
            .map(|(_, t)| !t.as_ref().unwrap().transform_names.is_empty())
            .collect();
        assert_eq!(transformed, [true, true, true, true, false, false]);
    }

    const GEMMA3_4B: &str = r#"{
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": {
            "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2560, "intermediate_size": 10240, "model_type": "gemma3_text",
            "num_attention_heads": 8, "num_hidden_layers": 12, "num_key_value_heads": 4,
            "query_pre_attn_scalar": 256, "rms_norm_eps": 1e-06,
            "rope_local_base_freq": 10000.0, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
            "rope_theta": 1000000.0, "sliding_window": 1024, "sliding_window_pattern": 6,
            "final_logit_softcapping": null, "vocab_size": 262208
        },
        "vision_config": {"model_type": "siglip_vision_model"}
    }"#;

    /// A multimodal Gemma 3 reads from its text section, in `transformers`
    /// 4's spelling: the pattern, the two tables, the global scaling.
    #[test]
    fn test_gemma3_multimodal() {
        let c = gemma_config_from_transformers(GEMMA3_4B).unwrap();
        assert_eq!(
            (
                c.n_vocab,
                c.d_model,
                c.n_layers,
                c.n_heads,
                c.n_kv_heads,
                c.head_dim,
                c.d_ff
            ),
            (262_208, 2560, 12, 8, 4, 256, 10240)
        );
        assert!(c.post_norms && c.qk_norm);
        assert_eq!(c.sliding_window, Some(1024));
        let full: Vec<usize> = (0..12)
            .filter(|&i| c.layer_kind(i) == GemmaLayerKind::Full)
            .collect();
        assert_eq!(full, [5, 11]);
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert_eq!(c.rope_scaling, RopeScaling::Linear { factor: 8.0 });
        assert_eq!(c.rope_local_theta, Some(10_000.0));
        assert_eq!(c.query_pre_attn_scalar, Some(256.0));
        assert_eq!(c.final_softcap, None);
    }

    /// Gemma 2's caps and alternating pattern; Gemma 1's two norms and one
    /// table.
    #[test]
    fn test_gemma2_and_gemma1() {
        let c = gemma_config_from_transformers(
            r#"{"model_type": "gemma2", "hidden_size": 16, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32,
                "num_hidden_layers": 4, "vocab_size": 50, "hidden_activation": "gelu_pytorch_tanh",
                "query_pre_attn_scalar": 12, "sliding_window": 4, "rope_theta": 10000.0,
                "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0}"#,
        )
        .unwrap();
        assert_eq!(
            c.layer_kinds,
            [
                GemmaLayerKind::Sliding,
                GemmaLayerKind::Full,
                GemmaLayerKind::Sliding,
                GemmaLayerKind::Full
            ]
        );
        assert_eq!((c.attn_softcap, c.final_softcap), (Some(50.0), Some(30.0)));
        assert!(c.post_norms && !c.qk_norm && c.rope_local_theta.is_none());

        let c = gemma_config_from_transformers(
            r#"{"model_type": "gemma", "hidden_size": 16, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32,
                "num_hidden_layers": 2, "vocab_size": 50, "hidden_act": "gelu",
                "hidden_activation": null, "rope_theta": 10000.0, "rope_scaling": null}"#,
        )
        .unwrap();
        assert!(!c.post_norms && c.layer_kinds.is_empty() && c.sliding_window.is_none());
    }

    #[test]
    fn test_refusals() {
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("model_type")
            .assert_err(&gemma_config_from_transformers(
                r#"{"model_type": "llama"}"#,
            ));
        let bidirectional = GEMMA3_4B.replace(
            r#""final_logit_softcapping": null,"#,
            r#""final_logit_softcapping": null, "use_bidirectional_attention": true,"#,
        );
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("use_bidirectional_attention")
            .assert_err(&gemma_config_from_transformers(&bidirectional));
        let yarn = GEMMA3_4B.replace(r#""rope_type": "linear""#, r#""rope_type": "yarn""#);
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("yarn")
            .assert_err(&gemma_config_from_transformers(&yarn));
    }
}
