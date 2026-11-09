//! Loading `transformers`-named tensors into a [`Qwen3Model`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    Qwen3Config,
    Qwen3Model,
};
use crate::{
    blocks::transformers::decoders::causal::{
        CausalDecoderBlock,
        DecoderLayerKeys,
    },
    burner::module::Rekey,
    data::pretrained::NamedTensorsApplied,
    errors::BunsenResult,
    kits::support::{
        construct::load_lm_rekeyed,
        model_config::{
            TransformersConfigJson,
            TransformersDecoderSection,
        },
    },
};

/// `transformers`' Qwen3 decoder layer names onto a
/// [`CausalDecoderBlock`]'s,
/// relative to it: `input_layernorm` the attention's norm,
/// `post_attention_layernorm` the feed-forward's, `self_attn`'s
/// `{q,k,v,o}_proj` the attention's projections and its `{q,k}_norm` its
/// QK-norms, and the `SwiGLU`'s projections as they are.
pub fn text_layer_rekey() -> Rekey {
    CausalDecoderBlock::layer_rekey(&DecoderLayerKeys::llama(
        "input_layernorm",
        "post_attention_layernorm",
    ))
}

/// `transformers`' `Qwen3ForCausalLM` state-dict names onto the module paths
/// of a [`Qwen3Model`]'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
/// under `model.`, the layers by [`text_layer_rekey`], the embedding
/// (`embed_tokens`, the model's `embedding`) and the final norm; and
/// `lm_head.weight`, the head's projection. Anything else is dropped.
///
/// A tied model has no `lm_head` to load: a checkpoint's `lm_head.weight`
/// (`Qwen3-0.6B` carries a copy of the embedding there) is then unused, as
/// `transformers` ties over it under `tie_word_embeddings`.
///
/// Normalization parameters keep their `weight` spelling, which
/// [`load_rekeyed`](crate::data::pretrained::load_rekeyed) resolves to `gamma`.
pub fn transformers_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount(
            "model.",
            "",
            Rekey::keep_rest()
                .mount_indexed("layers.", "stack.layers.", text_layer_rekey())
                .rename("embed_tokens.", "embedding.")
                .rename("norm.", "stack.norm."),
        )
        .exact("lm_head.weight", "head.proj.weight")
}

/// Loads tensors named as `transformers`' `Qwen3ForCausalLM` state dict into
/// `model`'s [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel): [`load_lm_rekeyed`] under
/// [`transformers_rekey`], in the dtype `model`
/// was built in. The checkpoints ship `bf16`, which not every backend computes
/// in.
///
/// # Errors
/// As [`load_lm_rekeyed`].
pub fn load_transformers_tensors(
    model: &mut Qwen3Model,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    load_lm_rekeyed(model, tensors, transformers_rekey(), None)
}

/// [`load_transformers_tensors`], every floating tensor placed in `dtype` (a
/// load policy's resolution, [`LoadPolicy::resolve`]) rather than the dtype the
/// model was built in: a freshly built model's parameters are lazy, so they
/// are never materialized in the build dtype first.
///
/// [`LoadPolicy::resolve`]: crate::kits::support::construct::LoadPolicy::resolve
///
/// # Errors
/// As [`load_lm_rekeyed`].
pub fn load_transformers_tensors_as(
    model: &mut Qwen3Model,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: burn::tensor::FloatDType,
) -> BunsenResult<NamedTensorsApplied> {
    load_lm_rekeyed(model, tensors, transformers_rekey(), Some(dtype))
}

/// The [`Qwen3Config`] of a `transformers` `Qwen3ForCausalLM` `config.json`:
/// what an `hf:` repo's geometry is, any dense Qwen3's.
///
/// As [`qwen3_config_from_transformers_section`] over the whole file, which
/// must say `model_type` `qwen3`: a mixture of experts (`qwen3_moe`) is
/// refused. The head is untied when the file does not say, as `transformers`'
/// `Qwen3Config` defaults it.
///
/// # Errors
/// As [`qwen3_config_from_transformers_section`].
pub fn qwen3_config_from_transformers(json: &str) -> BunsenResult<Qwen3Config> {
    qwen3_config_from_transformers_section(&TransformersConfigJson::parse(json)?, false)
}

/// The [`Qwen3Config`] of a `transformers` Qwen3 text config: a whole
/// `Qwen3ForCausalLM` `config.json`, or the `text_config` section of a model
/// a Qwen3 is the text model of. `tie_default` is whether the head is tied
/// when the section does not say.
///
/// The sizes, the norm epsilon, the rotary base and whether the head is tied
/// come from the section. What bunsen's model fixes is checked: `model_type`
/// `qwen3`, `SiLU`, no attention bias, no sliding window, and the default
/// rotary embedding (no scaling).
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the section lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Qwen3 is not.
pub fn qwen3_config_from_transformers_section(
    t: &TransformersConfigJson,
    tie_default: bool,
) -> BunsenResult<Qwen3Config> {
    t.expect_str("model_type", "qwen3")?;
    let d = TransformersDecoderSection::read(t)?;
    t.expect_eq("hidden_act", d.hidden_act.as_str(), "silu")?;
    t.expect_eq("attention_bias", d.attention_bias, false)?;
    t.expect_eq("sliding_window", d.sliding_window, None)?;
    t.expect_eq("rope_parameters.rope_type", d.rope_type.as_str(), "default")?;
    Ok(Qwen3Config::new(
        t.usize("vocab_size")?,
        d.hidden_size,
        d.num_hidden_layers,
        d.num_attention_heads,
        d.num_key_value_heads,
        d.head_dim,
        d.intermediate_size,
    )
    .with_rms_eps(d.rms_norm_eps)
    .with_rope_theta(d.rope_theta as usize)
    .with_tied_head(t.bool_or("tie_word_embeddings", tie_default)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    /// `transformers`' Qwen3 decoder-layer names, onto the block's own.
    #[test]
    fn test_text_layer_rekey() {
        let rekey = text_layer_rekey();
        for (hf, bunsen) in [
            ("input_layernorm.weight", "attn_norm.weight"),
            ("post_attention_layernorm.weight", "mlp_norm.weight"),
            ("self_attn.q_proj.weight", "attn.c_q.weight"),
            ("self_attn.k_proj.weight", "attn.c_k.weight"),
            ("self_attn.v_proj.weight", "attn.c_v.weight"),
            ("self_attn.o_proj.weight", "attn.c_proj.weight"),
            ("self_attn.q_norm.weight", "attn.q_norm.weight"),
            ("self_attn.k_norm.weight", "attn.k_norm.weight"),
            ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight"),
        ] {
            assert_eq!(rekey.apply(hf).as_deref(), Some(bunsen), "{hf}");
        }
    }

    #[test]
    fn test_transformers_rekey() {
        let rekey = transformers_rekey();
        for (hf, bunsen) in [
            ("model.embed_tokens.weight", Some("embedding.weight")),
            (
                "model.layers.0.self_attn.o_proj.weight",
                Some("stack.layers.0.attn.c_proj.weight"),
            ),
            (
                "model.layers.27.self_attn.k_norm.weight",
                Some("stack.layers.27.attn.k_norm.weight"),
            ),
            (
                "model.layers.3.mlp.up_proj.weight",
                Some("stack.layers.3.mlp.up_proj.weight"),
            ),
            ("model.norm.weight", Some("stack.norm.weight")),
            ("lm_head.weight", Some("head.proj.weight")),
            ("lm_head.bias", None),
        ] {
            assert_eq!(rekey.apply(hf).as_deref(), bunsen, "{hf}");
        }
    }

    macro_rules! config_json {
        ($name:literal) => {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/testdata/qwen3/",
                $name,
                ".config.json"
            ))
        };
    }

    /// Each pinned checkpoint's `config.json` is its built-in geometry, and a
    /// model bunsen's Qwen3 is not is refused by the field that says so.
    #[test]
    fn test_qwen3_config_from_transformers() {
        for (name, json, expected) in [
            (
                "qwen3-0.6b",
                config_json!("qwen3-0.6b"),
                Qwen3Config::qwen3_0_6b(),
            ),
            (
                "qwen3-0.6b-base",
                config_json!("qwen3-0.6b-base"),
                Qwen3Config::qwen3_0_6b(),
            ),
            (
                "qwen3-1.7b",
                config_json!("qwen3-1.7b"),
                Qwen3Config::qwen3_1_7b(),
            ),
            (
                "qwen3-1.7b-base",
                config_json!("qwen3-1.7b-base"),
                Qwen3Config::qwen3_1_7b(),
            ),
            (
                "qwen3-4b",
                config_json!("qwen3-4b"),
                Qwen3Config::qwen3_4b(),
            ),
            (
                "qwen3-4b-base",
                config_json!("qwen3-4b-base"),
                Qwen3Config::qwen3_4b(),
            ),
            (
                "qwen3-4b-instruct-2507",
                config_json!("qwen3-4b-instruct-2507"),
                Qwen3Config::qwen3_4b_2507(),
            ),
            (
                "qwen3-4b-thinking-2507",
                config_json!("qwen3-4b-thinking-2507"),
                Qwen3Config::qwen3_4b_2507(),
            ),
            (
                "qwen3-8b",
                config_json!("qwen3-8b"),
                Qwen3Config::qwen3_8b(),
            ),
            (
                "qwen3-8b-base",
                config_json!("qwen3-8b-base"),
                Qwen3Config::qwen3_8b(),
            ),
            (
                "qwen3-4b-saferl",
                config_json!("qwen3-4b-saferl"),
                Qwen3Config::qwen3_4b(),
            ),
            (
                "qwen3guard-gen-0.6b",
                config_json!("qwen3guard-gen-0.6b"),
                Qwen3Config::qwen3_0_6b(),
            ),
            (
                "qwen3guard-gen-4b",
                config_json!("qwen3guard-gen-4b"),
                Qwen3Config::qwen3_4b(),
            ),
            (
                "qwen3guard-gen-8b",
                config_json!("qwen3guard-gen-8b"),
                Qwen3Config::qwen3_8b(),
            ),
        ] {
            let config = qwen3_config_from_transformers(json).unwrap();
            assert_eq!(format!("{config:?}"), format!("{expected:?}"), "{name}");
        }

        let small = config_json!("qwen3-0.6b");
        for (from, to, field) in [
            (
                "\"model_type\": \"qwen3\"",
                "\"model_type\": \"qwen3_moe\"",
                "model_type",
            ),
            (
                "\"rope_scaling\": null",
                "\"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 4.0}",
                "rope_type",
            ),
        ] {
            assert!(small.contains(from), "{from}");
            ErrorMatcher::kind(BunsenErrorKind::Unsupported)
                .message_contains(field)
                .assert_err(&qwen3_config_from_transformers(&small.replace(from, to)));
        }

        // A window in force is refused; `use_sliding_window: false` switches
        // a stated one off, as `transformers` reads it.
        let windowed = small.replace("\"sliding_window\": null", "\"sliding_window\": 4096");
        assert!(
            qwen3_config_from_transformers(&windowed).is_ok(),
            "switched off"
        );
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("sliding_window")
            .assert_err(&qwen3_config_from_transformers(&windowed.replace(
                "\"use_sliding_window\": false",
                "\"use_sliding_window\": true",
            )));

        // The head is untied where the file says so, or says nothing.
        let untied = small.replace(
            "\"tie_word_embeddings\": true",
            "\"tie_word_embeddings\": false",
        );
        let unsaid = small.replace("\"tie_word_embeddings\": true,", "");
        for json in [untied, unsaid] {
            let config = qwen3_config_from_transformers(&json).unwrap();
            assert!(!config.tied_head);
            assert_eq!(config.d_model, 1024, "the rest as it was");
        }
    }
}
