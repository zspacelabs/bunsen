//! Loading `transformers`-named tensors into a [`Qwen3Model`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    Qwen3Config,
    Qwen3Model,
};
use crate::{
    burner::module::Rekey,
    data::pretrained::{
        HfConfigJson,
        NamedTensorsApplied,
        apply_rekeyed_tensors_as,
    },
    errors::BunsenResult,
};

/// `transformers`' Qwen3 decoder layer names onto a
/// [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)'s,
/// relative to it: `input_layernorm` the attention's norm,
/// `post_attention_layernorm` the feed-forward's, `self_attn`'s
/// `{q,k,v,o}_proj` the attention's projections and its `{q,k}_norm` its
/// QK-norms, and the `SwiGLU`'s projections as they are.
pub fn text_layer_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("input_layernorm.", "attn_norm.")
        .rename("post_attention_layernorm.", "mlp_norm.")
        .rename("self_attn.q_proj.", "attn.c_q.")
        .rename("self_attn.k_proj.", "attn.c_k.")
        .rename("self_attn.v_proj.", "attn.c_v.")
        .rename("self_attn.o_proj.", "attn.c_proj.")
        .rename("self_attn.", "attn.")
}

/// `transformers`' `Qwen3ForCausalLM` state-dict names onto a
/// [`Qwen3Model`]'s module paths: under `model.`, the layers by
/// [`text_layer_rekey`], and the embedding and final norm as they are.
/// Anything outside `model.` is dropped: the head is tied to the embedding,
/// so a checkpoint's `lm_head.weight` is not read, as `transformers` does
/// not read it under `tie_word_embeddings`.
///
/// Normalization parameters keep their `weight` spelling, which
/// [`apply_rekeyed_tensors_as`] resolves to `gamma`.
pub fn hf_rekey() -> Rekey {
    Rekey::drop_rest().mount(
        "model.",
        "",
        Rekey::keep_rest().mount_indexed("layers.", "layers.", text_layer_rekey()),
    )
}

/// Loads tensors named as `transformers`' `Qwen3ForCausalLM` state dict into
/// `model`: [`apply_rekeyed_tensors_as`] under [`hf_rekey`], every floating
/// tensor cast to the dtype `model` was built in. The checkpoints ship
/// `bf16`, which not every backend computes in.
///
/// # Errors
/// As [`apply_rekeyed_tensors_as`].
pub fn load_hf_tensors(
    model: &mut Qwen3Model,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let dtype = model.embed_tokens.weight.val().dtype();
    apply_rekeyed_tensors_as(model, tensors, &hf_rekey(), Some(dtype))
}

/// The [`Qwen3Config`] of a `transformers` `Qwen3ForCausalLM` `config.json`:
/// what an `hf:` repo's geometry is, any dense Qwen3's.
///
/// As [`qwen3_config_from_hf_section`] over the whole file, which must say
/// `model_type` `qwen3`: a mixture of experts (`qwen3_moe`) is refused. The
/// head is untied when the file does not say, as `transformers`' `Qwen3Config`
/// defaults it.
///
/// # Errors
/// As [`qwen3_config_from_hf_section`].
pub fn qwen3_config_from_hf(json: &str) -> BunsenResult<Qwen3Config> {
    qwen3_config_from_hf_section(&HfConfigJson::parse(json)?, false)
}

/// The [`Qwen3Config`] of a `transformers` Qwen3 text config: a whole
/// `Qwen3ForCausalLM` `config.json`, or the `text_config` section of a model
/// a Qwen3 is the text model of. `tie_default` is whether the head is tied
/// when the section does not say.
///
/// The sizes, the norm epsilon and the rotary base come from the section.
/// What bunsen's model fixes is checked: `model_type` `qwen3`, `SiLU`, no
/// attention bias, no sliding window, the default rotary embedding (no
/// scaling), and the head tied to the embedding.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the section lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Qwen3 is not.
pub fn qwen3_config_from_hf_section(
    t: &HfConfigJson,
    tie_default: bool,
) -> BunsenResult<Qwen3Config> {
    t.expect_str("model_type", "qwen3")?;
    t.expect_str("hidden_act", "silu")?;
    t.expect_eq("attention_bias", t.bool_or("attention_bias", false)?, false)?;
    t.expect_eq(
        "use_sliding_window",
        t.bool_or("use_sliding_window", false)?,
        false,
    )?;
    let tied = t.bool_or("tie_word_embeddings", tie_default)?;
    t.expect_eq("tie_word_embeddings", tied, true)?;
    for rope in ["rope_parameters", "rope_scaling"] {
        if let Ok(rope) = t.section(rope) {
            rope.expect_str("rope_type", "default")?;
        }
    }
    let rope_theta = t
        .section("rope_parameters")
        .and_then(|rope| rope.f64("rope_theta"))
        .or_else(|_| t.f64("rope_theta"))?;
    Ok(Qwen3Config::new(
        t.usize("vocab_size")?,
        t.usize("hidden_size")?,
        t.usize("num_hidden_layers")?,
        t.usize("num_attention_heads")?,
        t.usize("num_key_value_heads")?,
        t.usize("head_dim")?,
        t.usize("intermediate_size")?,
    )
    .with_rms_eps(t.f64("rms_norm_eps")?)
    .with_rope_theta(rope_theta as usize))
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
    fn test_hf_rekey() {
        let rekey = hf_rekey();
        for (hf, bunsen) in [
            ("model.embed_tokens.weight", Some("embed_tokens.weight")),
            (
                "model.layers.0.self_attn.o_proj.weight",
                Some("layers.0.attn.c_proj.weight"),
            ),
            (
                "model.layers.27.self_attn.k_norm.weight",
                Some("layers.27.attn.k_norm.weight"),
            ),
            (
                "model.layers.3.mlp.up_proj.weight",
                Some("layers.3.mlp.up_proj.weight"),
            ),
            ("model.norm.weight", Some("norm.weight")),
            ("lm_head.weight", None),
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
    fn test_qwen3_config_from_hf() {
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
        ] {
            let config = qwen3_config_from_hf(json).unwrap();
            assert_eq!(format!("{config:?}"), format!("{expected:?}"), "{name}");
        }

        let small = config_json!("qwen3-0.6b");
        for (from, to, field) in [
            (
                "\"tie_word_embeddings\": true",
                "\"tie_word_embeddings\": false",
                "tie_word_embeddings",
            ),
            (
                "\"model_type\": \"qwen3\"",
                "\"model_type\": \"qwen3_moe\"",
                "model_type",
            ),
            (
                "\"use_sliding_window\": false",
                "\"use_sliding_window\": true",
                "use_sliding_window",
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
                .assert_err(&qwen3_config_from_hf(&small.replace(from, to)));
        }

        let unsaid = small.replace("\"tie_word_embeddings\": true,", "");
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("tie_word_embeddings")
            .assert_err(&qwen3_config_from_hf(&unsaid));
    }
}
