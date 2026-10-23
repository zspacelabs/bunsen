//! Loading `transformers`-named tensors into a [`Qwen3Asr`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::blocks::{
    Qwen3Asr,
    Qwen3AsrAudioEncoderConfig,
    Qwen3AsrAudioEncoderMeta,
    Qwen3AsrConfig,
};
use crate::{
    blocks::{
        conv::ConvSeq2d,
        transformers::encoders::residual::ResidualEncoderAttentionBlock,
    },
    burner::module::Rekey,
    data::pretrained::{
        HfConfigJson,
        NamedTensorsApplied,
        apply_rekeyed_tensors_as,
    },
    errors::BunsenResult,
    kits::gpts::qwen3::{
        Qwen3Config,
        Qwen3Meta,
    },
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

/// `transformers`' `Qwen3ASRForConditionalGeneration` state-dict names onto
/// a [`Qwen3Asr`]'s module paths: under `model.`, the audio tower's 1-based
/// `conv2d{n}` onto its stem's 0-based `ConvSeq2d` blocks
/// ([`ConvSeq2d::one_based_rekey`]), its layers by
/// `transformers`' Whisper layer names
/// ([`ResidualEncoderAttentionBlock::whisper_hf_rekey`]), the text model's
/// layers by [`text_layer_rekey`], and everything else as it is. Anything
/// outside `model.` (the tied `lm_head.weight`) is dropped.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`apply_rekeyed_tensors_as`] resolves to `gamma` / `beta`.
pub fn hf_rekey() -> Rekey {
    Rekey::drop_rest().mount(
        "model.",
        "",
        Rekey::keep_rest()
            .mount(
                "audio_tower.conv2d",
                "audio_tower.stem.",
                ConvSeq2d::one_based_rekey(),
            )
            .mount_indexed(
                "audio_tower.layers.",
                "audio_tower.layers.",
                ResidualEncoderAttentionBlock::whisper_hf_rekey(),
            )
            .mount_indexed(
                "language_model.layers.",
                "language_model.layers.",
                text_layer_rekey(),
            ),
    )
}

/// Loads tensors named as `transformers`' `Qwen3ASRForConditionalGeneration`
/// state dict into `model`: [`apply_rekeyed_tensors_as`] under
/// [`hf_rekey`], every floating tensor cast to the dtype `model` was
/// built in. The checkpoints ship `bf16`, which not every backend computes
/// in.
///
/// # Errors
/// As [`apply_rekeyed_tensors_as`].
pub fn load_hf_tensors(
    model: &mut Qwen3Asr,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let dtype = model.multi_modal_projector.linear_2.weight.val().dtype();
    apply_rekeyed_tensors_as(model, tensors, &hf_rekey(), Some(dtype))
}

/// The [`Qwen3AsrConfig`] of a `transformers`
/// `Qwen3ASRForConditionalGeneration` `config.json`: what an `hf:` repo's
/// geometry is, `Qwen3-ASR-0.6B`'s or `Qwen3-ASR-1.7B`'s.
///
/// The audio tower's and the text model's sizes, the attention windows, the
/// norm epsilon, the rotary base and the audio placeholder come from the
/// file. What bunsen's model fixes is checked: a GELU audio tower with as
/// many key/value heads as query heads, whose output is the text model's
/// width; a Qwen3 text model with `SiLU`, no attention bias, full attention in
/// every layer, the default rotary embedding, and its head tied to its
/// embedding.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Qwen3-ASR is not.
pub fn qwen3_asr_config_from_hf(json: &str) -> BunsenResult<Qwen3AsrConfig> {
    let c = HfConfigJson::parse(json)?;
    c.expect_str("model_type", "qwen3_asr")?;

    let a = c.section("audio_config")?;
    a.expect_str("activation_function", "gelu")?;
    let heads = a.usize("encoder_attention_heads")?;
    if let Ok(kv) = a.usize("num_key_value_heads") {
        a.expect_eq("num_key_value_heads", kv, heads)?;
    }
    a.expect_eq(
        "scale_embedding",
        a.bool_or("scale_embedding", false)?,
        false,
    )?;
    let audio = Qwen3AsrAudioEncoderConfig::new(
        a.usize("num_mel_bins")?,
        a.usize("d_model")?,
        heads,
        a.usize("encoder_ffn_dim")?,
        a.usize("encoder_layers")?,
        a.usize("downsample_hidden_size")?,
    )
    .with_n_window(a.usize("n_window")?)
    .with_n_window_infer(a.usize("n_window_infer")?);
    if let Ok(positions) = a.usize("max_position_embeddings") {
        a.expect_eq("max_position_embeddings", positions, audio.chunk_out())?;
    }

    let t = c.section("text_config")?;
    t.expect_str("model_type", "qwen3")?;
    t.expect_str("hidden_act", "silu")?;
    t.expect_eq("attention_bias", t.bool_or("attention_bias", false)?, false)?;
    let tied = t.bool_or(
        "tie_word_embeddings",
        c.bool_or("tie_word_embeddings", true)?,
    )?;
    t.expect_eq("tie_word_embeddings", tied, true)?;
    if let Ok(rope) = t.section("rope_parameters") {
        rope.expect_str("rope_type", "default")?;
    }
    let rope_theta = t
        .section("rope_parameters")
        .and_then(|rope| rope.f64("rope_theta"))
        .or_else(|_| t.f64("rope_theta"))?;
    let text = Qwen3Config::new(
        t.usize("vocab_size")?,
        t.usize("hidden_size")?,
        t.usize("num_hidden_layers")?,
        t.usize("num_attention_heads")?,
        t.usize("num_key_value_heads")?,
        t.usize("head_dim")?,
        t.usize("intermediate_size")?,
    )
    .with_rms_eps(t.f64("rms_norm_eps")?)
    .with_rope_theta(rope_theta as usize);
    a.expect_eq("output_dim", a.usize("output_dim")?, text.d_model())?;

    Ok(Qwen3AsrConfig::new(audio, text).with_audio_token(c.usize("audio_token_id")?))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            (
                "model.audio_tower.conv2d1.weight",
                Some("audio_tower.stem.blocks.0.conv.weight"),
            ),
            (
                "model.audio_tower.conv2d3.bias",
                Some("audio_tower.stem.blocks.2.conv.bias"),
            ),
            (
                "model.audio_tower.conv_out.weight",
                Some("audio_tower.conv_out.weight"),
            ),
            (
                "model.audio_tower.layers.3.self_attn.k_proj.bias",
                Some("audio_tower.layers.3.attn.key.bias"),
            ),
            (
                "model.audio_tower.layers.3.self_attn.out_proj.weight",
                Some("audio_tower.layers.3.attn.output.weight"),
            ),
            (
                "model.audio_tower.layers.3.final_layer_norm.weight",
                Some("audio_tower.layers.3.mlp_norm.weight"),
            ),
            (
                "model.audio_tower.layers.3.fc2.bias",
                Some("audio_tower.layers.3.mlp.linear2.bias"),
            ),
            (
                "model.audio_tower.ln_post.weight",
                Some("audio_tower.ln_post.weight"),
            ),
            (
                "model.multi_modal_projector.linear_1.weight",
                Some("multi_modal_projector.linear_1.weight"),
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                Some("language_model.layers.0.attn.c_proj.weight"),
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                Some("language_model.layers.0.attn.q_norm.weight"),
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                Some("language_model.layers.0.mlp.gate_proj.weight"),
            ),
            (
                "model.language_model.norm.weight",
                Some("language_model.norm.weight"),
            ),
            ("lm_head.weight", None),
        ] {
            assert_eq!(rekey.apply(hf).as_deref(), bunsen, "{hf}");
        }
    }

    /// `Qwen3-ASR-0.6B-hf`'s `config.json` is the built-in geometry;
    /// `Qwen3-ASR-1.7B-hf`'s is the larger model's; a model bunsen's is not
    /// is refused by the field that says so.
    #[test]
    fn test_qwen3_asr_config_from_hf() {
        use crate::kits::speech::qwen3_asr::blocks::Qwen3AsrMeta;

        let small = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/qwen3_asr/qwen3-asr-0.6b-hf.config.json"
        ));
        let config = qwen3_asr_config_from_hf(small).unwrap();
        assert_eq!(
            format!("{config:?}"),
            format!("{:?}", Qwen3AsrConfig::qwen3_asr_0_6b())
        );

        let large = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/qwen3_asr/qwen3-asr-1.7b-hf.config.json"
        ));
        let config = qwen3_asr_config_from_hf(large).unwrap();
        assert_eq!(
            format!("{config:?}"),
            format!("{:?}", Qwen3AsrConfig::qwen3_asr_1_7b())
        );
        assert_eq!(config.d_audio(), 1024);
        assert_eq!(config.audio.n_layers(), 24);
        assert_eq!(config.d_text(), 2048);
        assert_eq!(config.text.d_ff, 6144);
        assert_eq!(config.n_vocab(), 151_936);

        let untied = small.replace(
            "\"tie_word_embeddings\": true",
            "\"tie_word_embeddings\": false",
        );
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("tie_word_embeddings")
            .assert_err(&qwen3_asr_config_from_hf(&untied));
    }
}
