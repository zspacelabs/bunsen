//! Loading `transformers`-named tensors into a [`VoxtralRealtime`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::blocks::{
    VoxtralAudioEncoderConfig,
    VoxtralRealtime,
    VoxtralRealtimeConfig,
    VoxtralTextConfig,
};
use crate::{
    blocks::conv::ConvSeq1d,
    burner::{
        module::Rekey,
        tensor::param_float_dtype,
    },
    data::pretrained::{
        HfConfigJson,
        NamedTensorsApplied,
        apply_rekeyed_tensors_as,
    },
    errors::BunsenResult,
    kits::gpts::qwen3::weights::text_layer_rekey,
};

/// `transformers`' Voxtral Realtime audio-encoder layer names onto a
/// [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)'s,
/// relative to it: `self_attn_layer_norm` the attention's norm,
/// `final_layer_norm` the feed-forward's, `self_attn`'s `{q,k,v,o}_proj` the
/// attention's projections, and the `SwiGLU`'s projections as they are.
pub fn audio_layer_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("self_attn_layer_norm.", "attn_norm.")
        .rename("final_layer_norm.", "mlp_norm.")
        .rename("self_attn.q_proj.", "attn.c_q.")
        .rename("self_attn.k_proj.", "attn.c_k.")
        .rename("self_attn.v_proj.", "attn.c_v.")
        .rename("self_attn.o_proj.", "attn.c_proj.")
}

/// `transformers`' Voxtral Realtime text-layer names onto a
/// [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)'s:
/// Mistral's, which are Qwen3's ([`text_layer_rekey`]), and `ada_rms_norm`
/// the block's
/// [`mlp_scale`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock::mlp_scale).
pub fn text_layer_rekey_voxtral() -> Rekey {
    text_layer_rekey().rename("ada_rms_norm.", "mlp_scale.")
}

/// The names of `mistralai/Voxtral-Mini-4B-Realtime-2602`'s
/// `model.safetensors` onto a [`VoxtralRealtime`]'s module paths: the audio
/// tower's `embedder.conv1` and `conv2` onto its stem's 0-based
/// [`ConvSeq1d`] blocks ([`ConvSeq1d::one_based_rekey`]), its layers by
/// [`audio_layer_rekey`]; the projector's `linear_1` onto its stacking's
/// projection; the text model, under `language_model.model.`, its layers by
/// [`text_layer_rekey_voxtral`]; and everything else as it is. The head is
/// tied, so nothing else is read.
///
/// Normalization parameters keep their `weight` spelling, which
/// [`apply_rekeyed_tensors_as`] resolves to `gamma`.
pub fn hf_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount(
            "audio_tower.",
            "audio_tower.",
            Rekey::keep_rest()
                .mount("embedder.conv", "stem.", ConvSeq1d::one_based_rekey())
                .mount_indexed("layers.", "layers.", audio_layer_rekey()),
        )
        .mount(
            "multi_modal_projector.",
            "multi_modal_projector.",
            Rekey::keep_rest().rename("linear_1.", "stacking.projection."),
        )
        .mount(
            "language_model.model.",
            "language_model.",
            Rekey::keep_rest().mount_indexed("layers.", "layers.", text_layer_rekey_voxtral()),
        )
}

/// Loads tensors named as `mistralai/Voxtral-Mini-4B-Realtime-2602`'s
/// `model.safetensors` names them into `model`: [`apply_rekeyed_tensors_as`]
/// under [`hf_rekey`], every floating tensor cast to the dtype `model` was
/// built in ([`param_float_dtype`]). The checkpoint ships `bf16`, which not
/// every backend computes in.
///
/// # Errors
/// As [`apply_rekeyed_tensors_as`].
pub fn load_hf_tensors(
    model: &mut VoxtralRealtime,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let dtype = param_float_dtype(&model.multi_modal_projector.linear_2.weight);
    apply_rekeyed_tensors_as(model, tensors, &hf_rekey(), Some(dtype))
}

/// A section's rotary base: `rope_parameters.rope_theta`, else `rope_theta`.
fn rope_theta(section: &HfConfigJson) -> BunsenResult<usize> {
    let theta = section
        .section("rope_parameters")
        .and_then(|rope| rope.f64("rope_theta"))
        .or_else(|_| section.f64("rope_theta"))?;
    Ok(theta as usize)
}

/// The [`VoxtralRealtimeConfig`] of a `transformers` `VoxtralRealtime`
/// `config.json`.
///
/// The audio encoder's and the text model's sizes, their windows, norm
/// epsilons and rotary bases, and the downsampling come from the file. What
/// bunsen's model fixes is checked: a GELU stem and a `SiLU` feed-forward in
/// both stacks, an audio encoder with as many key/value heads as query
/// heads, a projector with a GELU, and a head tied to the embedding.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Voxtral Realtime is not.
pub fn voxtral_config_from_hf(json: &str) -> BunsenResult<VoxtralRealtimeConfig> {
    let c = HfConfigJson::parse(json)?;
    c.expect_str("model_type", "voxtral_realtime")?;
    c.expect_str("projector_hidden_act", "gelu")?;

    let a = c.section("audio_config")?;
    a.expect_str("activation_function", "gelu")?;
    a.expect_str("hidden_act", "silu")?;
    let heads = a.usize("num_attention_heads")?;
    a.expect_eq(
        "num_key_value_heads",
        a.usize("num_key_value_heads")?,
        heads,
    )?;
    let audio = VoxtralAudioEncoderConfig::new(
        a.usize("num_mel_bins")?,
        a.usize("hidden_size")?,
        heads,
        a.usize("head_dim")?,
        a.usize("intermediate_size")?,
        a.usize("num_hidden_layers")?,
    )
    .with_rms_eps(a.f64("rms_norm_eps")?)
    .with_rope_theta(rope_theta(&a)?)
    .with_sliding_window(a.usize("sliding_window").ok());

    let t = c.section("text_config")?;
    t.expect_str("hidden_act", "silu")?;
    t.expect_eq(
        "tie_word_embeddings",
        t.bool_or(
            "tie_word_embeddings",
            c.bool_or("tie_word_embeddings", true)?,
        )?,
        true,
    )?;
    let text = VoxtralTextConfig::new(
        t.usize("vocab_size")?,
        t.usize("hidden_size")?,
        t.usize("num_hidden_layers")?,
        t.usize("num_attention_heads")?,
        t.usize("num_key_value_heads")?,
        t.usize("head_dim")?,
        t.usize("intermediate_size")?,
    )
    .with_rms_eps(t.f64("rms_norm_eps")?)
    .with_rope_theta(rope_theta(&t)?)
    .with_sliding_window(t.usize("sliding_window").ok());

    Ok(VoxtralRealtimeConfig::new(audio, text).with_downsample(c.usize("downsample_factor")?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::speech::voxtral::blocks::{
            VoxtralAudioEncoderMeta,
            VoxtralRealtimeMeta,
            VoxtralTextMeta,
        },
    };

    #[test]
    fn test_hf_rekey() {
        let rekey = hf_rekey();
        for (hf, bunsen) in [
            (
                "audio_tower.embedder.conv1.weight",
                Some("audio_tower.stem.blocks.0.conv.weight"),
            ),
            (
                "audio_tower.embedder.conv2.bias",
                Some("audio_tower.stem.blocks.1.conv.bias"),
            ),
            (
                "audio_tower.layers.31.self_attn_layer_norm.weight",
                Some("audio_tower.layers.31.attn_norm.weight"),
            ),
            (
                "audio_tower.layers.0.final_layer_norm.weight",
                Some("audio_tower.layers.0.mlp_norm.weight"),
            ),
            (
                "audio_tower.layers.0.self_attn.o_proj.bias",
                Some("audio_tower.layers.0.attn.c_proj.bias"),
            ),
            (
                "audio_tower.layers.0.mlp.down_proj.bias",
                Some("audio_tower.layers.0.mlp.down_proj.bias"),
            ),
            ("audio_tower.norm.weight", Some("audio_tower.norm.weight")),
            (
                "multi_modal_projector.linear_1.weight",
                Some("multi_modal_projector.stacking.projection.weight"),
            ),
            (
                "multi_modal_projector.linear_2.weight",
                Some("multi_modal_projector.linear_2.weight"),
            ),
            (
                "language_model.model.embed_tokens.weight",
                Some("language_model.embed_tokens.weight"),
            ),
            (
                "language_model.model.layers.25.ada_rms_norm.linear1.weight",
                Some("language_model.layers.25.mlp_scale.linear1.weight"),
            ),
            (
                "language_model.model.layers.3.self_attn.k_proj.weight",
                Some("language_model.layers.3.attn.c_k.weight"),
            ),
            (
                "language_model.model.layers.3.post_attention_layernorm.weight",
                Some("language_model.layers.3.mlp_norm.weight"),
            ),
            (
                "language_model.model.norm.weight",
                Some("language_model.norm.weight"),
            ),
            ("lm_head.weight", None),
        ] {
            assert_eq!(rekey.apply(hf).as_deref(), bunsen, "{hf}");
        }
    }

    const CONFIG_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/voxtral/voxtral-mini-4b-realtime-2602.config.json"
    ));

    /// The published `config.json` is the published geometry.
    #[test]
    fn test_config_from_hf() {
        let config = voxtral_config_from_hf(CONFIG_JSON).unwrap();
        let expected = VoxtralRealtimeConfig::voxtral_mini_4b_realtime();
        assert_eq!(config.audio.n_layers(), expected.audio.n_layers());
        assert_eq!(config.audio.d_model(), expected.audio.d_model());
        assert_eq!(config.audio.head_dim(), expected.audio.head_dim());
        assert_eq!(config.audio.d_ff, expected.audio.d_ff);
        assert_eq!(config.audio.sliding_window(), Some(750));
        assert_eq!(config.audio.rope_theta, 1_000_000);
        assert_eq!(config.text.n_vocab(), expected.text.n_vocab());
        assert_eq!(config.text.n_layers(), expected.text.n_layers());
        assert_eq!(config.text.n_kv_heads(), expected.text.n_kv_heads());
        assert_eq!(config.text.head_dim(), expected.text.head_dim());
        assert_eq!(config.text.d_ff, expected.text.d_ff);
        assert_eq!(config.text.sliding_window(), Some(8192));
        assert_eq!(config.downsample(), 4);
        assert_eq!(config.mel_frames_per_position(), 8);

        let untied = CONFIG_JSON.replace(
            "\"tie_word_embeddings\": true",
            "\"tie_word_embeddings\": false",
        );
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("tie_word_embeddings")
            .assert_err(&voxtral_config_from_hf(&untied));
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("model_type")
            .assert_err(&voxtral_config_from_hf(
                &CONFIG_JSON.replace("\"voxtral_realtime\"", "\"voxtral\""),
            ));
    }
}
