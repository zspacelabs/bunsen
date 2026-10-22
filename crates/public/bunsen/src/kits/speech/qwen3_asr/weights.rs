//! Loading `transformers`-named tensors into a [`Qwen3Asr`].
use std::sync::LazyLock;

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
        transformers::decoders::causal::CausalDecoderBlock,
    },
    burner::module::Rekey,
    data::pretrained::{
        HfConfigJson,
        NamedTensorsApplied,
        apply_named_tensors_as,
    },
    errors::BunsenResult,
    kits::gpts::qwen3::{
        Qwen3Config,
        Qwen3Meta,
    },
};

/// The bunsen module path of one `transformers`
/// `Qwen3ASRForConditionalGeneration` state-dict name, or `None` for an entry
/// that is not a parameter of its own (the tied `lm_head.weight`).
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`apply_named_tensors_as`] resolves to `gamma` / `beta`.
pub fn hf_to_bunsen_path(name: &str) -> Option<String> {
    let name = name.strip_prefix("model.")?;
    const RENAMES: &[(&str, &str)] = &[
        // The audio tower's layers are the shared encoder block's.
        (".self_attn.q_proj.", ".attn.query."),
        (".self_attn.k_proj.", ".attn.key."),
        (".self_attn.v_proj.", ".attn.value."),
        (".self_attn.out_proj.", ".attn.output."),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".final_layer_norm.", ".mlp_ln."),
        (".fc1.", ".mlp.linear1."),
        (".fc2.", ".mlp.linear2."),
    ];
    // The text model's layers are the shared causal decoder layer's.
    static TEXT_LAYERS: LazyLock<Rekey> = LazyLock::new(|| {
        Rekey::drop_rest().mount_indexed(
            "language_model.layers.",
            "language_model.layers.",
            CausalDecoderBlock::transformers_rekey(),
        )
    });
    if name.starts_with("language_model.layers.") {
        return TEXT_LAYERS.apply(name);
    }
    // The stem's 1-based `conv2d{n}` are the 0-based blocks of a `ConvSeq2d`.
    if let Some(rest) = name.strip_prefix("audio_tower.conv2d")
        && let Some((n, param)) = rest.split_once('.')
        && let Ok(n @ 1..) = n.parse::<usize>()
    {
        let conv = ConvSeq2d::block_conv_path("audio_tower.stem", n - 1);
        return Some(format!("{conv}.{param}"));
    }
    let renames = if name.starts_with("audio_tower.") {
        RENAMES
    } else {
        &[]
    };
    let mut path = name.to_string();
    for (from, to) in renames {
        if path.contains(from) {
            path = path.replacen(from, to, 1);
            break;
        }
    }
    Some(path)
}

/// Loads tensors named as `transformers`' `Qwen3ASRForConditionalGeneration`
/// state dict into `model`: [`apply_named_tensors_as`] under
/// [`hf_to_bunsen_path`], every floating tensor cast to the dtype `model` was
/// built in. The checkpoints ship `bf16`, which not every backend computes
/// in.
///
/// # Errors
/// As [`apply_named_tensors_as`].
pub fn load_hf_tensors(
    model: &mut Qwen3Asr,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let dtype = model.multi_modal_projector.linear_2.weight.val().dtype();
    apply_named_tensors_as(model, tensors, hf_to_bunsen_path, Some(dtype))
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

    #[test]
    fn test_hf_to_bunsen_path() {
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
                Some("audio_tower.layers.3.mlp_ln.weight"),
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
            assert_eq!(hf_to_bunsen_path(hf).as_deref(), bunsen, "{hf}");
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
