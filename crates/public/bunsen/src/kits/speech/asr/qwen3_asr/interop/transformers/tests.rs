use super::*;
use crate::{
    burner::module::testing::assert_rekeys,
    kits::speech::asr::qwen3_asr::{
        Qwen3AsrConfig,
        blocks::Qwen3AsrAudioEncoderMeta,
    },
    support::testing::assert_config_eq,
};

#[test]
fn test_qwen3_asr_rekey() {
    let rekey = qwen3_asr_rekey();
    assert_rekeys(
        &rekey,
        [
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
                Some("audio_tower.layers.3.attn.c_k.bias"),
            ),
            (
                "model.audio_tower.layers.3.self_attn.out_proj.weight",
                Some("audio_tower.layers.3.attn.c_proj.weight"),
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
                Some("multi_modal_projector.linear1.weight"),
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                Some("language_model.stack.layers.0.attn.c_proj.weight"),
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                Some("language_model.stack.layers.0.attn.q_norm.weight"),
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                Some("language_model.stack.layers.0.mlp.gate_proj.weight"),
            ),
            (
                "model.language_model.norm.weight",
                Some("language_model.stack.norm.weight"),
            ),
            (
                "model.language_model.embed_tokens.weight",
                Some("language_model.embedding.weight"),
            ),
            ("lm_head.weight", None),
        ],
    );
}

/// `Qwen3-ASR-0.6B-hf`'s `config.json` is the built-in geometry;
/// `Qwen3-ASR-1.7B-hf`'s is the larger model's; a model bunsen's is not
/// is refused by the field that says so.
#[test]
fn test_qwen3_asr_config_from_str() {
    use crate::kits::speech::asr::qwen3_asr::blocks::Qwen3AsrMeta;

    let small = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/qwen3_asr/qwen3-asr-0.6b-hf.config.json"
    ));
    let config = qwen3_asr_config_from_str(small).unwrap();
    assert_config_eq("the config", &config, &Qwen3AsrConfig::qwen3_asr_0_6b());

    let large = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/qwen3_asr/qwen3-asr-1.7b-hf.config.json"
    ));
    let config = qwen3_asr_config_from_str(large).unwrap();
    assert_config_eq("the config", &config, &Qwen3AsrConfig::qwen3_asr_1_7b());
    assert_eq!(config.d_audio(), 1024);
    assert_eq!(config.audio.n_layers(), 24);
    assert_eq!(config.d_text(), 2048);
    assert_eq!(config.text.d_ff, 6144);
    assert_eq!(config.n_vocab(), 151_936);

    let untied = small.replace(
        "\"tie_word_embeddings\": true",
        "\"tie_word_embeddings\": false",
    );
    crate::errors::testing::ErrorMatcher::unsupported()
        .message_contains("tie_word_embeddings")
        .assert_err(&qwen3_asr_config_from_str(&untied));
}
