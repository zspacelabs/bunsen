use super::*;
use crate::{
    blocks::transformers::models::language_decoder::LmDecoderMeta,
    burner::module::testing::assert_rekeys,
    errors::testing::ErrorMatcher,
    kits::speech::asr::voxtral::blocks::{
        VoxtralAudioEncoderMeta,
        VoxtralRealtimeConfig,
        VoxtralRealtimeMeta,
    },
};

#[test]
fn test_voxtral_realtime_rekey() {
    let rekey = voxtral_realtime_rekey();
    assert_rekeys(
        &rekey,
        [
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
                Some("audio_tower.stack.layers.31.attn_norm.weight"),
            ),
            (
                "audio_tower.layers.0.final_layer_norm.weight",
                Some("audio_tower.stack.layers.0.mlp_norm.weight"),
            ),
            (
                "audio_tower.layers.0.self_attn.o_proj.bias",
                Some("audio_tower.stack.layers.0.attn.c_proj.bias"),
            ),
            (
                "audio_tower.layers.0.mlp.down_proj.bias",
                Some("audio_tower.stack.layers.0.mlp.down_proj.bias"),
            ),
            (
                "audio_tower.norm.weight",
                Some("audio_tower.stack.norm.weight"),
            ),
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
                Some("language_model.lm.embedding.weight"),
            ),
            (
                "language_model.model.layers.25.ada_rms_norm.linear1.weight",
                Some("language_model.lm.stack.layers.25.mlp_scale.mlp.linear1.weight"),
            ),
            (
                "language_model.model.layers.3.self_attn.k_proj.weight",
                Some("language_model.lm.stack.layers.3.attn.c_k.weight"),
            ),
            (
                "language_model.model.layers.3.post_attention_layernorm.weight",
                Some("language_model.lm.stack.layers.3.mlp_norm.weight"),
            ),
            (
                "language_model.model.norm.weight",
                Some("language_model.lm.stack.norm.weight"),
            ),
            ("lm_head.weight", None),
        ],
    );
}

const CONFIG_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/voxtral/voxtral-mini-4b-realtime-2602.config.json"
));

/// The published `config.json` is the published geometry.
#[test]
fn test_voxtral_realtime_config() {
    let config = voxtral_realtime_config_from_str(CONFIG_JSON).unwrap();
    let expected = VoxtralRealtimeConfig::voxtral_mini_4b_realtime();
    assert_eq!(config.audio.n_layers(), expected.audio.n_layers());
    assert_eq!(config.audio.d_model(), expected.audio.d_model());
    assert_eq!(config.audio.head_dim(), expected.audio.head_dim());
    assert_eq!(config.audio.d_ff, expected.audio.d_ff);
    assert_eq!(config.audio.sliding_window(), Some(750));
    assert_eq!(config.audio.rope_theta, 1_000_000.0);
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
    ErrorMatcher::unsupported()
        .message_contains("tie_word_embeddings")
        .assert_err(&voxtral_realtime_config_from_str(&untied));
    ErrorMatcher::unsupported()
        .message_contains("model_type")
        .assert_err(&voxtral_realtime_config_from_str(
            &CONFIG_JSON.replace("\"voxtral_realtime\"", "\"voxtral\""),
        ));
}
