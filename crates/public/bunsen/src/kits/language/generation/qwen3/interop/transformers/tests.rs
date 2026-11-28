use super::*;
use crate::{
    burner::module::testing::assert_rekeys,
    errors::testing::ErrorMatcher,
    kits::language::generation::qwen3::Qwen3Config,
    support::testing::assert_config_eq,
};

/// `transformers`' Qwen3 decoder-layer names, onto the block's own.
#[test]
fn test_text_layer_rekey() {
    let rekey = text_layer_rekey();
    assert_rekeys(
        &rekey,
        [
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
        ],
    );
}

#[test]
fn test_qwen3_rekey() {
    let rekey = qwen3_rekey();
    assert_rekeys(
        &rekey,
        [
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
        ],
    );
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
fn test_qwen3_config_from_str() {
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
        let config = qwen3_config_from_str(json).unwrap();
        assert_config_eq(name, &config, &expected);
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
        ErrorMatcher::unsupported()
            .message_contains(field)
            .assert_err(&qwen3_config_from_str(&small.replace(from, to)));
    }

    // A window in force is refused; `use_sliding_window: false` switches
    // a stated one off, as `transformers` reads it.
    let windowed = small.replace("\"sliding_window\": null", "\"sliding_window\": 4096");
    assert!(qwen3_config_from_str(&windowed).is_ok(), "switched off");
    ErrorMatcher::unsupported()
        .message_contains("sliding_window")
        .assert_err(&qwen3_config_from_str(&windowed.replace(
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
        let config = qwen3_config_from_str(&json).unwrap();
        assert!(!config.tied_head);
        assert_eq!(config.d_model, 1024, "the rest as it was");
    }
}
