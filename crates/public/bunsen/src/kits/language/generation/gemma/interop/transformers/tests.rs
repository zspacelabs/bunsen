use super::*;
use crate::{
    blocks::transformers::embedding::RopeScaling,
    errors::testing::ErrorMatcher,
    kits::language::generation::gemma::GemmaLayerKind,
};

#[test]
fn test_gemma_rekey() {
    for (post_norms, hf, bunsen) in [
        (true, "model.embed_tokens.weight", Some("embedding.weight")),
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
        assert_eq!(gemma_rekey(post_norms).apply(hf).as_deref(), bunsen, "{hf}");
    }
}

/// Every norm loads as `1 + w`; nothing else is transformed.
#[test]
fn test_gemma_rekey_transforms_the_norms() {
    let plan = gemma_rekey(true).plan([
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
    let c = gemma_config_from_str(GEMMA3_4B).unwrap();
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
    let c = gemma_config_from_str(
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

    let c = gemma_config_from_str(
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
    ErrorMatcher::unsupported()
        .message_contains("model_type")
        .assert_err(&gemma_config_from_str(r#"{"model_type": "llama"}"#));
    let bidirectional = GEMMA3_4B.replace(
        r#""final_logit_softcapping": null,"#,
        r#""final_logit_softcapping": null, "use_bidirectional_attention": true,"#,
    );
    ErrorMatcher::unsupported()
        .message_contains("use_bidirectional_attention")
        .assert_err(&gemma_config_from_str(&bidirectional));
    let yarn = GEMMA3_4B.replace(r#""rope_type": "linear""#, r#""rope_type": "yarn""#);
    ErrorMatcher::unsupported()
        .message_contains("yarn")
        .assert_err(&gemma_config_from_str(&yarn));
}
