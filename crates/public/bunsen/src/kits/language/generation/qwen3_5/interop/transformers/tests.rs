use super::*;
use crate::{
    blocks::transformers::attention::csa::AttentionOutputGate,
    burner::module::testing::assert_rekeys,
    errors::testing::ErrorMatcher,
    kits::language::generation::qwen3_5::Qwen3_5ContractConfig,
};

/// The checkpoints' names land on the model's canonical names, the
/// vision tower and the prediction head dropped.
#[test]
fn test_qwen3_5_rekey() {
    let rekey = qwen3_5_rekey(&Qwen3_5ContractConfig::qwen3_5_0_8b());
    assert_rekeys(
        &rekey,
        [
            (
                "model.language_model.embed_tokens.weight",
                Some("embedding.weight"),
            ),
            ("model.embed_tokens.weight", Some("embedding.weight")),
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                Some("stack.layers.0.attn.in_proj_qkv.weight"),
            ),
            (
                "model.language_model.layers.0.linear_attn.A_log",
                Some("stack.layers.0.attn.a_log"),
            ),
            (
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                Some("stack.layers.0.attn.conv1d.weight"),
            ),
            (
                "model.language_model.layers.0.linear_attn.norm.weight",
                Some("stack.layers.0.attn.norm.norm.weight"),
            ),
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                Some("stack.layers.3.attn.c_q.weight"),
            ),
            (
                "model.language_model.layers.3.self_attn.q_norm.weight",
                Some("stack.layers.3.attn.q_norm.weight"),
            ),
            (
                "model.language_model.layers.3.input_layernorm.weight",
                Some("stack.layers.3.attn_norm.weight"),
            ),
            (
                "model.language_model.layers.3.mlp.down_proj.weight",
                Some("stack.layers.3.mlp.down_proj.weight"),
            ),
            (
                "model.language_model.norm.weight",
                Some("stack.norm.weight"),
            ),
            ("lm_head.weight", Some("head.proj.weight")),
            ("model.visual.patch_embed.proj.weight", None),
            ("mtp.fc.weight", None),
        ],
    );
}

/// The `1 + w` norms are loaded as `1 + w`; the `DeltaNet`'s gated norm,
/// and everything else, as stored.
#[test]
fn test_zero_centered_norms() {
    let rekey = qwen3_5_rekey(&Qwen3_5ContractConfig::qwen3_5_0_8b());
    let transformed = |name: &str| {
        rekey
            .target(name)
            .is_some_and(|target| !target.transform_names.is_empty())
    };
    for name in [
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "model.language_model.layers.3.self_attn.k_norm.weight",
        "model.language_model.norm.weight",
    ] {
        assert!(transformed(name), "{name}: `1 + w`");
    }
    for name in [
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.3.self_attn.q_proj.weight",
    ] {
        assert!(!transformed(name), "{name}: as stored");
    }

    let as_stored =
        qwen3_5_rekey(&Qwen3_5ContractConfig::qwen3_5_0_8b().with_zero_centered_norms(false));
    for name in [
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "model.language_model.norm.weight",
    ] {
        assert!(
            as_stored
                .target(name)
                .is_some_and(|target| target.transform_names.is_empty()),
            "{name}: as stored without zero-centered norms"
        );
    }
}

const QWEN3_5_0_8B: &str = r#"{
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5", "tie_word_embeddings": true,
    "text_config": {
        "model_type": "qwen3_5_text", "attention_bias": false,
        "attn_output_gate": true, "full_attention_interval": 4, "head_dim": 256,
        "hidden_act": "silu", "hidden_size": 1024, "intermediate_size": 3584,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
        "linear_num_key_heads": 16, "linear_num_value_heads": 16,
        "linear_value_head_dim": 128, "max_position_embeddings": 262144,
        "mlp_only_layers": [], "num_attention_heads": 8, "num_hidden_layers": 24,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "tie_word_embeddings": true,
        "vocab_size": 248320,
        "layer_types": [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention"
        ],
        "rope_parameters": {
            "mrope_interleaved": true, "mrope_section": [11, 11, 10],
            "rope_type": "default", "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
        }
    },
    "vision_config": {"depth": 12, "hidden_size": 768}
}"#;

/// `Qwen3.5-0.8B`'s `config.json`, as the Hub serves it.
#[test]
fn test_reads_the_0_8b() {
    let config = qwen3_5_config_from_str(QWEN3_5_0_8B).unwrap();
    assert_eq!(
        (
            config.n_vocab,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.d_ff
        ),
        (248_320, 1024, 24, 8, 2, 256, 3584)
    );
    assert_eq!(
        (
            config.linear_key_heads,
            config.linear_value_heads,
            config.linear_key_dim,
            config.linear_value_dim,
            config.linear_conv_kernel
        ),
        (16, 16, 128, 128, 4)
    );
    assert_eq!(config.rope_theta, 10_000_000.0);
    assert_eq!(config.rotary_dim(), 64);
    assert!(config.tied_head && config.attn_output_gate == Some(AttentionOutputGate::Elementwise));
    assert_eq!(
        config.layer_kinds,
        Qwen3_5ContractConfig::every_nth_full(24, 4)
    );
}

/// A file that omits the layer types, the tie and the rotary share takes
/// `transformers`' defaults: every `full_attention_interval`-th layer
/// full, an untied head, a quarter of each head rotated.
#[test]
fn test_defaults() {
    let json = QWEN3_5_0_8B
        .replace(r#""tie_word_embeddings": true,"#, "")
        .replace(
            r#""full_attention_interval": 4,"#,
            r#""full_attention_interval": 3,"#,
        )
        .replace(r#", "partial_rotary_factor": 0.25"#, "");
    let json = json[..json.find(r#""layer_types""#).unwrap()].to_string()
        + &json[json.find(r#""rope_parameters""#).unwrap()..];
    let config = qwen3_5_config_from_str(&json).unwrap();
    assert!(!config.tied_head);
    assert_eq!(config.rotary_dim(), 64);
    assert_eq!(
        config.layer_kinds,
        Qwen3_5ContractConfig::every_nth_full(24, 3)
    );
}

/// An f32 delta-rule state, as the release's files state it, is read.
#[test]
fn test_f32_state_accepted() {
    let json = QWEN3_5_0_8B.replace(
        r#""model_type": "qwen3_5_text", "attention_bias": false,"#,
        r#""model_type": "qwen3_5_text", "attention_bias": false,
           "mamba_ssm_dtype": "float32","#,
    );
    assert_ne!(json, QWEN3_5_0_8B);
    qwen3_5_config_from_str(&json).unwrap();
}

/// What bunsen's Qwen3.5 is not is refused, naming the field.
#[test]
fn test_refusals() {
    for (from, to, field) in [
        (
            r#""model_type": "qwen3_5","#,
            r#""model_type": "qwen3_5_moe","#,
            "model_type",
        ),
        (
            r#""hidden_act": "silu""#,
            r#""hidden_act": "gelu""#,
            "hidden_act",
        ),
        (
            r#""mrope_section": [11, 11, 10]"#,
            r#""mrope_section": [11, 11, 11]"#,
            "mrope_section",
        ),
        (
            r#""mlp_only_layers": []"#,
            r#""mlp_only_layers": [2]"#,
            "mlp_only_layers",
        ),
        (
            r#""rope_type": "default""#,
            r#""rope_type": "yarn""#,
            "rope_type",
        ),
        (
            r#"["Qwen3_5ForConditionalGeneration"]"#,
            r#"["Qwen3_5ForCausalLM"]"#,
            "architectures",
        ),
        (
            r#""model_type": "qwen3_5_text", "attention_bias": false,"#,
            r#""model_type": "qwen3_5_text", "attention_bias": false,
               "mamba_ssm_dtype": "bfloat16","#,
            "mamba_ssm_dtype",
        ),
    ] {
        let json = QWEN3_5_0_8B.replace(from, to);
        assert_ne!(json, QWEN3_5_0_8B, "{from}");
        ErrorMatcher::unsupported()
            .message_contains(field)
            .assert_err(&qwen3_5_config_from_str(&json));
    }
}
