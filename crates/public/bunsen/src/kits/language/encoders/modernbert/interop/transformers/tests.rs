use super::*;
use crate::{
    burner::module::{
        Rekey,
        testing::assert_rekeys,
    },
    errors::testing::ErrorMatcher,
    kits::language::encoders::modernbert::ModernBertLayerKind,
};

/// A layer's names, once split, land on the block's: the projections,
/// the norms, the `GeGLU`'s parts.
#[test]
fn test_modernbert_rekey() {
    let rekey = Rekey::drop_rest().mount("model.", "", modernbert_rekey());
    assert_rekeys(
        &rekey,
        [
            (
                "model.embeddings.tok_embeddings.weight",
                Some("embeddings.weight"),
            ),
            (
                "model.embeddings.norm.weight",
                Some("embeddings.norm.weight"),
            ),
            (
                "model.layers.1.attn_norm.weight",
                Some("stack.layers.1.attn_norm.weight"),
            ),
            (
                "model.layers.0.attn.Wq.weight",
                Some("stack.layers.0.attn.c_q.weight"),
            ),
            (
                "model.layers.0.attn.Wk.weight",
                Some("stack.layers.0.attn.c_k.weight"),
            ),
            (
                "model.layers.0.attn.Wv.weight",
                Some("stack.layers.0.attn.c_v.weight"),
            ),
            (
                "model.layers.0.attn.Wo.weight",
                Some("stack.layers.0.attn.c_proj.weight"),
            ),
            (
                "model.layers.0.mlp_norm.weight",
                Some("stack.layers.0.mlp_norm.weight"),
            ),
            (
                "model.layers.2.mlp.Wi_gate.weight",
                Some("stack.layers.2.mlp.gate_proj.weight"),
            ),
            (
                "model.layers.2.mlp.Wi_up.weight",
                Some("stack.layers.2.mlp.up_proj.weight"),
            ),
            (
                "model.layers.2.mlp.Wo.weight",
                Some("stack.layers.2.mlp.down_proj.weight"),
            ),
            ("model.final_norm.weight", Some("stack.norm.weight")),
            ("head.dense.weight", None),
            ("model.head.norm.weight", None),
        ],
    );
}

/// Laya's encoder config: 28 layers, every third global from the
/// first, its `rope_parameters` per kind.
#[test]
fn test_config_from_transformers() {
    let json = r#"{
        "model_type": "modernbert", "vocab_size": 50368, "hidden_size": 1024,
        "num_hidden_layers": 6, "num_attention_heads": 16, "intermediate_size": 2624,
        "hidden_activation": "gelu", "norm_eps": 1e-05, "norm_bias": false,
        "attention_bias": false, "mlp_bias": false, "local_attention": 128,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention",
                        "full_attention", "sliding_attention", "sliding_attention"],
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        }
    }"#;
    let config = modernbert_config_from_str(json).unwrap();
    assert_eq!(
        (
            config.n_vocab,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff
        ),
        (50368, 1024, 6, 16, 2624)
    );
    assert_eq!(config.local_radius, 64);
    assert_eq!(
        (config.global_rope_theta, config.local_rope_theta),
        (160000.0, 10000.0)
    );
    use ModernBertLayerKind::*;
    assert_eq!(
        config.layer_kinds,
        [Global, Local, Local, Global, Local, Local]
    );

    // The older spelling: a pattern and two thetas.
    let older = r#"{
        "model_type": "modernbert", "vocab_size": 8, "hidden_size": 8,
        "num_hidden_layers": 4, "num_attention_heads": 2, "intermediate_size": 12,
        "global_attn_every_n_layers": 2, "global_rope_theta": 1000.0,
        "local_rope_theta": 100.0, "local_attention": 8
    }"#;
    let config = modernbert_config_from_str(older).unwrap();
    assert_eq!(config.layer_kinds, [Global, Local, Global, Local]);
    assert_eq!(
        (
            config.global_rope_theta,
            config.local_rope_theta,
            config.local_radius
        ),
        (1000.0, 100.0, 4)
    );

    let biased = json.replace(r#""mlp_bias": false"#, r#""mlp_bias": true"#);
    ErrorMatcher::unsupported().assert_err(&modernbert_config_from_str(&biased));
}
