use super::*;
use crate::{
    burner::module::testing::assert_rekeys,
    errors::testing::ErrorMatcher,
};

/// The checkpoint's names land on the model's; the buffer is dropped.
#[test]
fn test_laya_rekey() {
    let rekey = laya_rekey();
    assert_rekeys(
        &rekey,
        [
            (
                "encoder.layers.3.attn.Wq.weight",
                Some("encoder.stack.layers.3.attn.c_q.weight"),
            ),
            (
                "encoder.final_norm.weight",
                Some("encoder.stack.norm.weight"),
            ),
            (
                "head.layers.1.self_attn.q_proj.bias",
                Some("head.layers.1.attn.c_q.bias"),
            ),
            (
                "head.layers.0.self_attn.out_proj.weight",
                Some("head.layers.0.attn.c_proj.weight"),
            ),
            (
                "head.layers.0.norm1.weight",
                Some("head.layers.0.attn_norm.weight"),
            ),
            (
                "head.layers.0.norm2.bias",
                Some("head.layers.0.mlp_norm.bias"),
            ),
            (
                "head.layers.1.linear2.weight",
                Some("head.layers.1.mlp.linear2.weight"),
            ),
            ("type_emb.weight", Some("type_emb.weight")),
            ("scorer.0.weight", Some("scorer.norm.weight")),
            ("scorer.3.bias", Some("scorer.mlp.linear2.bias")),
            ("act_head.2.weight", Some("act_head.linear2.weight")),
            ("temperature", None),
        ],
    );
}

const JSON: &str = r#"{
    "encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "max_len": 512,
    "head_max_len": 192, "act_costs": {"escalate": 0.5}, "amp_dtype": "bf16",
    "temperature": [1.5, 1.25, 2.0],
    "temperature_by_options": {"choice:3-5": 1.75, "noul:2": 1.9},
    "training": {"updates": 7313}
}"#;

#[test]
fn test_parse() {
    let config = LayaAgentMirror::from_json(JSON).unwrap().agent_config();
    assert_eq!(
        (config.head_layers, config.max_len, config.head_max_len),
        (2, 512, 192)
    );
    assert_eq!(config.n_act(), 2);
    ErrorMatcher::invalid_resource().assert_err(&LayaAgentMirror::from_json("{}"));
}
