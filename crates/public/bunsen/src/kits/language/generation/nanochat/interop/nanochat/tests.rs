use super::*;
use crate::{
    burner::module::testing::assert_rekeys,
    errors::testing::ErrorMatcher,
};

fn meta(window: Option<&str>) -> NanoChatMetaMirror {
    NanoChatMetaMirror {
        step: Some(1),
        model_config: NanoChatModelConfigMirror {
            sequence_len: 2048,
            vocab_size: 65536,
            n_layer: 20,
            n_head: 10,
            n_kv_head: 10,
            n_embd: 1280,
            window_pattern: window.map(str::to_string),
        },
    }
}

/// The pinned checkpoints' metas parse, the extra keys some uploads add
/// ignored.
#[test]
fn test_meta_from_json() {
    let m = NanoChatMetaMirror::from_json(
        r#"{"step": 650, "val_loss": 1.07, "model_config": {"sequence_len": 2048,
        "vocab_size": 65536, "n_layer": 20, "n_head": 10, "n_kv_head": 10, "n_embd": 1280,
        "bos_token": "<|bos|>", "chat_template": "..."}}"#,
    )
    .unwrap();
    assert_eq!(m.step, Some(650));
    assert_eq!(
        m,
        NanoChatMetaMirror {
            step: Some(650),
            ..meta(None)
        }
    );

    let err = NanoChatMetaMirror::from_json(r#"{"step": 1}"#).unwrap_err();
    ErrorMatcher::invalid_resource().assert(&err);
}

/// The generation follows the state dict and the meta together, and a
/// checkpoint between the two is refused.
#[test]
fn test_generation() {
    let v2025 = ["transformer.wte.weight", "lm_head.weight"];
    assert_eq!(
        meta(None).generation(&v2025).unwrap(),
        NanoChatGeneration::Upstream2025
    );
    let mut v2026: Vec<&str> = v2025.to_vec();
    v2026.extend(UPSTREAM_2026_MARKERS);
    v2026.push("value_embeds.1.weight");
    assert_eq!(
        meta(Some("L")).generation(&v2026).unwrap(),
        NanoChatGeneration::Upstream2026
    );

    let unsupported =
        ErrorMatcher::unsupported().message_contains("between upstream's 2025 and 2026 models");
    // The meta says 2026, the state dict 2025; and the other way round.
    unsupported.assert(&meta(Some("L")).generation(&v2025).unwrap_err());
    unsupported.assert(&meta(None).generation(&v2026).unwrap_err());
    // Some 2026 parts and not others: an intermediate upstream commit.
    let partial = ["transformer.wte.weight", "resid_lambdas", "x0_lambdas"];
    unsupported.assert(&meta(Some("L")).generation(&partial).unwrap_err());
}

/// The config takes the meta's geometry, the rotary table ten times its
/// context, and the generation's options.
#[test]
fn test_config() {
    let c = meta(None).gpt_config(NanoChatGeneration::Upstream2025);
    assert_eq!(
        (
            c.init_seq_len,
            c.max_seq_len_factor,
            c.vocab_size,
            c.n_layer,
            c.n_embed
        ),
        (2048, 10, 65536, 20, 1280)
    );
    assert!(!c.residual_lambdas && c.window_pattern.is_none() && c.rope_base == 10000.0);

    let c = meta(Some("SSSL")).gpt_config(NanoChatGeneration::Upstream2026);
    assert_eq!(c.window_pattern.as_deref(), Some("SSSL"));
    assert!(c.residual_lambdas && c.value_embeds && c.backout && c.rope_base == 100000.0);
}

/// Upstream's names land where bunsen keeps each part.
#[test]
fn test_nanochat_gpt_rekey() {
    let r = nanochat_gpt_rekey();
    assert_rekeys(
        &r,
        [
            ("transformer.wte.weight", "lm.embedding.weight"),
            (
                "transformer.h.3.attn.c_q.weight",
                "lm.stack.layers.3.attn.c_q.weight",
            ),
            (
                "transformer.h.3.attn.c_proj.weight",
                "lm.stack.layers.3.attn.c_proj.weight",
            ),
            (
                "transformer.h.0.mlp.c_fc.weight",
                "lm.stack.layers.0.mlp.linear1.weight",
            ),
            (
                "transformer.h.0.mlp.c_proj.weight",
                "lm.stack.layers.0.mlp.linear2.weight",
            ),
            (
                "transformer.h.17.attn.ve_gate.weight",
                "lm.stack.layers.17.attn.value_gate.weight",
            ),
            ("lm_head.weight", "lm.head.proj.weight"),
            ("value_embeds.17.weight", "value_embeds.17.weight"),
            ("smear_gate.weight", "smear_gate.weight"),
            ("resid_lambdas", "resid_lambdas"),
        ],
    );
}
