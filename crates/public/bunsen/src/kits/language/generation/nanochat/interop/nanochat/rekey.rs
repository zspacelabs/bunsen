//! The state-dict names.

use crate::burner::module::Rekey;

/// Upstream's state-dict names onto a
/// [`NanoChatGpt`](crate::kits::language::generation::nanochat::NanoChatGpt)'s module
/// paths: the `transformer.` prefix dropped; onto its language model's (`lm.`),
/// the embedding `wte` its `lm.embedding`, the blocks `h.{i}` the stack's
/// `lm.stack.layers.{i}` (the feed-forward's `c_fc` / `c_proj` its
/// `linear1` / `linear2`, the value gate `ve_gate` its `value_gate`), and the
/// head `lm_head` its projection, `lm.head.proj`; everything else
/// (`attn.c_q`, the lambdas, `value_embeds.{i}`, `smear_gate`) bunsen names
/// as upstream does.
///
/// Unlike the wrappers' rekeys, it names the model's paths, not its
/// language model's alone: the 2026 additions sit beside the language model.
pub fn nanochat_gpt_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("lm_head.", "lm.head.proj.")
        .mount(
            "transformer.",
            "",
            Rekey::keep_rest()
                .rename("wte.", "lm.embedding.")
                .mount_indexed(
                    "h.",
                    "lm.stack.layers.",
                    Rekey::keep_rest()
                        .rename("mlp.c_fc.", "mlp.linear1.")
                        .rename("mlp.c_proj.", "mlp.linear2.")
                        .rename("attn.ve_gate.", "attn.value_gate."),
                ),
        )
}
