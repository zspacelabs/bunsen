//! The state-dict names.

use crate::{
    blocks::transformers::decoders::causal::CausalDecoderBlock,
    burner::module::Rekey,
    kits::support::transformers::interop::llama_layer_keys,
};

/// `transformers`' Qwen3 decoder layer names onto a
/// [`CausalDecoderBlock`]'s,
/// relative to it: `input_layernorm` the attention's norm,
/// `post_attention_layernorm` the feed-forward's, `self_attn`'s
/// `{q,k,v,o}_proj` the attention's projections and its `{q,k}_norm` its
/// QK-norms, and the `SwiGLU`'s projections as they are.
pub fn text_layer_rekey() -> Rekey {
    CausalDecoderBlock::layer_rekey(&llama_layer_keys(
        "input_layernorm",
        "post_attention_layernorm",
    ))
}

/// `transformers`' `Qwen3ForCausalLM` state-dict names onto the module paths
/// of a [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel)'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
/// under `model.`, the layers by [`text_layer_rekey`], the embedding
/// (`embed_tokens`, the model's `embedding`) and the final norm; and
/// `lm_head.weight`, the head's projection. Anything else is dropped.
///
/// A tied model has no `lm_head` to load: a checkpoint's `lm_head.weight`
/// (`Qwen3-0.6B` carries a copy of the embedding there) is then unused, as
/// `transformers` ties over it under `tie_word_embeddings`.
///
/// Normalization parameters keep their `weight` spelling, which
/// [`Rekey::load`](crate::burner::module::Rekey::load) resolves to `gamma`.
pub fn qwen3_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount(
            "model.",
            "",
            Rekey::keep_rest()
                .mount_indexed("layers.", "stack.layers.", text_layer_rekey())
                .rename("embed_tokens.", "embedding.")
                .rename("norm.", "stack.norm."),
        )
        .exact("lm_head.weight", "head.proj.weight")
}
