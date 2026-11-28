//! The state-dict names.

use crate::{
    blocks::transformers::decoders::causal::CausalDecoderBlock,
    burner::module::{
        Rekey,
        Select,
        TensorLoadTransform,
    },
    kits::support::transformers::interop::llama_layer_keys,
};

/// The prefixes a Gemma's text model sits under: `transformers` 5's
/// `Gemma3ForConditionalGeneration`, its 4.x spelling, and a causal LM's
/// (`GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`), in the
/// order they are tried.
pub const GEMMA_TEXT_PREFIXES: [&str; 3] =
    ["model.language_model.", "language_model.model.", "model."];

/// `transformers`' Gemma decoder layer names onto a [`CausalDecoderBlock`]'s,
/// relative to it. With `post_norms` (Gemma 2 and 3), the four norms of the
/// sandwich: `input_layernorm` and `post_attention_layernorm` the
/// attention's, `pre_feedforward_layernorm` and `post_feedforward_layernorm`
/// the feed-forward's. Without (Gemma 1), Llama's two. `self_attn`'s
/// projections and QK-norms, and the `GeGLU`'s projections, as
/// [`CausalDecoderBlock::layer_rekey`] maps them.
pub fn text_layer_rekey(post_norms: bool) -> Rekey {
    if post_norms {
        CausalDecoderBlock::layer_rekey(
            &llama_layer_keys("input_layernorm", "pre_feedforward_layernorm")
                .with_post_norms("post_attention_layernorm", "post_feedforward_layernorm"),
        )
    } else {
        CausalDecoderBlock::layer_rekey(&llama_layer_keys(
            "input_layernorm",
            "post_attention_layernorm",
        ))
    }
}

/// `transformers`' Gemma state-dict names onto the module paths of a
/// [`GemmaModel`](crate::kits::language::generation::gemma::GemmaModel)'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
/// under any of [`GEMMA_TEXT_PREFIXES`], the layers by [`text_layer_rekey`],
/// the embedding (`embed_tokens`, the model's `embedding`) and the final norm;
/// every norm's gain, stored as `w` for `1 + w`, loaded as `1 + w`
/// ([`TensorLoadTransform::zero_centered_gain`]). Anything else is dropped: a
/// tied head's copy, a vision tower and its projector.
pub fn gemma_rekey(post_norms: bool) -> Rekey {
    let text = || {
        Rekey::keep_rest()
            .mount_indexed("layers.", "stack.layers.", text_layer_rekey(post_norms))
            .rename("embed_tokens.", "embedding.")
            .rename("norm.", "stack.norm.")
    };
    GEMMA_TEXT_PREFIXES
        .iter()
        .fold(Rekey::drop_rest(), |rekey, prefix| {
            rekey.mount(*prefix, "", text())
        })
        .transform(
            Select::suffix("norm.weight"),
            TensorLoadTransform::zero_centered_gain(),
        )
}
