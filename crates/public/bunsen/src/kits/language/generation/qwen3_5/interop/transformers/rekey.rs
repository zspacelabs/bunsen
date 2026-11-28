//! The state-dict names.

use crate::{
    blocks::transformers::decoders::causal::CausalDecoderBlock,
    burner::module::{
        Rekey,
        Select,
        TensorLoadTransform,
    },
    kits::{
        language::generation::qwen3_5::Qwen3_5ContractConfig,
        support::transformers::interop::llama_layer_keys,
    },
};

/// The prefixes a Qwen3.5's text model sits under: a
/// `Qwen3_5ForConditionalGeneration`'s (every published checkpoint), and a
/// `Qwen3_5ForCausalLM`'s, in the order they are tried.
pub const QWEN3_5_TEXT_PREFIXES: [&str; 2] = ["model.language_model.", "model."];

/// `transformers`' Qwen3.5 decoder layer names onto a [`CausalDecoderBlock`]'s,
/// relative to it: Llama's for the norms (`input_layernorm`,
/// `post_attention_layernorm`) and a full layer's `self_attn`, as
/// [`CausalDecoderBlock::layer_rekey`] maps them; a linear layer's
/// `linear_attn` the mixer's fields under their own names, but `A_log`,
/// which is `a_log`, and the gated norm's gain, which is its norm's
/// (`norm.norm.weight`); and the `SwiGLU`'s projections as they are.
pub fn text_layer_rekey() -> Rekey {
    CausalDecoderBlock::layer_rekey(&llama_layer_keys(
        "input_layernorm",
        "post_attention_layernorm",
    ))
    .rename("linear_attn.A_log", "attn.a_log")
    .rename("linear_attn.norm.", "attn.norm.norm.")
    .rename("linear_attn.", "attn.")
}

/// `transformers`' Qwen3.5 state-dict names onto the module paths of a
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel)'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
/// under either of [`QWEN3_5_TEXT_PREFIXES`], the layers by
/// [`text_layer_rekey`], the embedding (`embed_tokens`, the model's
/// `embedding`) and the final norm; `lm_head.weight`, the head's projection;
/// when `config`'s [`zero_centered_norms`](Qwen3_5ContractConfig::zero_centered_norms)
/// says so, every `1 + w` norm's gain, stored as `w`, loaded as `1 + w`
/// ([`TensorLoadTransform::zero_centered_gain`]). Anything else is dropped: the
/// vision tower, and the multi-token-prediction head (`mtp.`).
///
/// A tied model has no `lm_head` to load: a checkpoint's `lm_head.weight`
/// is then unused, as `transformers` ties over it.
pub fn qwen3_5_rekey(config: &Qwen3_5ContractConfig) -> Rekey {
    let zero_centered: &[&str] = if config.zero_centered_norms {
        &ZERO_CENTERED_LAYER_NORMS
    } else {
        &[]
    };
    let layer = zero_centered
        .iter()
        .fold(text_layer_rekey(), |rekey, norm| {
            rekey.transform(
                Select::suffix(*norm),
                TensorLoadTransform::zero_centered_gain(),
            )
        });
    let text = Rekey::keep_rest()
        .mount("visual.", "", Rekey::drop_rest())
        .mount_indexed("layers.", "stack.layers.", layer)
        .rename("embed_tokens.", "embedding.")
        .rename("norm.", "stack.norm.");
    let text = if config.zero_centered_norms {
        text.transform(
            Select::suffix("stack.norm.weight"),
            TensorLoadTransform::zero_centered_gain(),
        )
    } else {
        text
    };
    QWEN3_5_TEXT_PREFIXES
        .iter()
        .fold(Rekey::drop_rest(), |rekey, prefix| {
            rekey.mount(*prefix, "", text.clone())
        })
        .exact("lm_head.weight", "head.proj.weight")
}

/// The `1 + w` norms, by their paths under a layer of the canonical scheme:
/// the two sublayers' and the attention's QK-norms. The `DeltaNet`'s gated norm
/// (`attn.norm.norm`) applies its gain as stored.
const ZERO_CENTERED_LAYER_NORMS: [&str; 4] = [
    "attn_norm.weight",
    "mlp_norm.weight",
    "attn.q_norm.weight",
    "attn.k_norm.weight",
];
