//! The Llama line's decoder-layer key set.

use crate::blocks::transformers::decoders::causal::DecoderLayerKeys;

/// Llama's line (Qwen3, Mistral, Gemma): `self_attn` with `{q,k,v,o}_proj`,
/// under the layer's names for its two norms, the attention's and the
/// feed-forward's (`input_layernorm` and `post_attention_layernorm` in
/// Llama's own; `self_attn_layer_norm` and `final_layer_norm` in Voxtral's
/// audio tower).
pub fn llama_layer_keys(
    attn_norm: impl Into<String>,
    mlp_norm: impl Into<String>,
) -> DecoderLayerKeys {
    DecoderLayerKeys {
        attn_norm: attn_norm.into(),
        mlp_norm: mlp_norm.into(),
        post_norms: None,
        attn: "self_attn".to_string(),
        attn_projs: ["q_proj", "k_proj", "v_proj", "o_proj"].map(String::from),
    }
}
