//! The layer key sets `transformers` models share.

#[cfg(doc)]
use crate::blocks::transformers::encoders::residual::ResidualEncoderAttentionBlock;
use crate::{
    blocks::transformers::decoders::causal::DecoderLayerKeys,
    burner::module::Rekey,
};

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

/// `transformers`' Whisper encoder layer names (Whisper's own, and
/// Qwen3-ASR's audio tower's) onto a
/// [`ResidualEncoderAttentionBlock`]'s, relative to it:
/// `self_attn`'s `{q,k,v,out}_proj` the attention,
/// `self_attn_layer_norm` and `final_layer_norm` the norms, `fc{1,2}`
/// the MLP; everything else as it is.
pub fn whisper_encoder_layer_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("self_attn.q_proj.", "attn.query.")
        .rename("self_attn.k_proj.", "attn.key.")
        .rename("self_attn.v_proj.", "attn.value.")
        .rename("self_attn.out_proj.", "attn.output.")
        .rename("self_attn_layer_norm.", "attn_norm.")
        .rename("final_layer_norm.", "mlp_norm.")
        .rename("fc1.", "mlp.linear1.")
        .rename("fc2.", "mlp.linear2.")
}

/// `transformers`' Whisper decoder layer names onto a
/// cross-attention decoder block's,
/// relative to it: `encoder_attn`'s `{q,k,v,out}_proj` the
/// cross-attention and `encoder_attn_layer_norm` its norm; the rest as
/// an encoder layer's
/// ([`whisper_encoder_layer_rekey`]).
pub fn whisper_decoder_layer_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("encoder_attn.q_proj.", "cross_attn.query.")
        .rename("encoder_attn.k_proj.", "cross_attn.key.")
        .rename("encoder_attn.v_proj.", "cross_attn.value.")
        .rename("encoder_attn.out_proj.", "cross_attn.output.")
        .rename("encoder_attn_layer_norm.", "cross_attn_norm.")
        .mount("", "", whisper_encoder_layer_rekey())
}
