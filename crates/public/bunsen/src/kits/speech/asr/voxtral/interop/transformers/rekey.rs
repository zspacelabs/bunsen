//! The state-dict names.

use crate::{
    blocks::{
        conv::ConvSeq1d,
        transformers::decoders::causal::CausalDecoderBlock,
    },
    burner::module::Rekey,
    kits::{
        language::generation::qwen3::interop::transformers::text_layer_rekey,
        support::transformers::interop::llama_layer_keys,
    },
};

/// `transformers`' Voxtral Realtime audio-encoder layer names onto a
/// [`CausalDecoderBlock`]'s,
/// relative to it: `self_attn_layer_norm` the attention's norm,
/// `final_layer_norm` the feed-forward's, `self_attn`'s `{q,k,v,o}_proj` the
/// attention's projections, and the `SwiGLU`'s projections as they are.
pub fn audio_layer_rekey() -> Rekey {
    CausalDecoderBlock::layer_rekey(&llama_layer_keys(
        "self_attn_layer_norm",
        "final_layer_norm",
    ))
}

/// `transformers`' Voxtral Realtime text-layer names onto a
/// [`CausalDecoderBlock`]'s:
/// Mistral's, which are Qwen3's ([`text_layer_rekey`]), and `ada_rms_norm`
/// (its `linear1` / `linear2`) the MLP of the block's
/// [`mlp_scale`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock::mlp_scale).
pub fn text_layer_rekey_voxtral() -> Rekey {
    text_layer_rekey().rename("ada_rms_norm.", "mlp_scale.mlp.")
}

/// The names of `mistralai/Voxtral-Mini-4B-Realtime-2602`'s
/// `model.safetensors` onto a
/// [`VoxtralRealtime`](crate::kits::speech::asr::voxtral::VoxtralRealtime)'s
/// module paths: the audio tower's `embedder.conv1` and `conv2` onto its stem's
/// 0-based [`ConvSeq1d`] blocks ([`ConvSeq1d::one_based_rekey`]), its layers by
/// [`audio_layer_rekey`]; the projector's `linear_1` onto its stacking's
/// projection; the text model, under `language_model.model.`, onto its
/// language model (`language_model.lm.`): its layers by
/// [`text_layer_rekey_voxtral`], its embedding and final norm; and everything
/// else as it is. The head is
/// tied, so nothing else is read.
///
/// Normalization parameters keep their `weight` spelling, which
/// [`Rekey::load`](crate::burner::module::Rekey::load) resolves to `gamma`.
pub fn voxtral_realtime_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount(
            "audio_tower.",
            "audio_tower.",
            Rekey::keep_rest()
                .mount("embedder.conv", "stem.", ConvSeq1d::one_based_rekey())
                .mount_indexed("layers.", "stack.layers.", audio_layer_rekey())
                .rename("norm.", "stack.norm."),
        )
        .mount(
            "multi_modal_projector.",
            "multi_modal_projector.",
            Rekey::keep_rest().rename("linear_1.", "stacking.projection."),
        )
        .mount(
            "language_model.model.",
            "language_model.lm.",
            Rekey::keep_rest()
                .mount_indexed("layers.", "stack.layers.", text_layer_rekey_voxtral())
                .rename("embed_tokens.", "embedding.")
                .rename("norm.", "stack.norm."),
        )
}
