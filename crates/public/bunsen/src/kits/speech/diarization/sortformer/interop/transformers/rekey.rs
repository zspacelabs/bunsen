//! The state-dict names.

use crate::{
    blocks::signals::{
        FeatureStackingMeta,
        FeatureStackingOrder,
    },
    burner::module::Rekey,
};

/// `transformers`' `Nemotron3DiarizationEmbedder` names onto a
/// [`FeatureStacking`](crate::blocks::signals::FeatureStacking)'s, relative
/// to it: `projection.*`, its weight permuted when `stacking` stacks
/// otherwise than `transformers` does (frames outer). Anything else is
/// dropped.
pub fn embedder_rekey(stacking: &impl FeatureStackingMeta) -> Rekey {
    stacking.restack_from(
        FeatureStackingOrder::FramesFeatures,
        Rekey::drop_rest().rename("projection.", "projection."),
    )
}

/// `transformers`' `Nemotron3DiarizationEncoderLayer` names onto a
/// [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)'s,
/// relative to it: `layer_norm{1,2}` its norms, `self_attn`'s
/// `{q,k,v,o}_proj` the attention, `mlp.fc{1,2}` the MLP. Anything else is
/// dropped.
pub fn encoder_layer_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("layer_norm1.", "attn_norm.")
        .rename("layer_norm2.", "mlp_norm.")
        .rename("self_attn.q_proj.", "attn.c_q.")
        .rename("self_attn.k_proj.", "attn.c_k.")
        .rename("self_attn.v_proj.", "attn.c_v.")
        .rename("self_attn.o_proj.", "attn.c_proj.")
        .rename("mlp.fc1.", "mlp.linear1.")
        .rename("mlp.fc2.", "mlp.linear2.")
}

/// `transformers`' `Nemotron3DiarizationEncoder` names onto a
/// [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)'s,
/// relative to it: `input_layer_norm` its input norm, `layer_norm` its final
/// one, each `layers.{n}` by [`encoder_layer_rekey`]. Anything else (the
/// rotary table's `inv_freq`, a buffer) is dropped.
pub fn encoder_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("input_layer_norm.", "input_norm.")
        .rename("layer_norm.", "norm.")
        .mount_indexed("layers.", "layers.", encoder_layer_rekey())
}

/// `transformers`' `Nemotron3DiarizationForAudioFrameClassification`
/// state-dict names onto a
/// [`Sortformer`](crate::kits::speech::diarization::sortformer::Sortformer)'s
/// module paths: the audio tower's embedder by [`embedder_rekey`], its encoder
/// by [`encoder_rekey`], the projection, upsample and head by prefix. Anything
/// else is dropped.
///
/// `stacking` is the encoder's front end, a config or a built module: its
/// stack order fixes its map.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) resolves to
/// `gamma` / `beta`.
pub fn nemotron_diarization_rekey(stacking: &impl FeatureStackingMeta) -> Rekey {
    Rekey::drop_rest()
        .exact("silence_embeds", "silence")
        .mount(
            "model.audio_tower.embedder.",
            "encoder.subsampling.",
            embedder_rekey(stacking),
        )
        .mount(
            "model.audio_tower.",
            "encoder.transformer.",
            encoder_rekey(),
        )
        .rename("model.proj.", "projection.")
        .rename("model.upsampler.conv.", "upsample.conv.")
        .rename("classifier.dense.", "head.dense.")
        .rename("classifier.out_proj.", "head.out.")
}
