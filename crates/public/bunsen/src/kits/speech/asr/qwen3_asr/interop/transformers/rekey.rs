//! The state-dict names.

use crate::{
    blocks::conv::ConvSeq2d,
    burner::module::Rekey,
    kits::language::generation::qwen3::interop::transformers::text_layer_rekey,
};

/// `transformers`' `Qwen3ASRForConditionalGeneration` state-dict names onto
/// a [`Qwen3Asr`](crate::kits::speech::asr::qwen3_asr::Qwen3Asr)'s module
/// paths: under `model.`, the audio tower's 1-based `conv2d{n}` onto its stem's
/// 0-based `ConvSeq2d` blocks ([`ConvSeq2d::one_based_rekey`]), its layers by
/// `transformers`' Whisper layer names
/// ([`whisper_encoder_layer_rekey`](crate::kits::support::transformers::interop::whisper_encoder_layer_rekey)); the text
/// model onto its
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel)
/// (`language_model.lm.`): its layers by [`text_layer_rekey`], its embedding
/// and final norm; the projector's `linear_1` / `linear_2` as its
/// [`BunsenMlp`](crate::blocks::mlp::BunsenMlp)'s `linear1` / `linear2`; and
/// everything else as it is.
/// Anything outside `model.` (the tied `lm_head.weight`) is dropped.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`Rekey::load`] resolves to `gamma` / `beta`.
pub fn qwen3_asr_rekey() -> Rekey {
    Rekey::drop_rest().mount(
        "model.",
        "",
        Rekey::keep_rest()
            .mount(
                "audio_tower.conv2d",
                "audio_tower.stem.",
                ConvSeq2d::one_based_rekey(),
            )
            .mount_indexed(
                "audio_tower.layers.",
                "audio_tower.layers.",
                crate::kits::support::transformers::interop::whisper_encoder_layer_rekey(),
            )
            .mount_indexed(
                "language_model.layers.",
                "language_model.stack.layers.",
                text_layer_rekey(),
            )
            .rename("language_model.embed_tokens.", "language_model.embedding.")
            .rename("language_model.norm.", "language_model.stack.norm.")
            .mount(
                "multi_modal_projector.",
                "multi_modal_projector.",
                Rekey::keep_rest()
                    .rename("linear_1.", "linear1.")
                    .rename("linear_2.", "linear2."),
            ),
    )
}
