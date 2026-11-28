//! The state-dict names.

use crate::{
    blocks::conv::ConvSeq1d,
    burner::module::Rekey,
};

/// `transformers`' `WhisperForConditionalGeneration` state-dict names onto
/// a [`Whisper`](crate::kits::speech::asr::whisper::Whisper)'s module paths:
/// under `model.`, the layers by their blocks'
/// maps ([`whisper_encoder_layer_rekey`](crate::kits::support::transformers::interop::whisper_encoder_layer_rekey),
/// [`whisper_decoder_layer_rekey`](crate::kits::support::transformers::interop::whisper_decoder_layer_rekey)), `layers.N` as
/// `blocks.N`; the two convolutions, `conv1` and `conv2`, the encoder head's
/// blocks ([`ConvSeq1d::one_based_rekey`]); the positional tables
/// without their `.weight`; `embed_tokens` the token embedding; the final
/// norms `ln_post` and `ln`; everything else as it is. Anything outside
/// `model.` (a tied `proj_out.weight`) is dropped.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`Rekey::apply_to`] resolves to `gamma` / `beta`.
pub fn whisper_rekey() -> Rekey {
    Rekey::drop_rest().mount(
        "model.",
        "",
        Rekey::keep_rest()
            .mount(
                "encoder.conv",
                "encoder.head.",
                ConvSeq1d::one_based_rekey(),
            )
            .exact(
                "encoder.embed_positions.weight",
                "encoder.positional_embedding",
            )
            .exact(
                "decoder.embed_positions.weight",
                "decoder.positional_embedding",
            )
            .rename("decoder.embed_tokens.", "decoder.token_embedding.")
            .mount_indexed(
                "encoder.layers.",
                "encoder.blocks.",
                crate::kits::support::transformers::interop::whisper_encoder_layer_rekey(),
            )
            .mount_indexed(
                "decoder.layers.",
                "decoder.blocks.",
                crate::kits::support::transformers::interop::whisper_decoder_layer_rekey(),
            )
            .rename("encoder.layer_norm.", "encoder.ln_post.")
            .rename("decoder.layer_norm.", "decoder.ln."),
    )
}
