//! Loading a Qwen3-TTS model file's `talker.code_predictor.*` tensors into
//! the code predictor's [`DepthDecoder`].
//!
//! The file names the predictor as `qwen_tts`'
//! `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` does: its
//! decoder under `model.` (Qwen3's layer names, its per-codebook tables as
//! `codec_embedding.{i}`), its heads as `lm_head.{i}`, and, where the talker
//! is wider, the projection as `small_to_mtp_projection`.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use crate::{
    blocks::transformers::decoders::depth::DepthDecoder,
    burner::module::Rekey,
    data::formats::NamedTensorsApplied,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::qwen3::interop::transformers::text_layer_rekey,
        speech::tts::qwen3_tts::talker::weights::QWEN3_TTS_CODE_PREDICTOR_PREFIX,
    },
};

/// The predictor's names, without
/// [`QWEN3_TTS_CODE_PREDICTOR_PREFIX`], onto a [`DepthDecoder`]'s
/// canonical ones: the layers by Qwen3's [`text_layer_rekey`], the final
/// norm, the tables, the heads and the projection.
pub fn qwen3_tts_code_predictor_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount(
            "model.",
            "",
            Rekey::drop_rest()
                .mount_indexed("layers.", "stack.layers.", text_layer_rekey())
                .rename("norm.", "stack.norm.")
                .rename("codec_embedding.", "tables."),
        )
        .rename("lm_head.", "heads.")
        .rename("small_to_mtp_projection.", "input_proj.")
}

/// Loads a model file's code predictor tensors (`talker.code_predictor.*`;
/// any others are dropped) into `model`: the names mapped
/// ([`qwen3_tts_code_predictor_rekey`]), every floating tensor in `dtype`,
/// or, when `None`, in the dtype `model` was built in.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`Rekey::load`]; and naming any predictor tensor the model has no
/// parameter for.
pub fn load_qwen3_tts_code_predictor_tensors(
    model: &mut DepthDecoder,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| t.name.starts_with(QWEN3_TTS_CODE_PREDICTOR_PREFIX))
        .collect();
    let rekey = Rekey::drop_rest()
        .mount(
            QWEN3_TTS_CODE_PREDICTOR_PREFIX,
            "",
            qwen3_tts_code_predictor_rekey(),
        )
        .then(Rekey::keep_rest().mount_canonical("", model));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "Qwen3-TTS code predictor tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}
