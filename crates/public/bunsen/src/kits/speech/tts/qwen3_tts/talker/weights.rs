//! Loading a Qwen3-TTS model file's `talker.*` tensors into a
//! [`Qwen3TtsTalker`].
//!
//! A model's `model.safetensors` names the talker as `qwen_tts`'
//! `Qwen3TTSTalkerForConditionalGeneration` does, under `talker.`: its
//! decoder under `talker.model.` (Qwen3's layer names, the codec table as
//! `codec_embedding`, the text table as `text_embedding`), the head as
//! `talker.codec_head`, the text projection as `talker.text_projection`, and
//! the code predictor under `talker.code_predictor.`, which the code
//! predictor's loader reads.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::Qwen3TtsTalker;
use crate::{
    burner::module::Rekey,
    data::formats::NamedTensorsApplied,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::language::qwen3::interop::transformers::text_layer_rekey,
};

/// The prefix the talker's tensors sit under in a model file.
pub const QWEN3_TTS_TALKER_PREFIX: &str = "talker.";

/// The prefix of the code predictor's tensors, inside the talker's.
pub const QWEN3_TTS_CODE_PREDICTOR_PREFIX: &str = "talker.code_predictor.";

/// The talker's names, without [`QWEN3_TTS_TALKER_PREFIX`], onto a
/// [`Qwen3TtsTalker`]'s canonical ones: the decoder's layers by Qwen3's
/// [`text_layer_rekey`], its final norm, the codec table as the decoder's
/// embedding and `codec_head` as its head; the text table and its
/// projection as they are.
pub fn qwen3_tts_talker_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount(
            "model.",
            "",
            Rekey::drop_rest()
                .mount_indexed("layers.", "lm.stack.layers.", text_layer_rekey())
                .rename("norm.", "lm.stack.norm.")
                .rename("codec_embedding.", "lm.embedding.")
                .rename("text_embedding.", "text_embedding."),
        )
        .rename("codec_head.", "lm.head.proj.")
        .rename("text_projection.", "text_projection.")
}

/// Loads a model file's talker tensors (`talker.*` but the code
/// predictor's; any others are dropped) into `model`: the names mapped
/// ([`qwen3_tts_talker_rekey`]), every floating tensor in `dtype`, or, when
/// `None`, in the dtype `model` was built in.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`Rekey::load`]; and naming any talker tensor the model has no parameter
/// for.
pub fn load_qwen3_tts_talker_tensors(
    model: &mut Qwen3TtsTalker,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| {
            t.name.starts_with(QWEN3_TTS_TALKER_PREFIX)
                && !t.name.starts_with(QWEN3_TTS_CODE_PREDICTOR_PREFIX)
        })
        .collect();
    let rekey = Rekey::drop_rest()
        .mount(QWEN3_TTS_TALKER_PREFIX, "", qwen3_tts_talker_rekey())
        .then(Rekey::keep_rest().mount_canonical("lm.", &model.lm));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "Qwen3-TTS talker tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}
