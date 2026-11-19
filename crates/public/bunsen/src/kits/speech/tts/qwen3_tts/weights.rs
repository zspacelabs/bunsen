//! Loading a Qwen3-TTS model file into a [`Qwen3TtsModel`].

use alloc::{
    format,
    string::String,
    vec::Vec,
};

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    Qwen3TtsModel,
    code_predictor::weights::load_qwen3_tts_code_predictor_tensors,
    talker::weights::{
        QWEN3_TTS_CODE_PREDICTOR_PREFIX,
        load_qwen3_tts_talker_tensors,
    },
};
use crate::{
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
};

/// Loads a model file's tensors into `model`: the talker's
/// ([`load_qwen3_tts_talker_tensors`]) and the code predictor's
/// ([`load_qwen3_tts_code_predictor_tensors`]); any others (a Base model's
/// speaker encoder) are dropped. Every floating tensor is placed in `dtype`,
/// or, when `None`, in the dtype `model` was built in.
///
/// # Returns
/// Both loads' paths, under the model's (`talker.`, `code_predictor.`).
///
/// # Errors
/// As the two loaders.
pub fn load_qwen3_tts_model_tensors(
    model: &mut Qwen3TtsModel,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let (predictor, talker): (Vec<_>, Vec<_>) = tensors
        .into_iter()
        .partition(|t| t.name.starts_with(QWEN3_TTS_CODE_PREDICTOR_PREFIX));
    let talker = load_qwen3_tts_talker_tensors(&mut model.talker, talker, dtype)?;
    let predictor =
        load_qwen3_tts_code_predictor_tensors(&mut model.code_predictor, predictor, dtype)?;
    Ok(NamedTensorsApplied {
        applied: under("talker.", talker.applied)
            .chain(under("code_predictor.", predictor.applied))
            .collect(),
        unused: Vec::new(),
    })
}

/// `paths`, each under `prefix`.
fn under(
    prefix: &'static str,
    paths: Vec<String>,
) -> impl Iterator<Item = String> {
    paths.into_iter().map(move |p| format!("{prefix}{p}"))
}
