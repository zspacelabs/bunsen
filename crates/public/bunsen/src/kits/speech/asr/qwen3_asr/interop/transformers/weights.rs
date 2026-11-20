//! Loading a `Qwen3ASRForConditionalGeneration` checkpoint.
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::qwen3_asr_rekey;
use crate::{
    burner::module::CanonicalKeys,
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
    kits::speech::asr::qwen3_asr::Qwen3Asr,
};

/// Loads tensors named as `transformers`' `Qwen3ASRForConditionalGeneration`
/// state dict into `model`:
/// [`Rekey::load`](crate::burner::module::Rekey::load) under
/// [`qwen3_asr_rekey`], every floating tensor in `dtype` (a load policy's
/// resolution), or, with `None`, in the dtype `model` was built in. The
/// checkpoints ship `bf16`, which not every backend computes in.
///
/// # Errors
/// As [`Rekey::load`](crate::burner::module::Rekey::load).
pub fn load_qwen3_asr_tensors(
    model: &mut Qwen3Asr,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let rekey = qwen3_asr_rekey().then(model.canonical_rekey());
    rekey.load(model, tensors, dtype)
}

/// The model of `loaded`'s `transformers` checkpoint (its `checkpoint`,
/// one file or shards), built from `config`, and its `tokenizer.json`
/// vocabulary (its `vocabulary`): the weights in the stored dtype (bf16)
/// where `policy` and the device allow it, never f16, which overflows its
/// Qwen3 text model.
///
/// # Errors
/// When a file can't be read or parsed, and as [`load_qwen3_asr_tensors`].
#[cfg(feature = "cache")]
pub fn load_qwen3_asr(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::speech::asr::qwen3_asr::Qwen3AsrConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<(Qwen3Asr, crate::kits::support::tokens::ByteLevelVocabulary)> {
    use crate::kits::speech::asr::qwen3_asr::pretrained::{
        CHECKPOINT,
        VOCABULARY,
    };
    let tokenizer = loaded.expect(VOCABULARY)?;
    let json =
        std::fs::read_to_string(tokenizer).map_err(crate::errors::sys_at("read", tokenizer))?;
    let vocab = crate::kits::support::tokens::ByteLevelVocabulary::from_tokenizer_json(&json)
        .map_err(|e| e.context(tokenizer.display()))?;
    let tensors =
        crate::data::formats::SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
    Ok((build_qwen3_asr(tensors, config, device, policy)?, vocab))
}

/// `config`'s model with `tensors` loaded, in the stored dtype where `policy`
/// and the device allow it, never f16: what each source's `load_qwen3_asr`
/// ends with.
///
/// # Errors
/// As [`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve) and
/// [`load_qwen3_asr_tensors`].
#[cfg(feature = "cache")]
pub(crate) fn build_qwen3_asr(
    tensors: Vec<PackTensor>,
    config: &crate::kits::speech::asr::qwen3_asr::Qwen3AsrConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<Qwen3Asr> {
    use crate::burner::module::ModuleInit;
    let stored = crate::data::formats::stored_float_dtype(&tensors).unwrap_or(FloatDType::F32);
    let dtype = policy.resolve(stored, device, |d| d != FloatDType::F16)?;
    let mut asr: Qwen3Asr = config.try_init(device)?;
    load_qwen3_asr_tensors(&mut asr, tensors, Some(dtype))?;
    Ok(asr)
}
