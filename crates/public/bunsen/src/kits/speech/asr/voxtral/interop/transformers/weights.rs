//! Loading a `VoxtralRealtimeForConditionalGeneration` checkpoint.
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::voxtral_realtime_rekey;
use crate::{
    burner::module::CanonicalKeys,
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
    kits::speech::asr::voxtral::VoxtralRealtime,
};

/// Loads tensors named as `mistralai/Voxtral-Mini-4B-Realtime-2602`'s
/// `model.safetensors` names them into `model`:
/// [`Rekey::load`](crate::burner::module::Rekey::load) under
/// [`voxtral_realtime_rekey`], every floating tensor in `dtype` (a load
/// policy's resolution), or, with `None`, in the dtype `model` was built in.
/// The checkpoint ships `bf16`, which not every backend computes in.
///
/// # Errors
/// As [`Rekey::load`](crate::burner::module::Rekey::load).
pub fn load_voxtral_realtime_tensors(
    model: &mut VoxtralRealtime,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let rekey = voxtral_realtime_rekey().then(model.canonical_rekey());
    rekey.load(model, tensors, dtype)
}

/// The model of `loaded`'s checkpoint (its `checkpoint`, one file or
/// shards), built from `config`: the weights in the stored dtype (bf16)
/// where `policy` and the device allow it.
///
/// # Errors
/// When a file can't be read, and as
/// [`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve) and
/// [`load_voxtral_realtime_tensors`].
#[cfg(feature = "cache")]
pub fn load_voxtral_realtime(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::speech::asr::voxtral::VoxtralRealtimeConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<VoxtralRealtime> {
    crate::kits::support::interop::load_checkpoint_as(
        loaded,
        crate::kits::speech::asr::voxtral::pretrained::CHECKPOINT,
        config,
        device,
        policy,
        |_| true,
        |model, tensors, dtype| load_voxtral_realtime_tensors(model, tensors, Some(dtype)),
    )
}
