//! Loading a `Qwen3ForCausalLM` state dict into a `Qwen3Model`.
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::qwen3_rekey;
use crate::{
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
    kits::{
        language::qwen3::Qwen3Model,
        support::interop::load_lm_rekeyed,
    },
};

/// Loads tensors named as `transformers`' `Qwen3ForCausalLM` state dict into
/// `model`'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
/// [`load_lm_rekeyed`] under [`qwen3_rekey`]. Every floating tensor is placed
/// in `dtype` (a load policy's resolution,
/// [`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve)), or,
/// with `None`, in the dtype `model` was built in. A freshly built model's
/// parameters are lazy, so they are never materialized in the build dtype
/// first. The checkpoints ship `bf16`, which not every backend computes in.
///
/// # Errors
/// As [`load_lm_rekeyed`].
pub fn load_qwen3_tensors(
    model: &mut Qwen3Model,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    load_lm_rekeyed(model, tensors, qwen3_rekey(), dtype)
}

/// The model of `loaded`'s checkpoint (its `checkpoint`, one file or
/// shards), built from `config`: the weights in the stored dtype (bf16)
/// where `policy` and the device allow it, never f16, which overflows
/// Qwen3's activations.
///
/// # Errors
/// As
/// [`load_transformers_lm`](crate::kits::support::interop::transformers::load_transformers_lm)
/// and [`load_qwen3_tensors`].
#[cfg(feature = "cache")]
pub fn load_qwen3(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::language::qwen3::Qwen3Config,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<Qwen3Model> {
    crate::kits::support::interop::transformers::load_transformers_lm(
        loaded,
        crate::kits::language::qwen3::pretrained::CHECKPOINT,
        config,
        device,
        policy,
        |model, tensors, dtype| load_qwen3_tensors(model, tensors, Some(dtype)),
    )
}
