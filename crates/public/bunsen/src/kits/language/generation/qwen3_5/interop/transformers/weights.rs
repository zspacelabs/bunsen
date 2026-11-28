//! Loading a `transformers` Qwen3.5 checkpoint.
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::qwen3_5_rekey;
use crate::{
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
    kits::{
        language::generation::qwen3_5::{
            Qwen3_5Config,
            Qwen3_5Model,
        },
        support::interop::load_lm_rekeyed,
    },
};

/// Loads tensors named as a `transformers` Qwen3.5 state dict into `model`'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
/// [`load_lm_rekeyed`] under [`qwen3_5_rekey`] of `config`, `model`'s, every
/// floating tensor in `dtype` (a load policy's resolution), or, with `None`, in
/// the dtype `model` was built in.
///
/// # Errors
/// As [`load_lm_rekeyed`].
pub fn load_qwen3_5_tensors(
    model: &mut Qwen3_5Model,
    config: &Qwen3_5Config,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    load_lm_rekeyed(model, tensors, qwen3_5_rekey(config), dtype)
}

/// The model of `loaded`'s checkpoint (its `checkpoint`, one file or
/// shards), built from `config`: the weights in the stored dtype (bf16)
/// where `policy` and the device allow it, never f16.
///
/// # Errors
/// As
/// [`load_transformers_lm`](crate::kits::support::transformers::interop::load_transformers_lm)
/// and [`load_qwen3_5_tensors`].
#[cfg(feature = "cache")]
pub fn load_qwen3_5(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::language::generation::qwen3_5::Qwen3_5Config,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<Qwen3_5Model> {
    crate::kits::support::transformers::interop::load_transformers_lm(
        loaded,
        crate::kits::language::generation::qwen3_5::pretrained::CHECKPOINT,
        config,
        device,
        policy,
        |model, tensors, dtype| load_qwen3_5_tensors(model, config, tensors, Some(dtype)),
    )
}
