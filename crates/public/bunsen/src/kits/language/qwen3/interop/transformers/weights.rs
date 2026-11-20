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
        support::construct::load_lm_rekeyed,
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
