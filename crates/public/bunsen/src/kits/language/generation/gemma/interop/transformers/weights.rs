//! Loading a `transformers` Gemma checkpoint.
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::gemma_rekey;
use crate::{
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
    kits::{
        language::generation::gemma::GemmaModel,
        support::interop::load_lm_rekeyed,
    },
};

/// Loads tensors named as a `transformers` Gemma state dict into `model`'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
/// [`load_lm_rekeyed`] under [`gemma_rekey`] for `model`'s generation, every
/// floating tensor in `dtype` (a load policy's resolution), or, with `None`,
/// in the dtype `model` was built in.
///
/// # Errors
/// As [`load_lm_rekeyed`].
pub fn load_gemma_tensors(
    model: &mut GemmaModel,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let rekey = gemma_rekey(model.lm.decoder.stack.layers[0].attn.post.is_some());
    load_lm_rekeyed(model, tensors, rekey, dtype)
}

/// The model of `loaded`'s checkpoint (its `checkpoint`, one file or
/// shards), built from `config`: the weights in the stored dtype (bf16)
/// where `policy` and the device allow it, never f16.
///
/// # Errors
/// As
/// [`load_transformers_lm`](crate::kits::support::transformers::interop::load_transformers_lm)
/// and [`load_gemma_tensors`].
#[cfg(feature = "cache")]
pub fn load_gemma(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::language::generation::gemma::GemmaConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<GemmaModel> {
    crate::kits::support::transformers::interop::load_transformers_lm(
        loaded,
        crate::kits::language::generation::gemma::pretrained::CHECKPOINT,
        config,
        device,
        policy,
        |model, tensors, dtype| load_gemma_tensors(model, tensors, Some(dtype)),
    )
}
