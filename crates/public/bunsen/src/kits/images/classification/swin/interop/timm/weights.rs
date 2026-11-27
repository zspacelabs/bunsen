//! Loading a timm Swin checkpoint.

use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    swin_rekey,
    swin_row_splits,
};
use crate::{
    blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel,
    burner::module::CanonicalKeys,
    data::formats::{
        NamedTensorsApplied,
        split_rows,
    },
    errors::BunsenResult,
};

/// Loads tensors named as timm's `SwinTransformer` or `SwinTransformerV2`
/// state dict into `model`: the fused `qkv` split by rows
/// ([`swin_row_splits`]), then
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`swin_rekey`] and the model's canonical map.
///
/// # Errors
/// As [`split_rows`], and as
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) when the
/// checkpoint does not load into the model.
pub fn load_swin_tensors(
    model: &mut StagedVisionEncoderModel,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors = split_rows(tensors, &swin_row_splits())?;
    swin_rekey()
        .then(model.canonical_rekey())
        .apply_to(model, tensors)
}

/// The model of `loaded`'s timm safetensors checkpoint (its `checkpoint`
/// resource), built from `config` (a Swin kit's policy): the file's
/// precision, or wider as `policy` asks.
///
/// # Errors
/// As [`ModuleInit::try_init`](crate::burner::module::ModuleInit::try_init)
/// of `config`; as
/// [`SafetensorsCheckpoint::tensors`](crate::data::formats::SafetensorsCheckpoint::tensors)
/// reading the file; otherwise as [`load_swin_tensors`], under a frame
/// naming the file, and
/// [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_swin(
    loaded: &crate::data::resources::LoadedResourceSet,
    checkpoint: &str,
    config: &impl crate::burner::module::ModuleInit<StagedVisionEncoderModel>,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<StagedVisionEncoderModel> {
    crate::kits::support::interop::load_checkpoint_placed(
        loaded,
        checkpoint,
        config,
        device,
        policy,
        load_swin_tensors,
    )
}
