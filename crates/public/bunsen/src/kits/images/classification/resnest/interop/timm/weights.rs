//! Loading a timm `ResNeSt` checkpoint.

use burn::store::burn_pack::Tensor as PackTensor;

use super::resnest_rekey;
use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
    kits::images::classification::resnet::interop::torchvision::ResNetStateLayout,
};

/// Loads tensors named as timm's `ResNeSt` state dict into `model`:
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`resnest_rekey`] of the model's layout ([`ResNetStateLayout::of`]).
///
/// # Errors
/// As [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to), when the
/// checkpoint does not load into the model.
pub fn load_resnest_tensors(
    model: &mut ConvNetModel,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    resnest_rekey(ResNetStateLayout::of(model)).apply_to(model, tensors)
}

/// The model of `loaded`'s timm safetensors checkpoint (its `checkpoint`),
/// built from `config`: the file's precision, or wider as `policy` asks.
///
/// # Errors
/// As [`SafetensorsCheckpoint::tensors`](crate::data::formats::SafetensorsCheckpoint::tensors)
/// reading the file; otherwise as [`load_resnest_tensors`], under a frame
/// naming the file, and
/// [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_resnest(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::images::classification::resnest::ResNeStContractConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<ConvNetModel> {
    crate::kits::support::interop::load_checkpoint_placed(
        loaded,
        crate::kits::images::classification::resnest::pretrained::CHECKPOINT,
        config,
        device,
        policy,
        load_resnest_tensors,
    )
}
