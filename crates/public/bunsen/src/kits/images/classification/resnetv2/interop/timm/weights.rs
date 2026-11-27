//! Loading a timm `ResNetV2` checkpoint.

use burn::store::burn_pack::Tensor as PackTensor;

use super::resnetv2_rekey;
use crate::{
    blocks::images::models::conv_net::{
        ConvNetMeta,
        ConvNetModel,
    },
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
};

/// Loads tensors named as timm's `ResNetV2` state dict into `model`:
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`resnetv2_rekey`] of the model's head, pre-activation unless no unit
/// normalizes its input (the hybrid `ViT`'s backbone).
///
/// # Errors
/// As [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to), when the
/// checkpoint does not load into the model.
pub fn load_resnetv2_tensors(
    model: &mut ConvNetModel,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let preact = model
        .stages
        .iter()
        .flat_map(|stage| &stage.units)
        .any(|unit| unit.pre.is_some());
    resnetv2_rekey(model.num_classes(), model.d_features(), preact).apply_to(model, tensors)
}

/// The model of `loaded`'s timm safetensors checkpoint (its `checkpoint`),
/// built from `config`: the file's precision, or wider as `policy` asks.
///
/// # Errors
/// As [`SafetensorsCheckpoint::tensors`](crate::data::formats::SafetensorsCheckpoint::tensors)
/// reading the file; otherwise as [`load_resnetv2_tensors`], under a frame
/// naming the file, and
/// [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_resnetv2(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::images::classification::resnetv2::ResNetV2ContractConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<ConvNetModel> {
    crate::kits::support::interop::load_checkpoint_placed(
        loaded,
        crate::kits::images::classification::resnetv2::pretrained::CHECKPOINT,
        config,
        device,
        policy,
        load_resnetv2_tensors,
    )
}
