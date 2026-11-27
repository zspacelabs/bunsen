//! Loading a timm `ResNet` checkpoint.

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    errors::BunsenResult,
    kits::images::classification::resnet::ResNetContractConfig,
};

/// The model of `loaded`'s timm safetensors checkpoint (its `checkpoint`),
/// built from `config`, under torchvision's names
/// ([`load_resnet_tensors`](super::super::torchvision::load_resnet_tensors)):
/// the file's precision, or wider as `policy` asks.
///
/// # Errors
/// As [`SafetensorsCheckpoint::tensors`](crate::data::formats::SafetensorsCheckpoint::tensors)
/// reading the file; otherwise as `load_resnet_tensors`, under a frame naming
/// the file, and [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_resnet(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &ResNetContractConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<ConvNetModel> {
    crate::kits::support::interop::load_checkpoint_placed(
        loaded,
        crate::kits::images::classification::resnet::pretrained::CHECKPOINT,
        config,
        device,
        policy,
        |model, tensors| {
            crate::kits::images::classification::resnet::interop::torchvision::load_resnet_tensors(
                model, tensors,
            )
        },
    )
}
