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
    use crate::{
        burner::module::ModuleInit,
        data::formats::SafetensorsCheckpoint,
        errors::ResultContext,
        kits::images::classification::resnet::{
            interop::torchvision::load_resnet_tensors,
            pretrained::CHECKPOINT,
        },
    };
    let path = loaded.expect(CHECKPOINT)?;
    let tensors = SafetensorsCheckpoint::single(path).tensors()?;
    let mut resnet: ConvNetModel = config.try_init(device)?;
    load_resnet_tensors(&mut resnet, tensors)
        .with_context(|| format!("loading {}", path.display()))?;
    policy.place(resnet, device, |_| true)
}
