//! Loading a torchvision `ResNet` checkpoint.
use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    ResNetStateLayout,
    resnet_rekey,
};
use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
};

/// Loads tensors named as torchvision's or timm's `ResNet` state dict into
/// `model`: [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`resnet_rekey`] of the model's own [`ResNetStateLayout`], whose indices
/// are the checkpoint's own.
///
/// # Errors
/// As [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to), when the
/// checkpoint does not load into the model.
pub fn load_resnet_tensors(
    model: &mut ConvNetModel,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    resnet_rekey(ResNetStateLayout::of(model)).apply_to(model, tensors)
}

/// The model of `loaded`'s `PyTorch` checkpoint (its `checkpoint`), built from
/// `config`: the files' precision (f32), or wider as `policy` asks.
///
/// # Errors
/// As [`read_pytorch_tensors`](crate::data::formats::read_pytorch_tensors)
/// reading the file: a missing or forbidden file is a
/// [`Lookup`](crate::errors::BunsenErrorKind::Lookup), a file that is not a
/// `PyTorch` checkpoint an
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource);
/// otherwise as [`load_resnet_tensors`], under a frame naming the file, and
/// [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_resnet(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::images::classification::resnet::ResNetContractConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<ConvNetModel> {
    crate::kits::support::interop::load_checkpoint_placed(
        loaded,
        crate::kits::images::classification::resnet::pretrained::CHECKPOINT,
        config,
        device,
        policy,
        load_resnet_tensors,
    )
}
