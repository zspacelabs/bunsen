//! Loading a timm `DeiT` checkpoint.

use crate::{
    blocks::transformers::models::vision_encoder::VisionEncoderModel,
    errors::BunsenResult,
};

/// The model of `loaded`'s timm checkpoint (its `checkpoint`), built from
/// `config`: the vit kit's
/// [`load_vit`](crate::kits::images::classification::vit::interop::timm::load_vit)
/// of the `DeiT`'s model, the file's precision, or wider as `policy` asks.
///
/// # Errors
/// As [`ModuleInit::try_init`](crate::burner::module::ModuleInit::try_init)
/// of `config`, and as `load_vit`'s reading and loading.
#[cfg(feature = "cache")]
pub fn load_deit(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::images::classification::deit::DeiTContractConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<VisionEncoderModel> {
    use crate::{
        burner::module::ModuleInit,
        kits::images::classification::vit::interop::timm::load_vit_into,
    };
    let model: VisionEncoderModel = config.try_init(device)?;
    load_vit_into(loaded, model, device, policy)
}
