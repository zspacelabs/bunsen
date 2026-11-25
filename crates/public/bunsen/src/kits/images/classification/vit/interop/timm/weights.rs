//! Loading a timm `ViT` checkpoint.
use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    vit_rekey,
    vit_row_splits,
};
use crate::{
    blocks::transformers::models::vision_encoder::VisionEncoderModel,
    burner::module::CanonicalKeys,
    data::formats::{
        NamedTensorsApplied,
        split_rows,
    },
    errors::BunsenResult,
};

/// Loads tensors named as timm's `VisionTransformer` state dict into
/// `model`: the fused `qkv` projections split by rows ([`vit_row_splits`]),
/// then [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`vit_rekey`] and the model's canonical map.
///
/// # Errors
/// As [`split_rows`], and as
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) when the
/// checkpoint does not load into the model.
pub fn load_vit_tensors(
    model: &mut VisionEncoderModel,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors = split_rows(tensors, &vit_row_splits())?;
    vit_rekey()
        .then(model.canonical_rekey())
        .apply_to(model, tensors)
}

/// The model of `loaded`'s timm safetensors checkpoint (its `checkpoint`),
/// built from `config`: the file's precision, or wider as `policy` asks.
///
/// # Errors
/// As [`SafetensorsCheckpoint::tensors`](crate::data::formats::SafetensorsCheckpoint::tensors)
/// reading the file; otherwise as [`load_vit_tensors`], under a frame naming
/// the file, and [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_vit(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::images::classification::vit::ViTContractConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<VisionEncoderModel> {
    use crate::{
        burner::module::ModuleInit,
        data::formats::SafetensorsCheckpoint,
        errors::ResultContext,
        kits::images::classification::vit::pretrained::CHECKPOINT,
    };
    let path = loaded.expect(CHECKPOINT)?;
    let tensors = SafetensorsCheckpoint::single(path).tensors()?;
    let mut vit: VisionEncoderModel = config.try_init(device)?;
    load_vit_tensors(&mut vit, tensors).with_context(|| format!("loading {}", path.display()))?;
    policy.place(vit, device, |_| true)
}
