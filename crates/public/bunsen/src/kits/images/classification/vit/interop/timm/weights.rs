//! Loading a timm `ViT` checkpoint.
use alloc::vec::Vec;

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
    kits::images::classification::vit::interop::openai_clip::{
        is_openai_clip,
        load_openai_clip_tensors,
    },
};

/// Loads tensors named as timm's `VisionTransformer` state dict into
/// `model`: the fused `qkv` projections split by rows ([`vit_row_splits`]),
/// then [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`vit_rekey`] and the model's canonical map. A checkpoint in `OpenAI`
/// CLIP's layout ([`is_openai_clip`]) goes through
/// [`load_openai_clip_tensors`] instead, as timm's `checkpoint_filter_fn`
/// converts it.
///
/// # Errors
/// As [`split_rows`], and as
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) when the
/// checkpoint does not load into the model.
pub fn load_vit_tensors(
    model: &mut VisionEncoderModel,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors.into_iter().collect();
    if is_openai_clip(tensors.iter().map(|t| t.name.as_str())) {
        return load_openai_clip_tensors(model, tensors);
    }
    let tensors = split_rows(tensors, &vit_row_splits())?;
    vit_rekey()
        .then(model.canonical_rekey())
        .apply_to(model, tensors)
}

/// The model of `loaded`'s timm checkpoint (its `checkpoint`: a
/// `model.safetensors`, or an older repo's `pytorch_model.bin`, as its
/// format says), built from `config`: the file's precision, or wider as
/// `policy` asks.
///
/// # Errors
/// As [`ModuleInit::try_init`](crate::burner::module::ModuleInit::try_init)
/// of `config`, and as [`load_vit_into`].
#[cfg(feature = "cache")]
pub fn load_vit(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::images::classification::vit::ViTContractConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<VisionEncoderModel> {
    use crate::burner::module::ModuleInit;
    load_vit_into(loaded, config.try_init(device)?, device, policy)
}

/// `model`, with `loaded`'s timm checkpoint (its `checkpoint`: a
/// `model.safetensors`, or an older repo's `pytorch_model.bin`, as its
/// format says) read into it: the file's precision, or wider as `policy`
/// asks. A `ViT` family's loader builds its model, then reads into it here.
///
/// # Errors
/// As [`SafetensorsCheckpoint::tensors`](crate::data::formats::SafetensorsCheckpoint::tensors)
/// or [`read_pytorch_tensors`](crate::data::formats::read_pytorch_tensors)
/// reading the file; otherwise as [`load_vit_tensors`], under a frame naming
/// the file, and [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_vit_into(
    loaded: &crate::data::resources::LoadedResourceSet,
    mut model: VisionEncoderModel,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<VisionEncoderModel> {
    use crate::{
        data::{
            formats::{
                SafetensorsCheckpoint,
                read_pytorch_tensors,
            },
            resources::SAFETENSORS,
        },
        errors::ResultContext,
        kits::images::classification::vit::pretrained::CHECKPOINT,
    };
    let path = loaded.expect(CHECKPOINT)?;
    let safetensors =
        loaded.map.get(CHECKPOINT).and_then(|r| r.format.as_deref()) == Some(SAFETENSORS);
    let applied = match safetensors {
        true => load_vit_tensors(&mut model, SafetensorsCheckpoint::single(path).tensors()?),
        false => load_vit_tensors(&mut model, read_pytorch_tensors(path)?),
    };
    applied.with_context(|| format!("loading {}", path.display()))?;
    policy.place(model, device, |_| true)
}
