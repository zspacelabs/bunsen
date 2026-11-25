//! Loading an `OpenAI` CLIP checkpoint's image tower.
use alloc::{
    format,
    vec,
    vec::Vec,
};

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::TensorData,
};

use super::{
    OPENAI_CLIP_PREFIX,
    openai_clip_rekey,
    openai_clip_row_splits,
};
use crate::{
    blocks::{
        images::patching::prefix_tokens::PrefixTokensMeta,
        transformers::models::vision_encoder::{
            VisionEncoderMeta,
            VisionEncoderModel,
        },
    },
    burner::module::CanonicalKeys,
    data::formats::{
        NamedTensorsApplied,
        split_rows,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::classification::vit::interop::timm::vit_rekey,
};

/// Loads an `OpenAI` CLIP checkpoint's image tower into `model`: the fused
/// `in_proj`s split by rows ([`openai_clip_row_splits`]), then
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`openai_clip_rekey`], [`vit_rekey`] and the model's canonical map. The
/// head's bias, which CLIP's projection has none of, is zero, as timm's
/// conversion makes it.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// the model has a position embedding the checkpoint has nothing for (no
/// position embedding); as [`split_rows`], and as
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) when the
/// checkpoint does not load into the model.
pub fn load_openai_clip_tensors(
    model: &mut VisionEncoderModel,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let pos_embed_len = model.tokens.pos_embed_len().ok_or_else(|| {
        BunsenError::invalid_resource(
            "an OpenAI CLIP checkpoint has a position embedding; the model has none",
        )
    })?;
    let mut tensors = split_rows(tensors, &openai_clip_row_splits())?;
    let classes = model.num_classes();
    if classes > 0 {
        tensors.push(burn::store::bridge::from_data(
            TensorData::new(vec![0.0f32; classes], [classes]),
            format!("{OPENAI_CLIP_PREFIX}head.bias"),
            None,
        ));
    }
    let tensors: Vec<PackTensor> = tensors;
    openai_clip_rekey(model.d_model(), pos_embed_len)
        .then(vit_rekey())
        .then(model.canonical_rekey())
        .apply_to(model, tensors)
}
