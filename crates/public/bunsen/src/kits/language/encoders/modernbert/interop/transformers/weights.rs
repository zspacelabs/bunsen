//! Loading a `transformers` `ModernBERT` checkpoint.
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    modernbert_rekey,
    modernbert_row_splits,
};
use crate::{
    burner::module::{
        CanonicalKeys,
        Rekey,
    },
    data::formats::{
        NamedTensorsApplied,
        split_rows,
    },
    errors::BunsenResult,
    kits::language::encoders::modernbert::ModernBertModel,
};

/// The prefix a checkpoint's encoder sits under: `model.` for a
/// `ModernBertFor*`'s (a masked LM's, a classifier's), none for a bare
/// `ModernBertModel`'s.
pub fn modernbert_prefix(tensors: &[PackTensor]) -> &'static str {
    if tensors.iter().any(|t| t.name.starts_with("model.layers.")) {
        "model."
    } else {
        ""
    }
}

/// Loads tensors named as a `transformers` `ModernBERT` state dict (a bare
/// model's, or under [`modernbert_prefix`]) into `model`: the fused
/// projections split ([`modernbert_row_splits`]), then [`Rekey::load`]
/// under [`modernbert_rekey`]; in `dtype` (a load policy's resolution),
/// or, when `None`, in the dtype `model` was built in.
///
/// # Errors
/// As [`split_rows`] and [`Rekey::load`](crate::burner::module::Rekey::load).
pub fn load_modernbert_tensors(
    model: &mut ModernBertModel,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors.into_iter().collect();
    let prefix = modernbert_prefix(&tensors);
    let tensors = split_rows(tensors, &modernbert_row_splits())?;
    let rekey = Rekey::drop_rest().mount(prefix, "", modernbert_rekey());
    let rekey = rekey.then(model.canonical_rekey());
    rekey.load(model, tensors, dtype)
}
