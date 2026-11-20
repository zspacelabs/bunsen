//! Loading tensors under a checkpoint's names.
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use crate::{
    blocks::transformers::models::language_decoder::LmDecoderHolder,
    burner::module::{
        CanonicalKeys,
        Rekey,
    },
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
};

/// Loads tensors under a checkpoint's names into `model`'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel),
/// not its holder: [`Rekey::load`] under `rekey` (onto the language model's
/// own paths, `embedding.`, `head.`, `stack.`) then its canonical keys; every
/// floating tensor in `dtype`, or, when `None`, the dtype it was built in.
///
/// # Errors
/// As [`Rekey::load`].
pub fn load_lm_rekeyed<W: LmDecoderHolder>(
    model: &mut W,
    tensors: impl IntoIterator<Item = PackTensor>,
    rekey: Rekey,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let lm = model.lm_mut();
    let rekey = rekey.then(lm.canonical_rekey());
    rekey.load(lm, tensors, dtype)
}
