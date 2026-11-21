//! Loading a Laya checkpoint.
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::laya_rekey;
use crate::{
    burner::module::CanonicalKeys,
    data::formats::{
        NamedTensorsApplied,
        split_rows,
    },
    errors::BunsenResult,
    kits::language::{
        decision::laya::LayaModel,
        encoders::modernbert::interop::transformers::modernbert_row_splits,
    },
};

/// Loads a Laya `model.safetensors`' tensors into `model`: the encoder's
/// and the head's fused projections split by rows, then
/// [`Rekey::load`](crate::burner::module::Rekey::load)
/// under [`laya_rekey`]; in `dtype` (a load policy's resolution), or, when
/// `None`, in the dtype `model` was built in.
///
/// # Errors
/// As [`split_rows`] and [`Rekey::load`](crate::burner::module::Rekey::load).
pub fn load_laya_tensors(
    model: &mut LayaModel,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let splits: Vec<_> = modernbert_row_splits()
        .into_iter()
        .chain(crate::kits::support::interop::torch::transformer_encoder_layer_row_splits())
        .collect();
    let tensors = split_rows(tensors, &splits)?;
    let rekey = laya_rekey().then(model.canonical_rekey());
    rekey.load(model, tensors, dtype)
}

/// The model of `loaded`'s `model.safetensors` (its `checkpoint`), built
/// from `config`: the weights in the stored dtype where `policy` and the
/// device allow it.
///
/// # Errors
/// When the file can't be read; as
/// [`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve) and
/// [`load_laya_tensors`].
#[cfg(feature = "cache")]
pub fn load_laya(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::language::decision::laya::LayaConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<LayaModel> {
    use crate::{
        burner::module::ModuleInit,
        data::formats::{
            SafetensorsCheckpoint,
            stored_float_dtype,
        },
        kits::language::decision::laya::pretrained::CHECKPOINT,
    };
    let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
    let stored = stored_float_dtype(&tensors).unwrap_or(FloatDType::F32);
    let dtype = policy.resolve(stored, device, |_| true)?;
    let mut model: LayaModel = config.try_init(device)?;
    load_laya_tensors(&mut model, tensors, Some(dtype))?;
    Ok(model)
}
