//! Loading a `transformers` language model's checkpoint.
#[cfg(feature = "cache")]
use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

#[cfg(feature = "cache")]
use crate::{
    data::formats::NamedTensorsApplied,
    errors::BunsenResult,
};

/// The model of `loaded`'s safetensors checkpoint (under `checkpoint`, one
/// file or shards), built from `config` and loaded by `load_tensors` (the
/// kit's `load_{target}_tensors`): every floating tensor in the stored dtype
/// where `policy` and the device allow it, never f16, whose range a
/// bf16-trained model's activations exceed. What a kit's `load_{target}`
/// wraps.
///
/// # Errors
/// When the files can't be read; as
/// [`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve), the
/// config's [`try_init`](crate::burner::module::ModuleInit::try_init), and
/// `load_tensors`.
#[cfg(feature = "cache")]
pub fn load_transformers_lm<C, M>(
    loaded: &crate::data::resources::LoadedResourceSet,
    checkpoint: &str,
    config: &C,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
    load_tensors: impl FnOnce(&mut M, Vec<PackTensor>, FloatDType) -> BunsenResult<NamedTensorsApplied>,
) -> BunsenResult<M>
where
    M: burn::module::Module,
    C: crate::burner::module::ModuleInit<M>,
{
    use crate::data::formats::{
        SafetensorsCheckpoint,
        stored_float_dtype,
    };
    let tensors = SafetensorsCheckpoint::from_loaded(loaded, checkpoint)?.tensors()?;
    let stored = stored_float_dtype(&tensors).unwrap_or(FloatDType::F32);
    let dtype = policy.resolve(stored, device, |d| d != FloatDType::F16)?;
    let mut lm: M = config.try_init(device)?;
    load_tensors(&mut lm, tensors, dtype)?;
    Ok(lm)
}
