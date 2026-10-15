//! Loading named tensors into a module.
use burn::{
    store::{
        FloatCastAdapter,
        ModuleAdapter,
        ModuleSnapshot,
        PyTorchToBurnAdapter,
        burn_pack::Tensor as PackTensor,
    },
    tensor::DType,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// What [`apply_named_tensors`] did.
#[derive(Clone, Debug, Default)]
pub struct NamedTensorsApplied {
    /// The module paths loaded.
    pub applied: Vec<String>,

    /// The source names that matched no parameter, or that `rename` dropped.
    pub unused: Vec<String>,
}

/// Loads tensors named as a `PyTorch` state dict (`transformers`', say) into
/// `module`.
///
/// Each name is rewritten by `rename` to the module path it loads into, or
/// dropped when `rename` returns `None` (a buffer the module has no
/// parameter for). The tensors are adapted by [`PyTorchToBurnAdapter`]:
/// linear weights transposed, normalization `weight` / `bias` read as
/// `gamma` / `beta`. Enum variants are skipped in paths.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// a tensor does not fit its parameter, or when a parameter of the module was
/// not among the tensors; the details list them.
pub fn apply_named_tensors<M: ModuleSnapshot>(
    module: &mut M,
    tensors: impl IntoIterator<Item = PackTensor>,
    rename: impl Fn(&str) -> Option<String>,
) -> BunsenResult<NamedTensorsApplied> {
    apply_named_tensors_as(module, tensors, rename, None)
}

/// [`apply_named_tensors`], every floating tensor cast to `float_dtype` on
/// the host as it is applied (burn's [`FloatCastAdapter`]); `None` keeps each
/// tensor's own dtype.
///
/// burn's store loads a floating parameter in the dtype its source
/// declares, so a `bf16` checkpoint applied as it is builds a `bf16` model.
/// A backend without `bf16` (`wgpu`'s WGSL) can neither run that model nor
/// cast it afterwards; casting here never puts a `bf16` tensor on the device.
///
/// # Errors
/// As [`apply_named_tensors`].
pub fn apply_named_tensors_as<M: ModuleSnapshot>(
    module: &mut M,
    tensors: impl IntoIterator<Item = PackTensor>,
    rename: impl Fn(&str) -> Option<String>,
    float_dtype: Option<DType>,
) -> BunsenResult<NamedTensorsApplied> {
    let mut renamed = Vec::new();
    let mut dropped = Vec::new();
    for mut tensor in tensors {
        match rename(&tensor.name) {
            Some(path) => {
                tensor.name = path;
                renamed.push(tensor);
            }
            None => dropped.push(tensor.name),
        }
    }

    let adapter: Box<dyn ModuleAdapter> = match float_dtype {
        Some(dtype) => Box::new(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype))),
        None => Box::new(PyTorchToBurnAdapter),
    };
    let result = module.apply(renamed, None, Some(adapter), true);
    if !result.errors.is_empty() {
        let details: Vec<String> = result.errors.iter().map(|e| format!("{e:?}")).collect();
        return Err(BunsenError::invalid_resource(format!(
            "{} tensors do not fit the model",
            details.len()
        ))
        .with_details(details.join("\n")));
    }
    if !result.missing.is_empty() {
        let missing: Vec<&str> = result.missing.iter().map(|(p, _)| p.as_str()).collect();
        return Err(BunsenError::invalid_resource(format!(
            "the tensors lack {} of the model's parameters: {}",
            missing.len(),
            missing
                .iter()
                .take(8)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .with_details(missing.join("\n")));
    }
    let mut unused = result.unused;
    unused.extend(dropped);
    Ok(NamedTensorsApplied {
        applied: result.applied,
        unused,
    })
}
