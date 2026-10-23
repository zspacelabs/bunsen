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

use crate::{
    burner::module::{
        Rekey,
        Target,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
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

/// [`apply_named_tensors`] under a [`Rekey`]: each tensor loads where the
/// map sends it, and its data first changes as the map says
/// ([`Rekey::transform`]), on the host, before the store's adapter.
///
/// A transformed tensor is read whole at that point; the others stay as
/// lazy as their source.
///
/// The map is planned over every name first, seeing the module's own paths
/// ([`Rekey::plan_for`]), and a plan that cannot load as written is refused
/// before any tensor is read.
///
/// # Errors
/// As [`apply_named_tensors`];
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) too
/// as [`RekeyPlan::check`](crate::burner::module::RekeyPlan::check) refuses
/// the plan, or when a transformed tensor's bytes cannot be read.
pub fn apply_rekeyed_tensors<M: ModuleSnapshot>(
    module: &mut M,
    tensors: impl IntoIterator<Item = PackTensor>,
    rekey: &Rekey,
) -> BunsenResult<NamedTensorsApplied> {
    apply_rekeyed_tensors_as(module, tensors, rekey, None)
}

/// [`apply_rekeyed_tensors`], every floating tensor cast to `float_dtype`
/// as [`apply_named_tensors_as`] casts it.
///
/// # Errors
/// As [`apply_rekeyed_tensors`].
pub fn apply_rekeyed_tensors_as<M: ModuleSnapshot>(
    module: &mut M,
    tensors: impl IntoIterator<Item = PackTensor>,
    rekey: &Rekey,
    float_dtype: Option<DType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors.into_iter().collect();
    let module_paths = module
        .collect(None, None, true)
        .into_iter()
        .map(|t| t.name)
        .collect();
    let plan = rekey.plan_for(tensors.iter().map(|t| t.name.as_str()), &module_paths);
    plan.check()?;
    let mut placed = Vec::new();
    let mut dropped = Vec::new();
    for (mut tensor, (_, target)) in tensors.into_iter().zip(plan.targets) {
        match target {
            None => dropped.push(tensor.name),
            Some(Target {
                path,
                transform: None,
            }) => {
                tensor.name = path;
                placed.push(tensor);
            }
            Some(Target {
                path,
                transform: Some(f),
            }) => {
                let (name, param_id) = (tensor.name.clone(), tensor.param_id);
                let data = burn::store::bridge::into_data(tensor).map_err(|e| {
                    BunsenError::invalid_resource(format!("cannot read {name}: {e:?}"))
                })?;
                placed.push(burn::store::bridge::from_data(f(data), path, param_id));
            }
        }
    }
    let mut applied =
        apply_named_tensors_as(module, placed, |name| Some(name.to_string()), float_dtype)?;
    applied.unused.extend(dropped);
    Ok(applied)
}
