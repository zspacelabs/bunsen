//! Loading named tensors into a module.
use alloc::collections::BTreeSet;

use burn::{
    store::{
        FloatCastAdapter,
        ModuleAdapter,
        ModuleSnapshot,
        PathFilter,
        PyTorchToBurnAdapter,
        burn_pack::Tensor as PackTensor,
    },
    tensor::{
        DType,
        FloatDType,
        TensorData,
    },
};

use crate::{
    burner::module::{
        Rekey,
        Target,
        TensorLoadRowSplit,
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
/// The tensors are applied in batches of at most [`UPLOAD_BATCH_BYTES`],
/// and after each the module's devices are synchronized and their pools
/// release the pages the batch's uploads left free
/// (`Device::memory_cleanup`). A pooled allocator otherwise keeps every
/// page an upload passed through, and a load would reserve about twice the
/// model's size. The values loaded are the same either way.
///
/// # Errors
/// As [`apply_named_tensors`].
pub fn apply_named_tensors_as<M: ModuleSnapshot>(
    module: &mut M,
    tensors: impl IntoIterator<Item = PackTensor>,
    rename: impl Fn(&str) -> Option<String>,
    float_dtype: Option<DType>,
) -> BunsenResult<NamedTensorsApplied> {
    apply_in_batches(module, tensors, rename, float_dtype, UPLOAD_BATCH_BYTES)
}

/// [`apply_named_tensors_as`], in batches of at most `batch_bytes`.
fn apply_in_batches<M: ModuleSnapshot>(
    module: &mut M,
    tensors: impl IntoIterator<Item = PackTensor>,
    rename: impl Fn(&str) -> Option<String>,
    float_dtype: Option<DType>,
    batch_bytes: usize,
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

    let adapter = || -> Box<dyn ModuleAdapter> {
        match float_dtype {
            Some(dtype) => Box::new(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype))),
            None => Box::new(PyTorchToBurnAdapter),
        }
    };
    let paths: Vec<String> = module
        .collect(None, None, true)
        .into_iter()
        .map(|t| t.name)
        .collect();
    let devices = module.devices();

    let mut applied = Vec::new();
    let mut unmatched: Option<BTreeSet<String>> = None;
    let mut errors = Vec::new();
    for batch in upload_batches(renamed, batch_bytes) {
        let result = module.apply(batch, None, Some(adapter()), true);
        errors.extend(result.errors.iter().map(|e| format!("{e:?}")));
        applied.extend(result.applied);
        // Each tensor is in one batch, which reports it unused if no
        // parameter took it.
        let unused: BTreeSet<String> = result.unused.into_iter().collect();
        unmatched = Some(match unmatched {
            None => unused,
            Some(before) => before.union(&unused).cloned().collect(),
        });
        // The pages a batch's uploads went through are free once its work
        // is done, but the pool keeps them: hand them back before the next
        // batch, so a load peaks near the model's own size rather than
        // twice it.
        for device in &devices {
            let _ = device.sync();
            device.memory_cleanup();
        }
    }
    if !errors.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "{} tensors do not fit the model",
            errors.len()
        ))
        .with_details(errors.join("\n")));
    }
    let loaded: BTreeSet<&str> = applied.iter().map(String::as_str).collect();
    let missing: Vec<&str> = paths
        .iter()
        .map(String::as_str)
        .filter(|p| !loaded.contains(p))
        .collect();
    if !missing.is_empty() {
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
    let mut unused: Vec<String> = unmatched.unwrap_or_default().into_iter().collect();
    unused.extend(dropped);
    Ok(NamedTensorsApplied { applied, unused })
}

/// The most bytes of source tensors one batch of a load uploads before the
/// device's pool is handed back its free pages.
pub const UPLOAD_BATCH_BYTES: usize = 512 << 20;

/// `tensors` in order, cut into batches of at most `cap` bytes (a tensor
/// larger than that is a batch of its own).
fn upload_batches(
    tensors: Vec<PackTensor>,
    cap: usize,
) -> Vec<Vec<PackTensor>> {
    let mut batches = vec![Vec::new()];
    let mut bytes = 0;
    for tensor in tensors {
        let size = tensor.shape.iter().product::<usize>() * tensor.dtype.size();
        if bytes > 0 && bytes + size > cap {
            batches.push(Vec::new());
            bytes = 0;
        }
        bytes += size;
        batches.last_mut().expect("a batch").push(tensor);
    }
    batches
}

impl Rekey {
    /// [`apply_named_tensors`] under this map: each tensor loads where the
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
    pub fn apply_to<M: ModuleSnapshot>(
        &self,
        module: &mut M,
        tensors: impl IntoIterator<Item = PackTensor>,
    ) -> BunsenResult<NamedTensorsApplied> {
        self.apply_as(module, tensors, None)
    }
}

/// The float dtype most of `tensors`' float elements are stored in: what a
/// checkpoint's floats are, for a hook to resolve its
/// [`LoadPolicy`](crate::data::formats::LoadPolicy) against; `None`
/// for a checkpoint with no float tensor.
pub fn stored_float_dtype(tensors: &[PackTensor]) -> Option<FloatDType> {
    let mut counts: Vec<(DType, usize)> = Vec::new();
    for tensor in tensors {
        if !tensor.dtype.is_float() {
            continue;
        }
        let n = tensor.shape.iter().product::<usize>();
        match counts.iter_mut().find(|(d, _)| *d == tensor.dtype) {
            Some((_, count)) => *count += n,
            None => counts.push((tensor.dtype, n)),
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(dtype, _)| FloatDType::from(dtype))
}

/// A tensor of ones for each parameter of `module` that `select` picks,
/// named by its record path: the scales of norms a checkpoint stores none
/// for, because upstream's norm has no learned scale (a weightless RMS norm,
/// `F.rms_norm` without a weight). A unit scale is that norm exactly, in any
/// dtype.
///
/// `select` sees each parameter's record path and the types of the modules
/// it sits in (`Struct:Model.Enum:BunsenNormalization.…`). The tensors are
/// named by record path, so the load's map must pass those names through as
/// they are (an [`exact`](Rekey::exact) rule for each, ahead of the model's
/// canonical map).
pub fn unit_scale_tensors<M: ModuleSnapshot>(
    module: &M,
    select: fn(&str, &str) -> bool,
) -> Vec<PackTensor> {
    module
        .collect(Some(PathFilter::new().with_predicate(select)), None, true)
        .into_iter()
        .map(|t| {
            let n: usize = t.shape.iter().product();
            burn::store::bridge::from_data(
                TensorData::new(vec![1.0f32; n], t.shape.clone()),
                t.name.clone(),
                None,
            )
        })
        .collect()
}

/// `tensors` with each one a [`TensorLoadRowSplit`] takes split into its parts,
/// in place, in the checkpoint's names: the pass before a [`Rekey`] for a
/// checkpoint that fuses what a module holds apart (a packed `qkv`
/// projection, a gated MLP's two projections). The first split whose suffix
/// a name ends in takes it; every other tensor passes as it is.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// a tensor cannot be read, or its rows do not split as its split says.
pub fn split_rows(
    tensors: impl IntoIterator<Item = PackTensor>,
    splits: &[TensorLoadRowSplit],
) -> BunsenResult<Vec<PackTensor>> {
    let mut out = Vec::new();
    for tensor in tensors {
        let Some(split) = splits.iter().find(|s| tensor.name.ends_with(s.suffix())) else {
            out.push(tensor);
            continue;
        };
        let name = tensor.name.clone();
        let data = burn::store::bridge::into_data(tensor)
            .map_err(|e| BunsenError::invalid_resource(format!("cannot read {name}: {e:?}")))?;
        let parts = split
            .split(&name, data)
            .expect("the split's suffix matched")
            .map_err(BunsenError::invalid_resource)?;
        out.extend(
            parts
                .into_iter()
                .map(|(part, data)| burn::store::bridge::from_data(data, part, None)),
        );
    }
    Ok(out)
}

impl Rekey {
    /// Loads tensors under their checkpoint's names into `module`:
    /// [`Rekey::apply_as`] under this map, every floating tensor cast to
    /// `dtype` (a load policy's resolution), or, when `None`, to the dtype
    /// `module` was built in
    /// ([`module_float_dtype`](crate::burner::module::module_float_dtype),
    /// which leaves a freshly built module's lazy parameters uninitialized).
    /// What a kit's `transformers` loader is.
    ///
    /// # Errors
    /// As [`Rekey::apply_to`].
    pub fn load<M: ModuleSnapshot>(
        &self,
        module: &mut M,
        tensors: impl IntoIterator<Item = PackTensor>,
        dtype: Option<FloatDType>,
    ) -> BunsenResult<NamedTensorsApplied> {
        let dtype = dtype.or_else(|| crate::burner::module::module_float_dtype(module));
        self.apply_as(module, tensors, dtype.map(DType::from))
    }
}

impl Rekey {
    /// [`Rekey::apply_to`], every floating tensor cast to `float_dtype`
    /// as [`apply_named_tensors_as`] casts it.
    ///
    /// # Errors
    /// As [`Rekey::apply_to`].
    pub fn apply_as<M: ModuleSnapshot>(
        &self,
        module: &mut M,
        tensors: impl IntoIterator<Item = PackTensor>,
        float_dtype: Option<DType>,
    ) -> BunsenResult<NamedTensorsApplied> {
        let tensors: Vec<PackTensor> = tensors.into_iter().collect();
        let module_paths = module
            .collect(None, None, true)
            .into_iter()
            .map(|t| t.name)
            .collect();
        let plan = self.plan_for(tensors.iter().map(|t| t.name.as_str()), &module_paths);
        plan.check()?;
        let mut placed = Vec::new();
        let mut dropped = Vec::new();
        for (mut tensor, (_, target)) in tensors.into_iter().zip(plan.targets) {
            match target {
                None => dropped.push(tensor.name),
                Some(Target {
                    path,
                    transform: Some(f),
                    ..
                }) => {
                    let (name, param_id) = (tensor.name.clone(), tensor.param_id);
                    let data = burn::store::bridge::into_data(tensor).map_err(|e| {
                        BunsenError::invalid_resource(format!("cannot read {name}: {e:?}"))
                    })?;
                    placed.push(burn::store::bridge::from_data(f(data), path, param_id));
                }
                Some(Target { path, .. }) => {
                    tensor.name = path;
                    placed.push(tensor);
                }
            }
        }
        let mut applied =
            apply_named_tensors_as(module, placed, |name| Some(name.to_string()), float_dtype)?;
        applied.unused.extend(dropped);
        Ok(applied)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Module,
        nn::{
            Linear,
            LinearConfig,
        },
        prelude::*,
        store::bridge,
    };

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
    };

    #[derive(Module, Debug)]
    struct Two {
        a: Linear,
        b: Linear,
    }

    fn two(device: &Device) -> Two {
        Two {
            a: LinearConfig::new(3, 2).init(device),
            b: LinearConfig::new(2, 4).init(device),
        }
    }

    /// `[rows, cols]` of `value`, named as a `PyTorch` state dict names it.
    fn tensor(
        name: &str,
        rows: usize,
        cols: usize,
        value: f32,
    ) -> PackTensor {
        let data = TensorData::new(vec![value; rows * cols], [rows, cols]);
        bridge::from_data(data, name.to_string(), None)
    }

    fn bias(
        name: &str,
        n: usize,
        value: f32,
    ) -> PackTensor {
        bridge::from_data(TensorData::new(vec![value; n], [n]), name.to_string(), None)
    }

    #[test]
    fn test_upload_batches() {
        let sizes = |batches: Vec<Vec<PackTensor>>| -> Vec<usize> {
            batches.iter().map(Vec::len).collect()
        };
        let tensors = || {
            vec![
                tensor("x", 2, 2, 0.0),
                tensor("y", 2, 2, 0.0),
                tensor("z", 8, 8, 0.0),
                tensor("w", 1, 1, 0.0),
            ]
        };
        // 16, 16, 256 and 4 bytes.
        assert_eq!(sizes(upload_batches(tensors(), 32)), [2, 1, 1]);
        assert_eq!(sizes(upload_batches(tensors(), 1)), [1, 1, 1, 1]);
        assert_eq!(sizes(upload_batches(tensors(), 1 << 20)), [4]);
    }

    /// Applied in one-tensor batches, a load is the one-batch load: every
    /// parameter, the unmatched tensors once, a missing parameter refused.
    #[test]
    fn test_batches_load_alike() {
        let device = cpu_device();
        let tensors = || {
            vec![
                tensor("a.weight", 2, 3, 1.0),
                bias("a.bias", 2, 2.0),
                tensor("b.weight", 4, 2, 3.0),
                bias("b.bias", 4, 4.0),
                bias("extra", 1, 5.0),
            ]
        };
        let mut whole = two(&device);
        let one = apply_in_batches(
            &mut whole,
            tensors(),
            |n| Some(n.to_string()),
            None,
            1 << 20,
        )
        .unwrap();
        let mut batched = two(&device);
        let many =
            apply_in_batches(&mut batched, tensors(), |n| Some(n.to_string()), None, 1).unwrap();
        assert_eq!(one.unused, ["extra"]);
        assert_eq!(many.unused, ["extra"]);
        let sorted = |mut v: Vec<String>| {
            v.sort();
            v
        };
        assert_eq!(sorted(one.applied), sorted(many.applied.clone()));
        assert_eq!(many.applied.len(), 4);
        batched
            .b
            .weight
            .val()
            .to_data()
            .assert_eq(&whole.b.weight.val().to_data(), true);

        let partial = tensors().into_iter().filter(|t| t.name != "b.bias");
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("lack 1")
            .assert_err(&apply_in_batches(
                &mut two(&device),
                partial,
                |n| Some(n.to_string()),
                None,
                1,
            ));
    }

    #[derive(Module, Debug)]
    struct Qkv {
        q: Linear,
        k: Linear,
        v: Linear,
    }

    /// A fused `qkv` weight and bias, split by rows, load into three
    /// projections, each its run of rows; a tensor no split takes passes as
    /// it is.
    #[test]
    fn test_split_rows_loads_the_parts() {
        let device = cpu_device();
        let fused: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let tensors = vec![
            bridge::from_data(
                TensorData::new(fused.clone(), [6, 4]),
                "attn.qkv.weight".to_string(),
                None,
            ),
            bridge::from_data(
                TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [6]),
                "attn.qkv.bias".to_string(),
                None,
            ),
            bias("attn.other", 2, 9.0),
        ];
        let splits = [
            TensorLoadRowSplit::equal(".qkv.weight", [".q.weight", ".k.weight", ".v.weight"]),
            TensorLoadRowSplit::equal(".qkv.bias", [".q.bias", ".k.bias", ".v.bias"]),
        ];
        let split = split_rows(tensors, &splits).unwrap();
        let names: Vec<&str> = split.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "attn.q.weight",
                "attn.k.weight",
                "attn.v.weight",
                "attn.q.bias",
                "attn.k.bias",
                "attn.v.bias",
                "attn.other",
            ]
        );

        let mut module = Qkv {
            q: LinearConfig::new(4, 2).init(&device),
            k: LinearConfig::new(4, 2).init(&device),
            v: LinearConfig::new(4, 2).init(&device),
        };
        let applied = Rekey::keep_rest()
            .rename("attn.", "")
            .load(&mut module, split, None)
            .unwrap();
        assert_eq!(applied.unused, ["other"]);
        // The `PyTorch` layout's `[out, in]` rows, as burn's `[in, out]`.
        let rows = Tensor::<2>::from_floats(TensorData::new(fused, [6, 4]), &device);
        for (linear, (range, b)) in [&module.q, &module.k, &module.v].into_iter().zip([
            (0..2, [1.0f32, 2.0]),
            (2..4, [3.0, 4.0]),
            (4..6, [5.0, 6.0]),
        ]) {
            linear
                .weight
                .val()
                .into_data()
                .assert_eq(&rows.clone().slice(range).transpose().into_data(), true);
            linear
                .bias
                .as_ref()
                .unwrap()
                .val()
                .into_data()
                .assert_eq(&TensorData::from(b), true);
        }
    }

    /// A tensor whose rows do not split as its split says is refused, by
    /// name.
    #[test]
    fn test_split_rows_refuses_a_bad_split() {
        let splits = [TensorLoadRowSplit::equal(
            ".qkv.weight",
            [".q.weight", ".k.weight", ".v.weight"],
        )];
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("attn.qkv.weight: 4 rows")
            .assert_err(&split_rows(
                vec![tensor("attn.qkv.weight", 4, 2, 0.0)],
                &splits,
            ));
    }
}
