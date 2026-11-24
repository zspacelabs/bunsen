//! A load's tensors read and adapted on threads, before burn's apply.
use alloc::{
    boxed::Box,
    collections::BTreeMap,
    format,
    string::{
        String,
        ToString,
    },
    sync::Arc,
    vec::Vec,
};
use core::sync::atomic::{
    AtomicUsize,
    Ordering,
};
use std::sync::Mutex;

use burn::{
    store::{
        ModuleAdapter,
        ModuleContext,
        ModuleSnapshot,
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::DType,
};

use super::BunsenPyTorchToBurnAdapter;
use crate::{
    burner::tensor::{
        float_width,
        transpose_data,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The most threads a load reads and adapts its tensors on: past it, the
/// device's uploads are the bound (Qwen3-4B on CUDA: a 16-thread load is
/// a tenth faster than an 8-thread one).
pub const MAX_LOAD_THREADS: usize = 16;

/// The threads a load prepares its tensors on: the CPUs the process may
/// use, at most [`MAX_LOAD_THREADS`].
pub fn load_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAX_LOAD_THREADS)
}

/// The module type each of `module`'s parameter paths sits in, as burn's
/// apply sees it ([`ModuleContext::module_type`]): `collect`, its adapter
/// recording the context of every tensor it is handed. Nothing is read.
pub(crate) fn module_types<M: ModuleSnapshot>(module: &M) -> BTreeMap<String, String> {
    #[derive(Debug, Clone)]
    struct Record(Arc<Mutex<BTreeMap<String, String>>>);

    impl ModuleAdapter for Record {
        fn adapt(
            &self,
            tensor: PackTensor,
            ctx: ModuleContext<'_>,
        ) -> PackTensor {
            if let Some(module_type) = ctx.module_type() {
                self.0
                    .lock()
                    .expect("the module types' lock")
                    .insert(tensor.name.clone(), module_type.to_string());
            }
            tensor
        }

        fn clone_box(&self) -> Box<dyn ModuleAdapter> {
            Box::new(self.clone())
        }
    }

    let seen = Arc::new(Mutex::new(BTreeMap::new()));
    let _ = module.collect(None, Some(Box::new(Record(seen.clone()))), true);
    seen.lock().expect("the module types' lock").clone()
}

/// How a load casts its floating tensors: to `target`, but a tensor at a
/// path `keep_width` selects, stored wider, keeps its width.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatCast {
    pub target: DType,
    pub keep_width: fn(&str) -> bool,
}

/// `tensors` read and adapted as [`BunsenPyTorchToBurnAdapter`] and then
/// `cast` would adapt them inside burn's apply, on `threads` threads, each
/// taking the next tensor: their data in memory, under the names the module
/// holds them by, for an apply with no adapter. `types` is
/// [`module_types`] of the module.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
/// when a tensor cannot be read.
pub(crate) fn prepare(
    tensors: Vec<PackTensor>,
    types: &BTreeMap<String, String>,
    cast: Option<FloatCast>,
    threads: usize,
) -> BunsenResult<Vec<PackTensor>> {
    let slots: Vec<Mutex<Option<PackTensor>>> =
        tensors.into_iter().map(|t| Mutex::new(Some(t))).collect();
    let done: Vec<Mutex<Option<BunsenResult<PackTensor>>>> =
        (0..slots.len()).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(slot) = slots.get(i) else {
                break;
            };
            let tensor = slot
                .lock()
                .expect("a slot's lock")
                .take()
                .expect("taken once");
            *done[i].lock().expect("a slot's lock") = Some(prepare_one(tensor, types, cast));
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..threads.min(slots.len()) {
            scope.spawn(work);
        }
        work();
    });
    done.into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("a slot's lock")
                .expect("every tensor prepared")
        })
        .collect()
}

/// One tensor read and adapted: [`prepare`]'s unit.
fn prepare_one(
    tensor: PackTensor,
    types: &BTreeMap<String, String>,
    cast: Option<FloatCast>,
) -> BunsenResult<PackTensor> {
    let param = tensor.name.rsplit('.').next().unwrap_or("").to_string();
    let prefix = &tensor.name[..tensor.name.len() - param.len()];
    // A normalization's parameter is held under burn's name, which its
    // `PyTorch` name is read as.
    let module_type = types.get(&tensor.name).or_else(|| {
        let burn_name = match param.as_str() {
            "weight" => "gamma",
            "bias" => "beta",
            _ => return None,
        };
        types.get(&format!("{prefix}{burn_name}"))
    });
    let (transpose, renamed) = module_type.map_or((false, None), |module_type| {
        BunsenPyTorchToBurnAdapter::rule(module_type, &param, tensor.shape.len(), tensor.dtype)
    });
    let name = match renamed {
        Some(burn_name) => format!("{prefix}{burn_name}"),
        None => tensor.name.clone(),
    };
    let param_id = tensor.param_id;
    let source = tensor.name.clone();
    let mut data = bridge::into_data(tensor)
        .map_err(|e| BunsenError::invalid_resource(format!("cannot read {source}: {e:?}")))?;
    if transpose {
        data = transpose_data(data);
    }
    if let Some(FloatCast { target, keep_width }) = cast {
        let dtype = data.dtype();
        let wider = dtype.is_float()
            && target.is_float()
            && float_width(dtype.into()) > float_width(target.into());
        if dtype.is_float() && dtype != target && !(wider && keep_width(&name)) {
            data = data.convert_dtype(target);
        }
    }
    Ok(bridge::from_data(data, name, param_id))
}
