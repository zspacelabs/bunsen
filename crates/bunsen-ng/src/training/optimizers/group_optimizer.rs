#![allow(unused_imports)]
use std::{
    marker::PhantomData,
    sync::Arc,
};

use burn::{
    Tensor,
    grad_clipping::GradientClipping,
    module::{
        AutodiffModule,
        ModuleMapper,
        Param,
        ParamId,
    },
    optim::{
        GradientsParams,
        LearningRate,
        MultiGradientsParams,
        Optimizer,
        SimpleOptimizer,
        adaptor::OptimizerAdaptor,
        record::AdaptorRecord,
    },
    prelude::Backend,
    record::Record,
    tensor::backend::AutodiffBackend,
};
use hashbrown::{
    HashMap,
    HashSet,
};

use crate::training::optimizers::{
    FixedLrSelector,
    clone_simple_optimizer,
    compat::GradAdaptor,
    lr_selectors::LrSelector,
};

/// A group of [`ParamId`] assigned to a single optimizer instance.
#[derive(Clone)]
pub struct OptimizerGroup<B, O>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    /// The Parameters assigned to this group.
    pub params: HashSet<ParamId>,

    /// The optimizer instance assigned to this group.
    pub optim: O,

    /// Learning rate mapping function.
    pub lr_selector: Option<Arc<dyn LrSelector>>,

    phantom: PhantomData<B>,
}

impl<B, O> OptimizerGroup<B, O>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    /// Create a new `GroupOptimizer` with the given parameters and optimizer.
    pub fn new(
        params: HashSet<ParamId>,
        optim: O,
    ) -> Self {
        Self {
            params,
            optim,
            lr_selector: None,
            phantom: PhantomData,
        }
    }

    /// Build a [`OptimizerGroup`] from a [`OptimizerAdaptor`].
    pub fn from_adaptor<M, I>(
        params: I,
        adaptor: &OptimizerAdaptor<O, M, B>,
    ) -> Self
    where
        I: IntoIterator<Item = ParamId>,
        M: AutodiffModule<B>,
    {
        Self::new(
            params.into_iter().collect(),
            clone_simple_optimizer(adaptor),
        )
    }

    /// Get the learning rate for this group.
    pub fn lr(
        &self,
        global: LearningRate,
        named_lrs: &HashMap<String, LearningRate>,
    ) -> LearningRate {
        self.lr_selector
            .as_ref()
            .map(|lr_fn| lr_fn.select(global, named_lrs))
            .unwrap_or(global)
    }

    /// Get the learning rate mapping function.
    pub fn lr_selector(&self) -> Option<Arc<dyn LrSelector>> {
        self.lr_selector.clone()
    }

    /// Set the learning rate mapping function.
    pub fn with_lr_selector<F>(
        mut self,
        selector: F,
    ) -> Self
    where
        F: LrSelector + 'static,
    {
        self.lr_selector = Some(Arc::new(selector));
        self
    }

    /// Set a fixed learning rate for this group.
    pub fn with_fixed_lr(
        self,
        lr: LearningRate,
    ) -> Self {
        self.with_lr_selector(FixedLrSelector::new(lr))
    }
}

/// The state of an [`OptimizerGroup`].
#[derive(Clone)]
pub struct OptimizerGroupRecord<O, B>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    /// The optimizer states for each parameter in the group.
    pub param_map: HashMap<ParamId, AdaptorRecord<O, B>>,
}

impl<O, B> Record<B> for OptimizerGroupRecord<O, B>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    type Item<S2: burn::record::PrecisionSettings> =
        (Vec<(String, <AdaptorRecord<O, B> as Record<B>>::Item<S2>)>,);

    fn into_item<S2: burn::record::PrecisionSettings>(self) -> Self::Item<S2> {
        (self
            .param_map
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.into_item::<S2>()))
            .collect(),)
    }

    fn from_item<S2: burn::record::PrecisionSettings>(
        item: Self::Item<S2>,
        device: &<B as Backend>::Device,
    ) -> Self {
        Self {
            param_map: item
                .0
                .into_iter()
                .map(|(k, v)| {
                    (
                        ParamId::from(k.parse::<u64>().unwrap_or(0)),
                        AdaptorRecord::from_item::<S2>(v, device),
                    )
                })
                .collect(),
        }
    }
}

/// Error during `GroupOptimizerAdaptor2` construction.
#[derive(Debug)]
pub enum GroupOptimizerError {
    /// A `ParamId` was assigned to more than one optimizer group.
    DuplicateParamId {
        param_id: ParamId,
        /// (`type_tag`, index) of the first assignment
        first: (usize, usize),
        /// (`type_tag`, index) of the conflicting assignment
        second: (usize, usize),
    },
}

/// Execute a single optimizer step for one parameter, managing record
/// load/store.
///
/// Factored out to avoid duplicating the record-management logic per type arm.
#[inline(always)]
fn step_group<B, O, const D: usize>(
    optim: &O,
    records: &mut HashMap<ParamId, AdaptorRecord<O, B>>,
    id: ParamId,
    tensor: Tensor<B::InnerBackend, D>,
    grad: Tensor<B::InnerBackend, D>,
    device: &<B::InnerBackend as Backend>::Device,
    lr: LearningRate,
) -> Tensor<B::InnerBackend, D>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    let (key, record) = records.remove_entry(&id).unzip();
    let state = record.map(|r| O::to_device(r.into_state(), device));

    let (tensor, state) = optim.step(lr, tensor, grad, state);

    if let Some(state) = state {
        records.insert(key.unwrap_or(id), AdaptorRecord::from_state(state));
    }

    tensor
}

// ---------------------------------------------------------------------------
// Macro
// ---------------------------------------------------------------------------

/// Define a `GroupOptimizerAdaptorN` and its associated mapper for N
/// `SimpleOptimizer` types.
///
/// # Usage
///
/// ```ignore
/// define_group_optimizer_adaptor!(2, [(O1, 0), (O2, 1)]);
/// define_group_optimizer_adaptor!(3, [(O1, 0), (O2, 1), (O3, 2)]);
/// ```
///
/// Each invocation generates:
/// - `GroupOptimizerAdaptorN<O1, ..., ON, M, B>` — the adaptor struct
/// - `Optimizer<M, B>` impl with `Record` as a tuple of `Vec<HashMap<ParamId,
///   AdaptorRecord<Oi, B>>>`
macro_rules! define_group_optimizer_adaptor {
    ($N:tt, [$(($O:ident, $idx:tt)),+ $(,)?]) => {
        paste::paste! {
            #[doc=concat!("[`OptimizerGroup`] adapter for ", $N, "types")]
            #[derive(Clone)]
            pub struct [<GroupOptimizerAdaptor $N>]<$($O,)+ M, B>
            where
                $( $O: SimpleOptimizer<B::InnerBackend>, )+
                M: AutodiffModule<B>,
                B: AutodiffBackend,
            {
                $( [<groups_ $idx>]: Vec<OptimizerGroup<B, $O>>, )+

                /// `ParamId` → (`type_tag`, `group_index`)
                dispatch: HashMap<ParamId, (usize, usize)>,

                records: <Self as Optimizer<M, B>>::Record,

                grad_clipping: Option<GradientClipping>,
                _module: PhantomData<M>,
            }

            impl<$($O,)+ M, B> [<GroupOptimizerAdaptor $N>]<$($O,)+ M, B>
            where
                $( $O: SimpleOptimizer<B::InnerBackend>, )+
                M: AutodiffModule<B>,
                B: AutodiffBackend,
            {
                /// Construct and validate.
                ///
                /// Returns an error if any `ParamId` appears in more than one
                /// group.
                pub fn new(
                    $( [<groups_ $idx>]: Vec<OptimizerGroup<B, $O>>, )+
                ) -> Result<Self, GroupOptimizerError> {
                    let mut dispatch = HashMap::new();

                    $(
                        for (group_idx, group) in [<groups_ $idx>].iter().enumerate() {
                            for &param_id in &group.params {
                                if let Some(&first) = dispatch.get(&param_id) {
                                    return Err(GroupOptimizerError::DuplicateParamId {
                                        param_id,
                                        first,
                                        second: ($idx, group_idx),
                                    });
                                }
                                dispatch.insert(param_id, ($idx, group_idx));
                            }
                        }
                    )+

                    let records = (
                        $(
                            vec![
                                OptimizerGroupRecord {
                                    param_map: HashMap::new()
                                };
                                [<groups_ $idx>].len()
                            ],
                        )+
                    );

                    Ok(Self {
                        $( [<groups_ $idx>], )+
                        dispatch,
                        records,
                        grad_clipping: None,
                        _module: PhantomData,
                    })
                }

                /// Sets the gradient clipping.
                pub fn with_grad_clipping(
                    mut self,
                    grad_clipping: GradientClipping,
                ) -> Self {
                    self.grad_clipping = Some(grad_clipping);
                    self
                }

                fn step_common(
                    &mut self,
                    lr: LearningRate,
                    module: M,
                    mut grads: GradAdaptor,
                ) -> M {
                    let named_lrs: HashMap<String, LearningRate> = Default::default();

                    module.map(&mut [<GroupOptimizerMapper $N>] {
                        $( [<groups_ $idx>]: &self.[<groups_ $idx>], )+
                        dispatch: &self.dispatch,
                        $( [<records_ $idx>]: &mut self.records.$idx, )+
                        grads: &mut grads,
                        global_lr: lr,
                        named_lrs: &named_lrs,
                        grad_clipping: self.grad_clipping.as_ref(),
                    })
                }
            }

            impl<$($O,)+ M, B> Optimizer<M, B>
                for [<GroupOptimizerAdaptor $N>]<$($O,)+ M, B>
            where
                $( $O: SimpleOptimizer<B::InnerBackend>, )+
                M: AutodiffModule<B>,
                B: AutodiffBackend,
            {
                #[allow(clippy::type_complexity)]
                type Record = (
                    $( Vec<OptimizerGroupRecord<$O, B>>, )+
                );

                fn step(
                    &mut self,
                    lr: LearningRate,
                    module: M,
                    grads: GradientsParams,
                ) -> M {
                    self.step_common(lr, module, grads.into())
                }

                fn step_multi(
                    &mut self,
                    lr: LearningRate,
                    module: M,
                    grads: MultiGradientsParams,
                ) -> M {
                    self.step_common(lr, module, grads.into())
                }

                fn to_record(&self) -> Self::Record {
                    self.records.clone()
                }

                fn load_record(
                    mut self,
                    record: Self::Record,
                ) -> Self {
                    self.records = record;
                    self
                }
            }

            #[doc=concat!("Mapper for [`GroupOptimizer", $N, "'].")]
            struct [<GroupOptimizerMapper $N>]<'a, B, $($O,)+>
            where
                B: AutodiffBackend,
                $( $O: SimpleOptimizer<B::InnerBackend>, )+
            {
                $( [<groups_ $idx>]: &'a Vec<OptimizerGroup<B, $O>>, )+

                dispatch: &'a HashMap<ParamId, (usize, usize)>,

                $( [<records_ $idx>]: &'a mut Vec<OptimizerGroupRecord<$O, B>>, )+

                grads: &'a mut GradAdaptor,

                global_lr: LearningRate,
                named_lrs: &'a HashMap<String, LearningRate>,

                grad_clipping: Option<&'a GradientClipping>,
            }

            impl<B, $($O,)+> ModuleMapper<B>
                for [<GroupOptimizerMapper $N>]<'_, B, $($O,)+>
            where
                B: AutodiffBackend,
                $( $O: SimpleOptimizer<B::InnerBackend>, )+
            {
                fn map_float<const D: usize>(
                    &mut self,
                    param: Param<Tensor<B, D>>,
                ) -> Param<Tensor<B, D>> {
                    let (id, tensor, mapper) = param.consume();

                    let Some((grad, device)) =
                        self.grads.remove::<B::InnerBackend, D>(id)
                    else {
                        return Param::from_mapped_value(id, tensor, mapper);
                    };

                    let Some(&(type_tag, idx)) = self.dispatch.get(&id) else {
                        return Param::from_mapped_value(id, tensor, mapper);
                    };

                    let is_require_grad = tensor.is_require_grad();

                    let tensor = if tensor.device() != device {
                        tensor.to_device(&device)
                    } else {
                        tensor
                    };

                    let grad = if let Some(clipping) = self.grad_clipping {
                        clipping.clip_gradient(grad)
                    } else {
                        grad
                    };

                    let tensor = match type_tag {
                        $(
                            $idx => {
                                let group = &self.[<groups_ $idx>][idx];
                                let lr = group.lr(self.global_lr, self.named_lrs);

                                step_group::<B, $O, D>(
                                    &group.optim,
                                    &mut self.[<records_ $idx>][idx].param_map,
                                    id,
                                    tensor.inner(),
                                    grad,
                                    &device,
                                    lr,
                                )
                            },
                        )+
                        _ => unreachable!(
                            concat!(
                                stringify!([<GroupOptimizerAdaptor $N>]),
                                " only has type tags 0..",
                                $N,
                            )
                        ),
                    };

                    let mut tensor = Tensor::from_inner(tensor);
                    if is_require_grad {
                        tensor = tensor.require_grad();
                    }

                    Param::from_mapped_value(id, tensor, mapper)
                }
            }

        } // paste!
    };
}

// ---------------------------------------------------------------------------
// Instantiations
// ---------------------------------------------------------------------------

define_group_optimizer_adaptor!(2, [(O1, 0), (O2, 1)]);
define_group_optimizer_adaptor!(3, [(O1, 0), (O2, 1), (O3, 2)]);
define_group_optimizer_adaptor!(4, [(O1, 0), (O2, 1), (O3, 2), (O4, 3)]);
define_group_optimizer_adaptor!(5, [(O1, 0), (O2, 1), (O3, 2), (O4, 3), (O5, 4)]);
define_group_optimizer_adaptor!(6, [(O1, 0), (O2, 1), (O3, 2), (O4, 3), (O5, 4), (O6, 5)]);
define_group_optimizer_adaptor!(
    7,
    [
        (O1, 0),
        (O2, 1),
        (O3, 2),
        (O4, 3),
        (O5, 4),
        (O6, 5),
        (O7, 6)
    ]
);

#[cfg(test)]
mod tests {
    #[test]
    fn test_nothing() {}
}
