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
        Module,
        ModuleMapper,
        ModuleVisitor,
        Param,
        ParamId,
    },
    optim::{
        GradientsParams,
        LearningRate,
        MultiGradientsParams,
        Optimizer,
        SimpleOptimizer,
        adaptor::{
            GradAdaptor,
            OptimizerAdaptor,
        },
        record::AdaptorRecord,
    },
    prelude::{
        Backend,
        Device,
    },
    record::Record,
    tensor::backend::{
        AutodiffBackend,
        BackendTypes,
    },
};
use hashbrown::{
    HashMap,
    HashSet,
};

use crate::burner::optim::{
    FixedLrSelector,
    lr_selectors::LrSelector,
};

/// One parameter group: a set of [`ParamId`]s, the optimizer instance that
/// steps them, and an optional [`LrSelector`].
///
/// Build one with [`OptimizerGroup::from_adaptor`] (or
/// [`OptimizerGroup::new`]), usually from a `ParamId` set selected with
/// [`XmlModuleTree`](crate::burner::module::reflection::XmlModuleTree). Pass
/// the module and the groups to a `GroupOptimizerAdaptorN::new` (e.g.
/// [`GroupOptimizerAdaptor2::new`]), one `Vec` of groups per optimizer type.
/// See the [module docs](crate::burner::optim) for the lifecycle.
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
    /// Creates a group of `params`, stepped by `optim`, at the global
    /// learning rate.
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

    /// Creates a group of `params`, stepped by a clone of the optimizer
    /// inside `adaptor`, at the global learning rate.
    ///
    /// `adaptor` is what a `burn` optimizer config's `init()` returns, e.g.
    /// `AdamWConfig::new().init::<B, M>()`. Only its optimizer is used; its
    /// gradient clipping and state are not.
    pub fn from_adaptor<M, I>(
        params: I,
        adaptor: &OptimizerAdaptor<O, M, B>,
    ) -> Self
    where
        I: IntoIterator<Item = ParamId>,
        M: AutodiffModule<B>,
    {
        Self::new(params.into_iter().collect(), adaptor.optim().clone())
    }

    /// Returns this group's learning rate: `global` mapped through the
    /// [`LrSelector`], or `global` itself when there is none.
    pub fn lr(
        &self,
        global: LearningRate,
    ) -> LearningRate {
        self.lr_selector
            .as_ref()
            .map(|lr_fn| lr_fn.select(global))
            .unwrap_or(global)
    }

    /// Returns the learning rate mapping function.
    pub fn lr_selector(&self) -> Option<Arc<dyn LrSelector>> {
        self.lr_selector.clone()
    }

    /// Sets the learning rate mapping function.
    ///
    /// A `Send + Sync` closure from the global rate to this group's rate is
    /// an [`LrSelector`]: `|lr| lr * 0.5`.
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

    /// Sets a fixed learning rate for this group, ignoring the global rate
    /// and its schedule.
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
        Vec<(String, <AdaptorRecord<O, B> as Record<B>>::Item<S2>)>;

    fn into_item<S2: burn::record::PrecisionSettings>(self) -> Self::Item<S2> {
        self.param_map
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.into_item::<S2>()))
            .collect()
    }

    fn from_item<S2: burn::record::PrecisionSettings>(
        item: Self::Item<S2>,
        device: &B::Device,
    ) -> Self {
        Self {
            param_map: item
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

/// Error from `GroupOptimizerAdaptorN::new`, for every `N`
/// ([`GroupOptimizerAdaptor1::new`] through
/// [`GroupOptimizerAdaptor7::new`]): the groups are not a partition of the
/// module's float parameters.
#[derive(Debug, thiserror::Error)]
pub enum GroupOptimizerError {
    /// A `ParamId` was assigned to more than one optimizer group.
    ///
    /// Positions are `(optimizer type, group index)`: the position of the
    /// group's `Vec` among the arguments to `new`, and of the group in that
    /// `Vec`, both from 0.
    #[error(
        "parameter {param_id} is in more than one optimizer group: \
         {first:?} and {second:?}, as (optimizer type, group index)"
    )]
    DuplicateParamId {
        /// The `ParamId` of the conflicting assignment.
        param_id: ParamId,
        /// (optimizer type, group index) of the first group that claims it.
        first: (usize, usize),
        /// (optimizer type, group index) of the second group that claims it.
        second: (usize, usize),
    },

    /// Float parameters of the module are in no optimizer group, so they
    /// would never be stepped.
    ///
    /// Every float parameter must be in a group, including one that gets no
    /// gradient today (such a parameter is not stepped, whatever its group).
    /// Collect the parameters no other group claims into a remnant group to
    /// cover the rest of the module.
    #[error(
        "{} float parameter(s) of the module are in no optimizer group, \
         and would never be stepped: {}",
        .param_ids.len(),
        join_param_ids(.param_ids)
    )]
    UnassignedParamIds {
        /// The unassigned `ParamId`s, in the order the module visits them.
        param_ids: Vec<ParamId>,
    },
}

fn join_param_ids(param_ids: &[ParamId]) -> String {
    param_ids
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Collects the `ParamId`s of a module's float parameters, in visiting
/// order.
struct FloatParamIds(Vec<ParamId>);

impl<B: Backend> ModuleVisitor<B> for FloatParamIds {
    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D>>,
    ) {
        self.0.push(param.id);
    }
}

/// Checks that every float parameter of `module` is in `dispatch`.
fn check_float_params_assigned<B, M>(
    module: &M,
    dispatch: &HashMap<ParamId, (usize, usize)>,
) -> Result<(), GroupOptimizerError>
where
    B: Backend,
    M: Module<B>,
{
    let mut float_params = FloatParamIds(Vec::new());
    module.visit(&mut float_params);

    let unassigned: Vec<ParamId> = float_params
        .0
        .into_iter()
        .filter(|id| !dispatch.contains_key(id))
        .collect();

    if unassigned.is_empty() {
        Ok(())
    } else {
        Err(GroupOptimizerError::UnassignedParamIds {
            param_ids: unassigned,
        })
    }
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
    device: &<<B as AutodiffBackend>::InnerBackend as BackendTypes>::Device,
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

/// Defines a `GroupOptimizerAdaptorN` and its associated mapper for N
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
/// - the `Optimizer<M, B>` impl; its `Record` is a tuple with one
///   `Vec<OptimizerGroupRecord<Oi, B>>` per optimizer type
macro_rules! define_group_optimizer_adaptor {
    ($N:tt, [$(($O:ident, $idx:tt)),+ $(,)?]) => {
        paste::paste! {
            #[doc = concat!(
                "An [`Optimizer`] over [`OptimizerGroup`]s of ",
                $N,
                " optimizer type(s): `new` takes one `Vec` of groups per type.\n\n",
                "Each parameter is stepped by the group that claims it. `new` ",
                "checks that every float parameter of the module is in exactly ",
                "one group. See the ",
                "[module docs](crate::burner::optim) for the lifecycle and an example.",
            )]
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
                /// Builds the adaptor for `module` from one `Vec` of groups
                /// per optimizer type, in type-parameter order.
                ///
                /// `module` is only read, to list its float parameters; pass
                /// the module you will step.
                ///
                /// # Errors
                /// - [`GroupOptimizerError::DuplicateParamId`] if a `ParamId`
                ///   appears in more than one group, of the same type or not.
                /// - [`GroupOptimizerError::UnassignedParamIds`] if a float
                ///   parameter of `module` is in no group.
                #[allow(clippy::too_many_arguments)]
                pub fn new(
                    module: &M,
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

                    check_float_params_assigned(module, &dispatch)?;

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

                /// Sets the gradient clipping, applied to each parameter's
                /// gradient on its own before its group's step.
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
                    module.map(&mut [<GroupOptimizerMapper $N>] {
                        $( [<groups_ $idx>]: &self.[<groups_ $idx>], )+
                        dispatch: &self.dispatch,
                        $( [<records_ $idx>]: &mut self.records.$idx, )+
                        grads: &mut grads,
                        global_lr: lr,
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

            #[doc = concat!(
                "[`ModuleMapper`] that steps parameters for [`GroupOptimizerAdaptor",
                $N,
                "`].",
            )]
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
                                let lr = group.lr(self.global_lr);

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

define_group_optimizer_adaptor!(1, [(O1, 0)]);
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
    use burn::{
        backend::Autodiff,
        nn::{
            Linear,
            LinearConfig,
        },
        optim::{
            AdamW,
            AdamWConfig,
            Sgd,
            SgdConfig,
        },
        tensor::TensorData,
    };

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        default_device,
    };

    type B = Autodiff<CpuBackend>;
    type Net = (Linear<B>, Linear<B>);
    type SgdO = Sgd<CpuBackend>;

    fn net() -> Net {
        let device = default_device();
        (
            LinearConfig::new(3, 3).init(&device),
            LinearConfig::new(3, 2).init(&device),
        )
    }

    fn sgd() -> OptimizerAdaptor<SgdO, Net, B> {
        SgdConfig::new().init()
    }

    fn adamw() -> OptimizerAdaptor<AdamW, Net, B> {
        AdamWConfig::new().init()
    }

    /// `[weight_0, bias_0, weight_1, bias_1]`.
    fn ids(net: &Net) -> [ParamId; 4] {
        [
            net.0.weight.id,
            net.0.bias.as_ref().unwrap().id,
            net.1.weight.id,
            net.1.bias.as_ref().unwrap().id,
        ]
    }

    /// `[weight_0, bias_0, weight_1, bias_1]`.
    fn values(net: &Net) -> [TensorData; 4] {
        [
            net.0.weight.val().into_data(),
            net.0.bias.as_ref().unwrap().val().into_data(),
            net.1.weight.val().into_data(),
            net.1.bias.as_ref().unwrap().val().into_data(),
        ]
    }

    /// Gradients of one backward pass; every parameter gets one.
    fn grads(net: &Net) -> GradientsParams {
        let x = Tensor::<B, 2>::ones([2, 3], &default_device());
        let loss = net.1.forward(net.0.forward(x)).sum();
        GradientsParams::from_grads(loss.backward(), net)
    }

    #[test]
    fn test_new_rejects_duplicate_param_ids() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);

        // Disjoint groups are accepted.
        assert!(
            GroupOptimizerAdaptor2::<SgdO, AdamW, Net, B>::new(
                &net,
                vec![OptimizerGroup::from_adaptor([w0, w1], &sgd())],
                vec![OptimizerGroup::from_adaptor([b0, b1], &adamw())],
            )
            .is_ok()
        );

        // Across optimizer types.
        let err = GroupOptimizerAdaptor2::<SgdO, AdamW, Net, B>::new(
            &net,
            vec![OptimizerGroup::from_adaptor([w0, w1], &sgd())],
            vec![OptimizerGroup::from_adaptor([b0, b1, w1], &adamw())],
        )
        .err()
        .unwrap();
        let GroupOptimizerError::DuplicateParamId {
            param_id,
            first,
            second,
        } = &err
        else {
            panic!("expected DuplicateParamId, got {err:?}");
        };
        assert_eq!((*param_id, *first, *second), (w1, (0, 0), (1, 0)));
        assert!(err.to_string().contains(&w1.to_string()), "{err}");

        // Within one optimizer type.
        let err = GroupOptimizerAdaptor1::<SgdO, Net, B>::new(
            &net,
            vec![
                OptimizerGroup::from_adaptor([w0, b0, w1, b1], &sgd()),
                OptimizerGroup::from_adaptor([w0], &sgd()),
            ],
        )
        .err()
        .unwrap();
        assert!(matches!(
            err,
            GroupOptimizerError::DuplicateParamId {
                first: (0, 0),
                second: (0, 1),
                ..
            }
        ));
    }

    #[test]
    fn test_new_rejects_unassigned_params() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);

        // `b0` and `b1` are in no group.
        let err = GroupOptimizerAdaptor2::<SgdO, AdamW, Net, B>::new(
            &net,
            vec![OptimizerGroup::from_adaptor([w0, w1], &sgd())],
            vec![],
        )
        .err()
        .unwrap();
        let GroupOptimizerError::UnassignedParamIds { param_ids } = &err else {
            panic!("expected UnassignedParamIds, got {err:?}");
        };
        // In the order the module visits them.
        assert_eq!(param_ids, &[b0, b1]);
        let msg = err.to_string();
        assert!(msg.contains(&format!("{b0}, {b1}")), "{msg}");

        // Empty groups are allowed, as long as every parameter has a group.
        assert!(
            GroupOptimizerAdaptor2::<SgdO, AdamW, Net, B>::new(
                &net,
                vec![
                    OptimizerGroup::from_adaptor([w0, w1, b0, b1], &sgd()),
                    OptimizerGroup::from_adaptor([], &sgd()),
                ],
                vec![],
            )
            .is_ok()
        );
    }

    #[test]
    fn test_step_updates_every_group() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let before = values(&net);

        let mut optim = GroupOptimizerAdaptor2::<SgdO, AdamW, Net, B>::new(
            &net,
            vec![OptimizerGroup::from_adaptor([w0, w1], &sgd())],
            vec![OptimizerGroup::from_adaptor([b0, b1], &adamw())],
        )
        .unwrap();

        let grads = grads(&net);
        let net = optim.step(0.1, net, grads);
        let after = values(&net);

        for (i, (a, b)) in after.iter().zip(&before).enumerate() {
            assert_ne!(a, b, "parameter {i} was not stepped");
        }
    }

    #[test]
    fn test_step_uses_each_groups_lr() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let [w0_before, b0_before, w1_before, _] = values(&net);

        // Plain SGD at a fixed rate of 0 does not move, whatever the global
        // rate.
        let mut optim = GroupOptimizerAdaptor1::<SgdO, Net, B>::new(
            &net,
            vec![
                OptimizerGroup::from_adaptor([w0, b0], &sgd()).with_fixed_lr(0.0),
                OptimizerGroup::from_adaptor([w1, b1], &sgd()),
            ],
        )
        .unwrap();

        let grads = grads(&net);
        let net = optim.step(0.1, net, grads);
        let [w0_after, b0_after, w1_after, _] = values(&net);

        assert_eq!(w0_after, w0_before);
        assert_eq!(b0_after, b0_before);
        assert_ne!(w1_after, w1_before);
    }
}
