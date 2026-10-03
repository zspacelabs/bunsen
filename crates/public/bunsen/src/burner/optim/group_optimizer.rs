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
    FrozenOptimizer,
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
/// [`OptimizerGroup::frozen`] builds a group whose parameters are never
/// moved.
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

impl<B> OptimizerGroup<B, FrozenOptimizer>
where
    B: AutodiffBackend,
{
    /// Creates a frozen group: its `params` are never moved.
    ///
    /// The group's optimizer is [`FrozenOptimizer`]. The parameters count as
    /// assigned, so this is how to leave parameters fixed on purpose rather
    /// than out of every group.
    pub fn frozen<I>(params: I) -> Self
    where
        I: IntoIterator<Item = ParamId>,
    {
        Self::new(params.into_iter().collect(), FrozenOptimizer)
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
/// module's float parameters. `new_with_policy` returns it too, but under
/// [`UnknownParamPolicy::Warn`] or [`Freeze`](UnknownParamPolicy::Freeze)
/// never as [`UnassignedParamIds`](Self::UnassignedParamIds).
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
    /// Returned under [`UnknownParamPolicy::Panic`], the default. Every
    /// float parameter must then be in a group, including one that gets no
    /// gradient today (such a parameter is not stepped, whatever its group).
    /// Put parameters to keep fixed in a frozen group
    /// ([`OptimizerGroup::frozen`]), and collect the parameters no other group
    /// claims into a remnant group to cover the rest of the module.
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

/// What a `GroupOptimizerAdaptorN` does with a float parameter that is in no
/// group.
///
/// There are two cases, and the policy governs both:
///
/// - at `new`, a float parameter of the module that no group claims;
/// - at `step`, a float parameter whose `ParamId` the adaptor doesn't know.
///   That happens when the module changed after `new` (surgery, a fresh head),
///   when `Module::load_record` gave its parameters the record's ids, or when
///   `step` gets a different module than `new` did.
///
/// The adaptor never steps such a parameter, and its gradient is dropped;
/// the policy decides whether that is an error. `new` uses the default,
/// [`Panic`](Self::Panic); `new_with_policy` (e.g.
/// [`GroupOptimizerAdaptor2::new_with_policy`]) takes one. To keep
/// parameters fixed on purpose, put them in a frozen group
/// ([`OptimizerGroup::frozen`]) rather than loosening the policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UnknownParamPolicy {
    /// Fail. `new` returns [`GroupOptimizerError::UnassignedParamIds`].
    /// `step`, which can't return an error, panics before it steps anything,
    /// naming the unknown `ParamId`s.
    #[default]
    Panic,

    /// Log a warning, through the `log` crate, the first time each unknown
    /// `ParamId` is seen (at `new` or at `step`), then leave the parameter
    /// unchanged.
    Warn,

    /// Leave the parameter unchanged, silently.
    Freeze,
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

/// Lists the float parameters of `module` that are not in `dispatch`, in
/// visiting order.
fn unassigned_float_params<B, M>(
    module: &M,
    dispatch: &HashMap<ParamId, (usize, usize)>,
) -> Vec<ParamId>
where
    B: Backend,
    M: Module<B>,
{
    let mut float_params = FloatParamIds(Vec::new());
    module.visit(&mut float_params);

    float_params
        .0
        .into_iter()
        .filter(|id| !dispatch.contains_key(id))
        .collect()
}

/// Applies `policy`, for `adaptor`'s `new`, to the module's float parameters
/// in no group.
///
/// Under [`UnknownParamPolicy::Warn`], the warned ids go into `warned`.
fn admit_unassigned(
    adaptor: &str,
    policy: UnknownParamPolicy,
    unassigned: Vec<ParamId>,
    warned: &mut HashSet<ParamId>,
) -> Result<(), GroupOptimizerError> {
    if unassigned.is_empty() {
        return Ok(());
    }
    match policy {
        UnknownParamPolicy::Panic => Err(GroupOptimizerError::UnassignedParamIds {
            param_ids: unassigned,
        }),
        UnknownParamPolicy::Warn => {
            warn_unknown(adaptor, "new", &unassigned, warned);
            Ok(())
        }
        UnknownParamPolicy::Freeze => Ok(()),
    }
}

/// Applies `policy`, for `adaptor`'s `step`, to the float parameters of
/// `module` that are not in `dispatch`.
///
/// # Panics
/// Under [`UnknownParamPolicy::Panic`], if there are any.
fn check_step_params<B, M>(
    adaptor: &str,
    policy: UnknownParamPolicy,
    module: &M,
    dispatch: &HashMap<ParamId, (usize, usize)>,
    warned: &mut HashSet<ParamId>,
) where
    B: Backend,
    M: Module<B>,
{
    if policy == UnknownParamPolicy::Freeze {
        return;
    }
    let unknown = unassigned_float_params(module, dispatch);
    if unknown.is_empty() {
        return;
    }
    match policy {
        UnknownParamPolicy::Panic => panic!(
            "{adaptor}::step: {} float parameter(s) of the module are in no \
             optimizer group, and would never be stepped: {}. The module changed \
             after `new` (surgery, a new head, or `load_record`, which takes the \
             record's ids), or is not the module `new` was given. Build the \
             groups from this module, or pass an `UnknownParamPolicy` to \
             `new_with_policy`.",
            unknown.len(),
            join_param_ids(&unknown),
        ),
        UnknownParamPolicy::Warn => warn_unknown(adaptor, "step", &unknown, warned),
        UnknownParamPolicy::Freeze => {}
    }
}

/// Logs one warning naming the ids in `unknown` not yet in `warned`, and adds
/// them to it; logs nothing if there are none.
fn warn_unknown(
    adaptor: &str,
    site: &str,
    unknown: &[ParamId],
    warned: &mut HashSet<ParamId>,
) {
    let fresh: Vec<ParamId> = unknown
        .iter()
        .copied()
        .filter(|id| warned.insert(*id))
        .collect();
    if !fresh.is_empty() {
        log::warn!(
            "{adaptor}::{site}: {} float parameter(s) of the module are in no \
             optimizer group, and are left unchanged \
             (`UnknownParamPolicy::Warn`; each is reported once): {}",
            fresh.len(),
            join_param_ids(&fresh),
        );
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
                "one group, and `step` that it knows every float parameter of the ",
                "module it is given; the [`UnknownParamPolicy`] decides what a ",
                "parameter in no group does. See the ",
                "[module docs](crate::burner::optim) for the lifecycle and an example.",
                "\n\n# Panics\n\n",
                "`step` panics, under [`UnknownParamPolicy::Panic`] (the default), ",
                "when the module has a float parameter whose `ParamId` is in no group.",
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

                unknown_param_policy: UnknownParamPolicy,

                /// The unknown `ParamId`s already logged under
                /// [`UnknownParamPolicy::Warn`].
                warned_param_ids: HashSet<ParamId>,

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
                /// per optimizer type, in type-parameter order, under the
                /// default [`UnknownParamPolicy::Panic`].
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
                    Self::new_with_policy(
                        module,
                        UnknownParamPolicy::default(),
                        $( [<groups_ $idx>], )+
                    )
                }

                /// Builds the adaptor like `new`, with `policy` for float
                /// parameters in no group, at `new` and at every `step`.
                ///
                /// # Errors
                /// - [`GroupOptimizerError::DuplicateParamId`] if a `ParamId`
                ///   appears in more than one group, of the same type or not.
                /// - [`GroupOptimizerError::UnassignedParamIds`] if a float
                ///   parameter of `module` is in no group and `policy` is
                ///   [`UnknownParamPolicy::Panic`]. Under
                ///   [`Warn`](UnknownParamPolicy::Warn), `new_with_policy`
                ///   logs those parameters and accepts the module; under
                ///   [`Freeze`](UnknownParamPolicy::Freeze), it accepts it.
                #[allow(clippy::too_many_arguments)]
                pub fn new_with_policy(
                    module: &M,
                    policy: UnknownParamPolicy,
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

                    let mut warned_param_ids = HashSet::new();
                    admit_unassigned(
                        stringify!([<GroupOptimizerAdaptor $N>]),
                        policy,
                        unassigned_float_params(module, &dispatch),
                        &mut warned_param_ids,
                    )?;

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
                        unknown_param_policy: policy,
                        warned_param_ids,
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
                    check_step_params(
                        stringify!([<GroupOptimizerAdaptor $N>]),
                        self.unknown_param_policy,
                        &module,
                        &self.dispatch,
                        &mut self.warned_param_ids,
                    );

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

                    // In no group: `check_step_params` has already applied
                    // the `UnknownParamPolicy`, so leave it unchanged.
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
    use std::{
        sync::{
            Mutex,
            Once,
        },
        thread::{
            self,
            ThreadId,
        },
    };

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

    /// `old` with its head replaced by a fresh one, whose `ParamId`s are new.
    fn with_fresh_head(old: Net) -> Net {
        (old.0, net().1)
    }

    /// The warnings this module logs, with the thread that logged each.
    static WARNINGS: Mutex<Vec<(ThreadId, String)>> = Mutex::new(Vec::new());

    /// A `log` logger that keeps this module's warnings in [`WARNINGS`].
    struct WarningLogger;

    impl log::Log for WarningLogger {
        fn enabled(
            &self,
            metadata: &log::Metadata,
        ) -> bool {
            metadata.level() <= log::Level::Warn
                && Some(metadata.target()) == module_path!().strip_suffix("::tests")
        }

        fn log(
            &self,
            record: &log::Record,
        ) {
            if self.enabled(record.metadata()) {
                WARNINGS
                    .lock()
                    .unwrap()
                    .push((thread::current().id(), record.args().to_string()));
            }
        }

        fn flush(&self) {}
    }

    /// The warnings this module logged on this thread since
    /// [`Warnings::start`].
    ///
    /// Tests run in parallel, so each reads only its own thread's warnings.
    struct Warnings {
        thread: ThreadId,
        from: usize,
    }

    impl Warnings {
        /// Installs [`WarningLogger`], once per test binary, and starts here.
        fn start() -> Self {
            static INSTALL: Once = Once::new();
            INSTALL.call_once(|| {
                log::set_logger(&WarningLogger)
                    .expect("nothing else in these tests sets a `log` logger");
                log::set_max_level(log::LevelFilter::Warn);
            });
            Self {
                thread: thread::current().id(),
                from: WARNINGS.lock().unwrap().len(),
            }
        }

        fn messages(&self) -> Vec<String> {
            WARNINGS.lock().unwrap()[self.from..]
                .iter()
                .filter(|(thread, _)| *thread == self.thread)
                .map(|(_, msg)| msg.clone())
                .collect()
        }
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

        // `new` is `new_with_policy` under the default policy, `Panic`.
        assert_eq!(UnknownParamPolicy::default(), UnknownParamPolicy::Panic);
        assert!(matches!(
            GroupOptimizerAdaptor1::<SgdO, Net, B>::new_with_policy(
                &net,
                UnknownParamPolicy::Panic,
                vec![OptimizerGroup::from_adaptor([w0, w1], &sgd())],
            ),
            Err(GroupOptimizerError::UnassignedParamIds { .. })
        ));

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
    fn test_new_warn_accepts_unassigned_params() {
        let warnings = Warnings::start();
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let before = values(&net);

        // `b0` and `b1` are in no group.
        let mut optim = GroupOptimizerAdaptor1::<SgdO, Net, B>::new_with_policy(
            &net,
            UnknownParamPolicy::Warn,
            vec![OptimizerGroup::from_adaptor([w0, w1], &sgd())],
        )
        .unwrap();
        let logged = warnings.messages();
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert!(
            logged[0].contains("GroupOptimizerAdaptor1::new"),
            "{logged:?}"
        );
        assert!(logged[0].contains(&format!("{b0}, {b1}")), "{logged:?}");
        assert!(!logged[0].contains(&w0.to_string()), "{logged:?}");

        // `step` leaves them unchanged, and doesn't warn about them again.
        let grads = grads(&net);
        let net = optim.step(0.1, net, grads);
        let after = values(&net);
        assert_ne!(after[0], before[0]);
        assert_eq!(after[1], before[1]);
        assert_ne!(after[2], before[2]);
        assert_eq!(after[3], before[3]);
        assert_eq!(warnings.messages().len(), 1);
    }

    #[test]
    fn test_new_freeze_accepts_unassigned_params() {
        let warnings = Warnings::start();
        let net = net();
        let [w0, _, w1, _] = ids(&net);
        let before = values(&net);

        // `b0` and `b1` are in no group.
        let mut optim = GroupOptimizerAdaptor1::<SgdO, Net, B>::new_with_policy(
            &net,
            UnknownParamPolicy::Freeze,
            vec![OptimizerGroup::from_adaptor([w0, w1], &sgd())],
        )
        .unwrap();

        let grads = grads(&net);
        let net = optim.step(0.1, net, grads);
        let after = values(&net);
        assert_ne!(after[0], before[0]);
        assert_eq!(after[1], before[1]);
        assert_ne!(after[2], before[2]);
        assert_eq!(after[3], before[3]);
        assert_eq!(warnings.messages(), Vec::<String>::new());
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

    /// The text of a caught panic.
    fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    }

    #[test]
    fn test_step_panics_on_unknown_params() {
        let known = net();
        let mut optim = GroupOptimizerAdaptor1::<SgdO, Net, B>::new(
            &known,
            vec![OptimizerGroup::from_adaptor(ids(&known), &sgd())],
        )
        .unwrap();

        // A second `Net` has fresh `ParamId`s, none of them known to `optim`,
        // and a gradient for each.
        let other = net();
        let other_ids = ids(&other);
        let before = values(&other);
        let grads = grads(&other);
        let stepped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            optim.step(0.1, other, grads)
        }));

        let payload = match stepped {
            Ok(other) => panic!(
                "step returned; were the unknown parameters left unchanged? {}",
                values(&other) == before
            ),
            Err(payload) => payload,
        };
        let msg = panic_message(payload.as_ref());
        for id in other_ids {
            assert!(msg.contains(&id.to_string()), "{msg}");
        }
    }

    #[test]
    fn test_step_warn_logs_each_unknown_param_once() {
        let warnings = Warnings::start();
        let known = net();
        let mut optim = GroupOptimizerAdaptor1::<SgdO, Net, B>::new_with_policy(
            &known,
            UnknownParamPolicy::Warn,
            vec![OptimizerGroup::from_adaptor(ids(&known), &sgd())],
        )
        .unwrap();
        assert_eq!(warnings.messages(), Vec::<String>::new());

        // Surgery: the head's weight and bias are new to `optim`.
        let module = with_fresh_head(known);
        let [w0, _, head_w, head_b] = ids(&module);
        let before = values(&module);
        let module_grads = grads(&module);
        let module = optim.step(0.1, module, module_grads);
        let after = values(&module);
        assert_ne!(after[0], before[0]);
        assert_ne!(after[1], before[1]);
        assert_eq!(after[2], before[2]);
        assert_eq!(after[3], before[3]);

        let logged = warnings.messages();
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert!(
            logged[0].contains("GroupOptimizerAdaptor1::step"),
            "{logged:?}"
        );
        assert!(
            logged[0].contains(&format!("{head_w}, {head_b}")),
            "{logged:?}"
        );
        assert!(!logged[0].contains(&w0.to_string()), "{logged:?}");

        // Stepping again doesn't repeat it.
        let module_grads = grads(&module);
        let module = optim.step(0.1, module, module_grads);
        assert_eq!(warnings.messages().len(), 1);

        // Another new head is reported, alone.
        let module = with_fresh_head(module);
        let [_, _, head_w2, head_b2] = ids(&module);
        let module_grads = grads(&module);
        optim.step(0.1, module, module_grads);
        let logged = warnings.messages();
        assert_eq!(logged.len(), 2, "{logged:?}");
        assert!(
            logged[1].contains(&format!("{head_w2}, {head_b2}")),
            "{logged:?}"
        );
        assert!(!logged[1].contains(&head_w.to_string()), "{logged:?}");
    }

    #[test]
    fn test_step_freeze_leaves_unknown_params_unchanged() {
        let warnings = Warnings::start();
        let known = net();
        let mut optim = GroupOptimizerAdaptor1::<SgdO, Net, B>::new_with_policy(
            &known,
            UnknownParamPolicy::Freeze,
            vec![OptimizerGroup::from_adaptor(ids(&known), &sgd())],
        )
        .unwrap();

        // Surgery: the head's weight and bias are new to `optim`.
        let module = with_fresh_head(known);
        let before = values(&module);
        let module_grads = grads(&module);
        let module = optim.step(0.1, module, module_grads);
        let after = values(&module);
        assert_ne!(after[0], before[0]);
        assert_ne!(after[1], before[1]);
        assert_eq!(after[2], before[2]);
        assert_eq!(after[3], before[3]);
        assert_eq!(warnings.messages(), Vec::<String>::new());
    }

    #[test]
    fn test_frozen_group_is_not_stepped() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let before = values(&net);

        // The head is frozen. A frozen group counts as assigned.
        let mut optim = GroupOptimizerAdaptor2::<SgdO, FrozenOptimizer, Net, B>::new(
            &net,
            vec![OptimizerGroup::from_adaptor([w0, b0], &sgd())],
            vec![OptimizerGroup::frozen([w1, b1])],
        )
        .unwrap();

        let grads = grads(&net);
        let net = optim.step(0.1, net, grads);
        let after = values(&net);
        assert_ne!(after[0], before[0]);
        assert_ne!(after[1], before[1]);
        assert_eq!(after[2], before[2]);
        assert_eq!(after[3], before[3]);

        // It keeps no state.
        let (_, frozen) = optim.to_record();
        assert!(frozen[0].param_map.is_empty());
    }
}
