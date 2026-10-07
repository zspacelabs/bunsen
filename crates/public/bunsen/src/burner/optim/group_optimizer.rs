use std::{
    borrow::Cow,
    fmt,
    sync::Arc,
};

use burn::{
    module::{
        Module,
        ModuleVisitor,
        Param,
        ParamGroup,
        ParamId,
    },
    optim::{
        LearningRate,
        ModuleOptimizer,
        Optimizer,
        grad_clipping::GradientClipping,
        lr_scheduler::module_lr_scheduler::ModuleLrScheduler,
    },
    tensor::{
        Device,
        Tensor,
    },
};
use hashbrown::{
    HashMap,
    HashSet,
};

use crate::{
    burner::optim::{
        FixedLrSelector,
        FrozenOptimizer,
        LrSelector,
        SelectedLr,
    },
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        Detailed,
        WithOkOrPanic,
    },
};

/// Adds a group to a [`ModuleOptimizer`], with the group's optimizer, whose
/// type it erases.
type AddGroup =
    dyn Fn(ModuleOptimizer, ParamGroup, Option<GradientClipping>) -> ModuleOptimizer + Send + Sync;

/// One parameter group: a set of [`ParamId`]s, the optimizer that steps
/// them, and optionally an [`LrSelector`] and a [`GradientClipping`].
///
/// Build one with [`OptimizerGroup::new`], from a `ParamId` set (selected
/// with [`XmlModuleTree`](crate::burner::module::reflection::XmlModuleTree),
/// a visitor, or by hand) and any `burn` [`Optimizer`], such as
/// `AdamWConfig::new().build()`. [`OptimizerGroup::frozen`] builds a group
/// whose parameters are never moved. Pass the groups, with the module they
/// were selected from, to [`GroupOptimizerPlan::try_new`].
///
/// The optimizer's type is erased, so groups of different optimizer types
/// go in one `Vec`. See the [module docs](crate::burner::optim) for the
/// lifecycle.
#[derive(Clone)]
pub struct OptimizerGroup {
    params: HashSet<ParamId>,
    optimizer_name: &'static str,
    add: Arc<AddGroup>,
    lr_selector: Option<Arc<dyn LrSelector>>,
    grad_clipping: Option<GradientClipping>,
}

impl fmt::Debug for OptimizerGroup {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("OptimizerGroup")
            .field("params", &self.params.len())
            .field("optimizer", &self.optimizer_name)
            .field("lr_selector", &self.lr_selector.is_some())
            .field("grad_clipping", &self.grad_clipping.is_some())
            .finish()
    }
}

impl OptimizerGroup {
    /// Creates a group of `params`, stepped by `optim`, at the global
    /// learning rate and without gradient clipping.
    ///
    /// `optim` is a `burn` [`Optimizer`]: what a config's `build()` returns,
    /// e.g. `AdamWConfig::new().build()`.
    pub fn new<O, I>(
        params: I,
        optim: O,
    ) -> Self
    where
        O: Optimizer,
        I: IntoIterator<Item = ParamId>,
    {
        Self {
            params: params.into_iter().collect(),
            optimizer_name: std::any::type_name::<O>(),
            add: Arc::new(move |module_optim, group, grad_clipping| {
                module_optim.with_group(group, optim.clone(), grad_clipping)
            }),
            lr_selector: None,
            grad_clipping: None,
        }
    }

    /// Creates a frozen group: its `params` are never moved.
    ///
    /// The group's optimizer is [`FrozenOptimizer`]. The parameters count as
    /// assigned, so this is how to leave parameters fixed on purpose rather
    /// than out of every group. Their gradients are still computed and then
    /// dropped; [`Module::freeze_group`] is the cheaper path, which computes
    /// none (see [`FrozenOptimizer`]). The two compose: freeze the module, and
    /// put the parameters in a frozen group so the plan accepts them.
    pub fn frozen<I>(params: I) -> Self
    where
        I: IntoIterator<Item = ParamId>,
    {
        Self::new(params, FrozenOptimizer)
    }

    /// Returns the group's parameters.
    pub fn params(&self) -> &HashSet<ParamId> {
        &self.params
    }

    /// Returns the type name of the group's optimizer.
    pub fn optimizer_name(&self) -> &'static str {
        self.optimizer_name
    }

    /// Returns this group's learning rate: `global` mapped through the
    /// [`LrSelector`], or `global` itself when there is none.
    pub fn lr(
        &self,
        global: LearningRate,
    ) -> LearningRate {
        self.lr_selector
            .as_ref()
            .map(|selector| selector.select(global))
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

    /// Returns the group's gradient clipping.
    pub fn grad_clipping(&self) -> Option<&GradientClipping> {
        self.grad_clipping.as_ref()
    }

    /// Sets the group's gradient clipping.
    ///
    /// It applies to each parameter's gradient on its own, before the
    /// group's step: a norm clip bounds each tensor's norm, not a global
    /// norm.
    pub fn with_grad_clipping(
        mut self,
        grad_clipping: GradientClipping,
    ) -> Self {
        self.grad_clipping = Some(grad_clipping);
        self
    }
}

/// The cause of the error [`GroupOptimizerPlan::try_new`] returns: the groups
/// are not a partition of the module's float parameters.
///
/// It converts to a [`BunsenError`] of kind
/// [`Illegal`](BunsenErrorKind::Illegal).
#[derive(Debug, Clone, thiserror::Error)]
pub enum GroupOptimizerError {
    /// A `ParamId` is in more than one optimizer group.
    ///
    /// Groups are numbered by their position in the `Vec` passed to
    /// [`GroupOptimizerPlan::try_new`], from 0.
    #[error(
        "parameter {param_id} is in more than one optimizer group: groups {first} and {second}"
    )]
    DuplicateParamId {
        /// The `ParamId` two groups claim.
        param_id: ParamId,
        /// The index of the first group that claims it.
        first: usize,
        /// The index of the second group that claims it.
        second: usize,
    },

    /// Float parameters of the module are in no optimizer group, so they
    /// would never be stepped.
    ///
    /// Returned under [`UnknownParamPolicy::Panic`], the default. Every
    /// float parameter must then be in a group, including one that gets no
    /// gradient (such a parameter is not stepped, whatever its group). Put
    /// parameters to keep fixed in a frozen group
    /// ([`OptimizerGroup::frozen`]), and collect the parameters no other group
    /// claims into a remnant group to cover the rest of the module.
    #[error(
        "{} float parameter(s) of the module are in no optimizer group, \
         and would never be stepped",
        .params.len()
    )]
    UnassignedParamIds {
        /// Each unassigned parameter's `ParamId` and module path, in the
        /// order the module visits them.
        params: Vec<(ParamId, String)>,
    },

    /// Two float parameters at the same module path are in different
    /// groups.
    ///
    /// The plan keys its groups by module path, so it cannot tell the two
    /// apart. This happens only under a `Module` impl that visits two
    /// parameters without entering a module between them.
    #[error(
        "two float parameters at module path {path:?} are in different \
         optimizer groups: groups {first} and {second}"
    )]
    SharedPath {
        /// The module path the two parameters share.
        path: String,
        /// The index of the group of the first parameter at `path`.
        first: usize,
        /// The index of the group of the second parameter at `path`.
        second: usize,
    },
}

impl Detailed for GroupOptimizerError {
    fn details(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::UnassignedParamIds { params } => Some(Cow::Owned(
                params
                    .iter()
                    .map(|(id, path)| format!("{path} ({id})"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )),
            _ => None,
        }
    }
}

impl From<GroupOptimizerError> for BunsenError {
    #[track_caller]
    fn from(error: GroupOptimizerError) -> Self {
        BunsenError::from_detailed(BunsenErrorKind::Illegal, error)
    }
}

/// What a [`GroupOptimizerPlan`] does with a float parameter that is in no
/// group.
///
/// There are two cases, and the policy governs both:
///
/// - at [`GroupOptimizerPlan::try_new`], a float parameter of the module that
///   no group claims;
/// - at `step`, a float parameter whose module path no group of the plan holds.
///   That happens when the module changed after the plan was built (surgery, a
///   head under a new field), or when `step` gets a different module.
///
/// The second case is the [`ModuleOptimizer`]'s fallback, the optimizer of
/// the parameters no group matches: this policy picks it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UnknownParamPolicy {
    /// Fail. `try_new` returns [`GroupOptimizerError::UnassignedParamIds`].
    /// `step` panics on the first such parameter it reaches, naming its rank
    /// and shape; parameters it reached before are already stepped.
    #[default]
    Panic,

    /// Leave the parameter unchanged, silently: the fallback is
    /// [`FrozenOptimizer`].
    Freeze,
}

/// The fallback optimizer under [`UnknownParamPolicy::Panic`]: its `step`
/// panics.
///
/// A [`ModuleOptimizer`]'s optimizer is not given the `ParamId` it steps, so
/// the message names the rank and shape.
#[derive(Clone, Copy, Debug)]
struct UnknownParamPanic;

impl Optimizer for UnknownParamPanic {
    type State<const D: usize> = ();

    fn step<const D: usize>(
        &self,
        _lr: LearningRate,
        tensor: Tensor<D>,
        _grad: Tensor<D>,
        _state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        panic!(
            "a float parameter of rank {D} and shape {:?} is in no group of the \
             GroupOptimizerPlan that built this optimizer (UnknownParamPolicy::Panic); \
             build the plan over the module that is stepped, after any surgery",
            tensor.dims()
        )
    }

    fn to_device<const D: usize>(
        state: Self::State<D>,
        _device: &Device,
    ) -> Self::State<D> {
        state
    }
}

/// Collects each float parameter of a module with its module path, in
/// visiting order.
///
/// The path is the names of the modules entered, joined with `.`: the
/// convention [`ModuleOptimizer`] and [`ParamGroup`] match paths by.
#[derive(Default)]
struct FloatParamPaths {
    path: Vec<String>,
    params: Vec<(ParamId, String)>,
}

impl ModuleVisitor for FloatParamPaths {
    fn enter_module(
        &mut self,
        name: &str,
        _container_type: &str,
    ) {
        self.path.push(name.to_string());
    }

    fn exit_module(
        &mut self,
        _name: &str,
        _container_type: &str,
    ) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<D>>,
    ) {
        self.params.push((param.id, self.path.join(".")));
    }
}

/// A group of a [`GroupOptimizerPlan`]: the [`OptimizerGroup`], and the
/// [`ParamGroup`] of the module paths of its parameters.
#[derive(Clone, Debug)]
struct PlannedGroup {
    group: OptimizerGroup,
    paths: ParamGroup,
}

/// Several optimizers, and several learning rates, over one module: a
/// builder for `burn`'s [`ModuleOptimizer`] and [`ModuleLrScheduler`].
///
/// [`try_new`](Self::try_new) takes the module and its [`OptimizerGroup`]s,
/// checks that the groups partition the module's float parameters, and keys
/// each group by the module paths of its parameters. Then
/// [`optimizer`](Self::optimizer) builds the [`ModuleOptimizer`] and
/// [`lr_scheduler`](Self::lr_scheduler) the [`ModuleLrScheduler`]: what a
/// [`Learner`](burn::train::Learner) takes, or what a loop of your own steps.
///
/// Groups are selected by `ParamId` and kept by path, because paths survive a
/// checkpoint and the ids a plan was built from do not have to: a resumed run
/// builds a fresh model, with fresh ids, and loading the model record brings
/// back the saved ids. Path-keyed groups match both. See the
/// [module docs](crate::burner::optim) for the lifecycle.
#[derive(Clone, Debug)]
pub struct GroupOptimizerPlan {
    groups: Vec<PlannedGroup>,
    policy: UnknownParamPolicy,
}

impl GroupOptimizerPlan {
    /// Plans `groups` over `module`, under the default
    /// [`UnknownParamPolicy`], `Panic`.
    ///
    /// See [`try_new_with_policy`](Self::try_new_with_policy), and the
    /// [`try_x` / `x` convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    /// `Illegal`, with a [`GroupOptimizerError`] cause:
    /// [`DuplicateParamId`](GroupOptimizerError::DuplicateParamId) when two
    /// groups claim a `ParamId`,
    /// [`UnassignedParamIds`](GroupOptimizerError::UnassignedParamIds) when a
    /// float parameter of `module` is in no group, and
    /// [`SharedPath`](GroupOptimizerError::SharedPath) when two groups hold
    /// parameters at one module path.
    pub fn try_new<M>(
        module: &M,
        groups: Vec<OptimizerGroup>,
    ) -> BunsenResult<Self>
    where
        M: Module,
    {
        Self::try_new_with_policy(module, groups, UnknownParamPolicy::default())
    }

    /// Plans `groups` over `module`, under the default
    /// [`UnknownParamPolicy`], `Panic`.
    ///
    /// The panicking twin of [`try_new`](Self::try_new).
    ///
    /// # Panics
    /// With the report of the error `try_new` returns.
    pub fn new<M>(
        module: &M,
        groups: Vec<OptimizerGroup>,
    ) -> Self
    where
        M: Module,
    {
        Self::try_new(module, groups).ok_or_panic()
    }

    /// Plans `groups` over `module`, under `policy`.
    ///
    /// Groups are numbered by their position in `groups`, from 0, in errors
    /// and in [`groups`](Self::groups).
    ///
    /// - A `ParamId` in two groups is an error.
    /// - A float parameter of `module` in no group is an error under
    ///   [`UnknownParamPolicy::Panic`], and is left unchanged under `Freeze`.
    /// - A `ParamId` that is not a float parameter of `module` (an int or bool
    ///   parameter, or one of another module) is ignored: it has no path in
    ///   `module`. An empty group is not an error.
    ///
    /// Follows the [`try_x` / `x`
    /// convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    /// `Illegal`, with a [`GroupOptimizerError`] cause:
    /// [`DuplicateParamId`](GroupOptimizerError::DuplicateParamId) when two
    /// groups claim a `ParamId`,
    /// [`UnassignedParamIds`](GroupOptimizerError::UnassignedParamIds) when a
    /// float parameter of `module` is in no group under `Panic`, and
    /// [`SharedPath`](GroupOptimizerError::SharedPath) when two groups hold
    /// parameters at one module path.
    pub fn try_new_with_policy<M>(
        module: &M,
        groups: Vec<OptimizerGroup>,
        policy: UnknownParamPolicy,
    ) -> BunsenResult<Self>
    where
        M: Module,
    {
        let mut owner: HashMap<ParamId, usize> = HashMap::new();
        for (index, group) in groups.iter().enumerate() {
            for &param_id in &group.params {
                if let Some(&first) = owner.get(&param_id) {
                    return Err(GroupOptimizerError::DuplicateParamId {
                        param_id,
                        first,
                        second: index,
                    }
                    .into());
                }
                owner.insert(param_id, index);
            }
        }

        let mut visitor = FloatParamPaths::default();
        module.visit(&mut visitor);

        let mut paths: Vec<Vec<String>> = vec![Vec::new(); groups.len()];
        let mut path_owner: HashMap<&str, usize> = HashMap::new();
        let mut unassigned: Vec<(ParamId, String)> = Vec::new();
        for (param_id, path) in &visitor.params {
            let Some(&index) = owner.get(param_id) else {
                if !unassigned.iter().any(|(id, _)| id == param_id) {
                    unassigned.push((*param_id, path.clone()));
                }
                continue;
            };
            match path_owner.get(path.as_str()) {
                Some(&first) if first != index => {
                    return Err(GroupOptimizerError::SharedPath {
                        path: path.clone(),
                        first,
                        second: index,
                    }
                    .into());
                }
                Some(_) => {}
                None => {
                    path_owner.insert(path.as_str(), index);
                    paths[index].push(path.clone());
                }
            }
        }

        if policy == UnknownParamPolicy::Panic && !unassigned.is_empty() {
            return Err(GroupOptimizerError::UnassignedParamIds { params: unassigned }.into());
        }

        let groups = groups
            .into_iter()
            .zip(paths)
            .map(|(group, paths)| PlannedGroup {
                group,
                paths: ParamGroup::from_paths(paths),
            })
            .collect();

        Ok(Self { groups, policy })
    }

    /// Plans `groups` over `module`, under `policy`.
    ///
    /// The panicking twin of
    /// [`try_new_with_policy`](Self::try_new_with_policy).
    ///
    /// # Panics
    /// With the report of the error `try_new_with_policy` returns.
    pub fn new_with_policy<M>(
        module: &M,
        groups: Vec<OptimizerGroup>,
        policy: UnknownParamPolicy,
    ) -> Self
    where
        M: Module,
    {
        Self::try_new_with_policy(module, groups, policy).ok_or_panic()
    }

    /// Returns the plan's [`UnknownParamPolicy`].
    pub fn policy(&self) -> UnknownParamPolicy {
        self.policy
    }

    /// Returns each group, in order, with the [`ParamGroup`] of the module
    /// paths of its parameters.
    ///
    /// The `ParamGroup` serves `burn`'s group APIs, such as
    /// [`Module::freeze_group`].
    pub fn groups(&self) -> impl ExactSizeIterator<Item = (&OptimizerGroup, &ParamGroup)> {
        self.groups.iter().map(|g| (&g.group, &g.paths))
    }

    /// Builds the [`ModuleOptimizer`]: each group's optimizer and gradient
    /// clipping over the group's paths, after a fallback for the parameters
    /// no group holds, which the [`UnknownParamPolicy`] picks.
    ///
    /// Build it from the same plan, or from a plan built the same way, to
    /// load an optimizer record: its states are matched to groups by the
    /// module path saved with each.
    pub fn optimizer(&self) -> ModuleOptimizer {
        let fallback: ModuleOptimizer = match self.policy {
            UnknownParamPolicy::Panic => UnknownParamPanic.into(),
            UnknownParamPolicy::Freeze => FrozenOptimizer.into(),
        };
        self.groups.iter().fold(fallback, |optim, g| {
            (g.group.add)(optim, g.paths.clone(), g.group.grad_clipping.clone())
        })
    }

    /// Builds the [`ModuleLrScheduler`] over the global schedule `base`.
    ///
    /// `base` is what a `burn` scheduler config's `init()` returns, or any
    /// [`LrScheduler`](burn::optim::lr_scheduler::LrScheduler), including a
    /// constant `LearningRate`. Groups without an [`LrSelector`] follow it.
    /// Each group with one follows a [`SelectedLr`] over its own clone of
    /// `base`; the clones step in lockstep.
    ///
    /// The scheduler's record holds one entry per scheduler, matched by
    /// position on load: build the resumed scheduler from a plan with the
    /// same groups, in the same order, over the same base.
    pub fn lr_scheduler(
        &self,
        base: impl Into<ModuleLrScheduler>,
    ) -> ModuleLrScheduler {
        let base = base.into();
        self.groups
            .iter()
            .fold(base.clone(), |scheduler, g| match &g.group.lr_selector {
                Some(selector) => scheduler.with_group(
                    g.paths.clone(),
                    SelectedLr::new(base.clone(), selector.clone()),
                ),
                None => scheduler,
            })
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{
        AssertUnwindSafe,
        catch_unwind,
    };

    use burn::{
        nn::{
            Linear,
            LinearConfig,
        },
        optim::{
            AdamWConfig,
            GradientsParams,
            MuonConfig,
            SgdConfig,
            lr_scheduler::linear::LinearLrSchedulerConfig,
        },
        tensor::{
            TensorData,
            Tolerance,
        },
        train::{
            InferenceStep,
            Learner,
            LearningCheckpointer,
            TrainOutput,
            TrainStep,
            checkpoint::{
                AsyncCheckpointer,
                Checkpointer,
                FileCheckpointer,
                KeepLastNCheckpoints,
            },
        },
    };

    use super::*;
    use crate::{
        errors::testing::{
            ErrorMatcher,
            predicate,
        },
        support::testing::cpu_device,
    };

    #[derive(Module, Debug)]
    struct Net {
        body: Linear,
        head: Linear,
    }

    fn device() -> Device {
        cpu_device().autodiff()
    }

    fn net() -> Net {
        let device = device();
        Net {
            body: LinearConfig::new(3, 3).init(&device),
            head: LinearConfig::new(3, 2).init(&device),
        }
    }

    fn bias(linear: &Linear) -> ParamId {
        linear.bias.as_ref().unwrap().id
    }

    /// `[body.weight, body.bias, head.weight, head.bias]`.
    fn ids(net: &Net) -> [ParamId; 4] {
        [
            net.body.weight.id,
            bias(&net.body),
            net.head.weight.id,
            bias(&net.head),
        ]
    }

    /// `[body.weight, body.bias, head.weight, head.bias]`.
    fn values(net: &Net) -> [TensorData; 4] {
        [
            net.body.weight.val().into_data(),
            net.body.bias.as_ref().unwrap().val().into_data(),
            net.head.weight.val().into_data(),
            net.head.bias.as_ref().unwrap().val().into_data(),
        ]
    }

    /// Gradients of one backward pass; every parameter gets one.
    fn grads(net: &Net) -> GradientsParams {
        let x = Tensor::<2>::ones([2, 3], &device());
        let loss = net.head.forward(net.body.forward(x)).sum();
        GradientsParams::from_grads(loss.backward(), net)
    }

    /// SGD for the weights, `AdamW` for the biases.
    fn sgd_adamw(net: &Net) -> Vec<OptimizerGroup> {
        let [w0, b0, w1, b1] = ids(net);
        vec![
            OptimizerGroup::new([w0, w1], SgdConfig::new().build()),
            OptimizerGroup::new([b0, b1], AdamWConfig::new().build()),
        ]
    }

    fn assert_close(
        a: &TensorData,
        b: &TensorData,
    ) {
        a.assert_approx_eq::<f32>(b, Tolerance::absolute(1e-6));
    }

    #[test]
    fn test_float_param_paths_match_the_module_optimizer() {
        let net = net();
        let mut visitor = FloatParamPaths::default();
        net.visit(&mut visitor);

        let paths: Vec<&str> = visitor.params.iter().map(|(_, p)| p.as_str()).collect();
        assert_eq!(
            paths,
            ["body.weight", "body.bias", "head.weight", "head.bias"]
        );
        let visited: Vec<ParamId> = visitor.params.iter().map(|(id, _)| *id).collect();
        assert_eq!(visited, ids(&net));
    }

    #[test]
    fn test_try_new_rejects_duplicate_param_ids() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let result = GroupOptimizerPlan::try_new(
            &net,
            vec![
                OptimizerGroup::new([w0, w1], SgdConfig::new().build()),
                OptimizerGroup::new([b0, b1], SgdConfig::new().build()),
                OptimizerGroup::new([w1], AdamWConfig::new().build()),
            ],
        );

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .cause(predicate(
                "w1 in groups 0 and 2",
                move |e: &GroupOptimizerError| {
                    matches!(
                        e,
                        GroupOptimizerError::DuplicateParamId {
                            param_id,
                            first: 0,
                            second: 2,
                        } if *param_id == w1
                    )
                },
            ))
            .assert_err(&result);
    }

    #[test]
    fn test_try_new_rejects_unassigned_params() {
        let net = net();
        let [w0, _, w1, _] = ids(&net);
        let result = GroupOptimizerPlan::try_new(
            &net,
            vec![OptimizerGroup::new([w0, w1], SgdConfig::new().build())],
        );

        let expected = vec![
            (bias(&net.body), "body.bias".to_string()),
            (bias(&net.head), "head.bias".to_string()),
        ];
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("2 float parameter(s)")
            .details_contains("head.bias")
            .cause(predicate(
                "the two biases, with their paths",
                move |e: &GroupOptimizerError| {
                    matches!(
                        e,
                        GroupOptimizerError::UnassignedParamIds { params } if *params == expected
                    )
                },
            ))
            .assert_err(&result);
    }

    #[test]
    #[should_panic(expected = "in no optimizer group")]
    fn test_new_panics_on_unassigned_params() {
        let net = net();
        GroupOptimizerPlan::new(&net, vec![]);
    }

    #[test]
    fn test_freeze_accepts_and_freezes_unassigned_params() {
        let net = net();
        let [w0, _, w1, _] = ids(&net);
        let plan = GroupOptimizerPlan::try_new_with_policy(
            &net,
            vec![OptimizerGroup::new([w0, w1], SgdConfig::new().build())],
            UnknownParamPolicy::Freeze,
        )
        .unwrap();
        assert_eq!(plan.policy(), UnknownParamPolicy::Freeze);

        let before = values(&net);
        let grads = grads(&net);
        let after = values(&plan.optimizer().step(0.1, net, grads));

        assert_ne!(after[0], before[0]);
        assert_eq!(after[1], before[1]);
        assert_ne!(after[2], before[2]);
        assert_eq!(after[3], before[3]);
    }

    #[test]
    fn test_try_new_ignores_ids_outside_the_module() {
        let net = net();
        let other = self::net();
        let mut groups = sgd_adamw(&net);
        groups.push(OptimizerGroup::new(ids(&other), SgdConfig::new().build()));

        let plan = GroupOptimizerPlan::try_new(&net, groups).unwrap();
        assert_eq!(plan.groups().len(), 3);
        let (group, paths) = plan.groups().nth(2).unwrap();
        assert_eq!(group.params().len(), 4);
        assert!(!paths.matches(&ids(&other)[0], Some("body.weight")));
    }

    #[test]
    fn test_groups_are_keyed_by_path() {
        let net = net();
        let plan = GroupOptimizerPlan::new(&net, sgd_adamw(&net));
        let fresh = ParamId::new();

        let groups: Vec<_> = plan.groups().collect();
        assert!(groups[0].1.matches(&fresh, Some("body.weight")));
        assert!(groups[0].1.matches(&fresh, Some("head.weight")));
        assert!(!groups[0].1.matches(&net.body.weight.id, Some("body.bias")));
        assert!(groups[1].1.matches(&fresh, Some("head.bias")));
        assert_eq!(
            groups[1].0.optimizer_name(),
            std::any::type_name::<burn::optim::AdamW>()
        );
    }

    /// One step of `optim` over `ids` alone, every other parameter frozen.
    fn step_alone<O: Optimizer>(
        net: &Net,
        ids: Vec<ParamId>,
        optim: O,
    ) -> [TensorData; 4] {
        let mut optim = ModuleOptimizer::from(FrozenOptimizer).with_group(
            ParamGroup::from_ids(ids),
            optim,
            None,
        );
        let grads = grads(net);
        values(&optim.step(0.1, net.clone(), grads))
    }

    #[test]
    fn test_step_updates_every_group_with_mixed_optimizer_types() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let plan = GroupOptimizerPlan::new(
            &net,
            vec![
                OptimizerGroup::new([w0, w1], MuonConfig::new().build()),
                OptimizerGroup::new([b0], AdamWConfig::new().build()),
                OptimizerGroup::new([b1], SgdConfig::new().build()),
            ],
        );

        let before = values(&net);
        let step_grads = grads(&net);
        let after = values(&plan.optimizer().step(0.1, net.clone(), step_grads));
        for (a, b) in after.iter().zip(&before) {
            assert_ne!(a, b);
        }

        // Each group steps as its optimizer alone would.
        let muon = step_alone(&net, vec![w0, w1], MuonConfig::new().build());
        let adamw = step_alone(&net, vec![b0], AdamWConfig::new().build());
        let sgd = step_alone(&net, vec![b1], SgdConfig::new().build());
        assert_close(&after[0], &muon[0]);
        assert_close(&after[2], &muon[2]);
        assert_close(&after[1], &adamw[1]);
        assert_close(&after[3], &sgd[3]);
    }

    #[test]
    fn test_frozen_group_is_not_stepped() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let plan = GroupOptimizerPlan::new(
            &net,
            vec![
                OptimizerGroup::new([w0, w1], SgdConfig::new().build()),
                OptimizerGroup::frozen([b0, b1]),
            ],
        );

        let before = values(&net);
        let grads = grads(&net);
        let after = values(&plan.optimizer().step(0.1, net, grads));

        assert_ne!(after[0], before[0]);
        assert_eq!(after[1], before[1]);
        assert_ne!(after[2], before[2]);
        assert_eq!(after[3], before[3]);
    }

    #[test]
    fn test_step_panics_on_params_outside_the_plan() {
        let net = net();
        let plan = GroupOptimizerPlan::new(&net, sgd_adamw(&net));

        // A model whose head sits under another field: no group holds its
        // paths.
        #[derive(Module, Debug)]
        struct Renamed {
            body: Linear,
            other: Linear,
        }
        let renamed = Renamed {
            body: net.body,
            other: net.head,
        };
        let x = Tensor::<2>::ones([2, 3], &device());
        let loss = renamed.other.forward(renamed.body.forward(x)).sum();
        let grads = GradientsParams::from_grads(loss.backward(), &renamed);

        let mut optim = plan.optimizer();
        let panic = catch_unwind(AssertUnwindSafe(|| optim.step(0.1, renamed, grads)))
            .expect_err("a parameter outside the plan should panic");
        let message = panic.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(message.contains("UnknownParamPolicy::Panic"), "{message}");
    }

    #[test]
    fn test_lr_scheduler_selects_each_groups_lr() {
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let plan = GroupOptimizerPlan::new(
            &net,
            vec![
                OptimizerGroup::new([w0, w1], SgdConfig::new().build()),
                OptimizerGroup::new([b0], SgdConfig::new().build())
                    .with_lr_selector(|lr: LearningRate| lr * 0.5),
                OptimizerGroup::new([b1], SgdConfig::new().build()).with_fixed_lr(0.03),
            ],
        );

        // 1.0 → 0.0 over 4 steps: 1.0, 0.75, ...
        let base = LinearLrSchedulerConfig::new(1.0, 0.0, 4).init().unwrap();
        let mut scheduler = plan.lr_scheduler(base);
        scheduler.step();
        let lr = scheduler.step();

        let id = ParamId::new();
        assert_eq!(lr.lr_from_param(id, Some("body.weight")), 0.75);
        assert_eq!(lr.lr_from_param(id, Some("head.weight")), 0.75);
        assert_eq!(lr.lr_from_param(id, Some("body.bias")), 0.375);
        assert_eq!(lr.lr_from_param(id, Some("head.bias")), 0.03);
    }

    #[test]
    fn test_step_uses_each_groups_lr() {
        // SGD moves a parameter by exactly `lr * grad`, so two copies of one
        // model, stepped at rates 0.1 and 0.05, show the group's rate.
        let net = net();
        let [w0, b0, w1, b1] = ids(&net);
        let plan = GroupOptimizerPlan::new(
            &net,
            vec![
                OptimizerGroup::new([w0, w1, b1], SgdConfig::new().build()),
                OptimizerGroup::new([b0], SgdConfig::new().build())
                    .with_lr_selector(|lr: LearningRate| lr * 0.5),
            ],
        );
        let mut scheduler = plan.lr_scheduler(0.1);

        let before = values(&net);
        let step_grads = grads(&net);
        let body_bias_grad = step_grads.get::<1>(b0).unwrap().into_data();
        let after = values(&plan.optimizer().step(scheduler.step(), net, step_grads));

        let delta: Vec<f32> = before[1]
            .iter::<f32>()
            .zip(after[1].iter::<f32>())
            .map(|(b, a)| b - a)
            .collect();
        let expected: Vec<f32> = body_bias_grad.iter::<f32>().map(|g| g * 0.05).collect();
        assert_close(
            &TensorData::from(delta.as_slice()),
            &TensorData::from(expected.as_slice()),
        );
    }

    /// The resume path: a run saves the model and optimizer records; a
    /// resumed run builds a fresh model (fresh `ParamId`s) and a plan over
    /// it, loads both records, and steps exactly as the first run would
    /// have.
    #[test]
    fn test_optimizer_record_resumes_on_a_fresh_model() {
        let new_plan = |net: &Net| GroupOptimizerPlan::new(net, sgd_adamw(net));

        // The first run: two steps, a checkpoint, then its next step.
        let mut net = net();
        let mut optim = new_plan(&net).optimizer();
        for _ in 0..2 {
            let grads = grads(&net);
            net = optim.step(0.1, net, grads);
        }
        let model_bytes = net.clone().into_record().into_bytes().unwrap();
        let optim_bytes = optim.into_bytes().unwrap();
        let step_grads = grads(&net);
        let expected = values(&optim.step(0.1, net, step_grads));

        // The resumed run: a fresh model, whose ids differ, and a plan over
        // it.
        let fresh = self::net();
        let mut resumed_optim = new_plan(&fresh).optimizer();
        let fresh_ids = ids(&fresh);
        let resumed =
            fresh.load_record(burn::store::ModuleRecord::from_bytes(model_bytes).unwrap());
        assert_ne!(ids(&resumed), fresh_ids);
        resumed_optim = resumed_optim.from_bytes(optim_bytes).unwrap();

        let step_grads = grads(&resumed);
        let actual = values(&resumed_optim.step(0.1, resumed, step_grads));
        for (a, e) in actual.iter().zip(&expected) {
            assert_close(a, e);
        }
    }

    /// Without the record, `AdamW` restarts its moments, and the step
    /// differs: the round trip above carries the per-group state.
    #[test]
    fn test_fresh_optimizer_state_differs_from_resumed() {
        let mut net = net();
        let mut optim = GroupOptimizerPlan::new(&net, sgd_adamw(&net)).optimizer();
        for _ in 0..2 {
            let grads = grads(&net);
            net = optim.step(0.1, net, grads);
        }
        let mut fresh_optim = GroupOptimizerPlan::new(&net, sgd_adamw(&net)).optimizer();

        let grads_a = grads(&net);
        let grads_b = grads(&net);
        let continued = values(&optim.step(0.1, net.clone(), grads_a));
        let restarted = values(&fresh_optim.step(0.1, net, grads_b));

        // The SGD weights agree; the `AdamW` biases do not.
        assert_close(&continued[0], &restarted[0]);
        assert_ne!(continued[1], restarted[1]);
    }

    /// [`Net`] for a `Learner`: its train step is one backward pass of
    /// [`grads`]' loss.
    #[derive(Module, Debug)]
    struct LearnerNet {
        body: Linear,
        head: Linear,
    }

    impl TrainStep for LearnerNet {
        type Input = ();
        type Output = ();

        fn step(
            &self,
            _item: (),
        ) -> TrainOutput<()> {
            let x = Tensor::<2>::ones([2, 3], &self.body.weight.device());
            let loss = self.head.forward(self.body.forward(x)).sum();
            TrainOutput::new(self, loss.backward(), ())
        }
    }

    impl InferenceStep for LearnerNet {
        type Input = ();
        type Output = ();

        fn step(
            &self,
            _item: (),
        ) {
        }
    }

    /// A `Learner` takes the plan's optimizer and scheduler, and resumes
    /// them from a checkpoint saved by an earlier run, over a fresh model.
    #[test]
    fn test_learner_resumes_from_a_checkpoint() {
        let device = device();
        let new_net = || LearnerNet {
            body: LinearConfig::new(3, 3).init(&device),
            head: LinearConfig::new(3, 2).init(&device),
        };
        // SGD for the weights, `AdamW` at half the rate for the biases.
        let new_plan = |net: &LearnerNet| {
            let bias = |linear: &Linear| linear.bias.as_ref().unwrap().id;
            GroupOptimizerPlan::new(
                net,
                vec![
                    OptimizerGroup::new(
                        [net.body.weight.id, net.head.weight.id],
                        SgdConfig::new().build(),
                    ),
                    OptimizerGroup::new(
                        [bias(&net.body), bias(&net.head)],
                        AdamWConfig::new().build(),
                    )
                    .with_lr_selector(|lr: LearningRate| lr * 0.5),
                ],
            )
        };
        let schedule = || LinearLrSchedulerConfig::new(0.4, 0.1, 3).init().unwrap();
        let values = |net: &LearnerNet| {
            [&net.body, &net.head].map(|linear| {
                (
                    linear.weight.val().into_data(),
                    linear.bias.as_ref().unwrap().val().into_data(),
                )
            })
        };

        let dir = tempfile::tempdir().unwrap();
        let model_files = FileCheckpointer::new(dir.path(), "model");
        let optim_files = FileCheckpointer::new(dir.path(), "optim");
        let scheduler_files = FileCheckpointer::new(dir.path(), "scheduler");

        // The first run: a step, a checkpoint at epoch 1, then its next step.
        let net = new_net();
        let plan = new_plan(&net);
        let mut optim = plan.optimizer();
        let mut scheduler = plan.lr_scheduler(schedule());
        let grads = TrainStep::step(&net, ()).grads;
        let net = optim.step(scheduler.step(), net, grads);
        model_files.save(1, net.clone().into_record()).unwrap();
        optim_files.save(1, optim.to_record()).unwrap();
        scheduler_files.save(1, scheduler.to_record()).unwrap();
        let grads = TrainStep::step(&net, ()).grads;
        let expected = values(&optim.step(scheduler.step(), net, grads));

        // The resumed run: a fresh model and plan in a `Learner`, restored by
        // `load_checkpoint`, which a `Learner` resuming at epoch 1 calls.
        let net = new_net();
        let plan = new_plan(&net);
        let learner = Learner::new(net, plan.optimizer(), plan.lr_scheduler(schedule()));
        let checkpointer = LearningCheckpointer::new(
            AsyncCheckpointer::new(model_files),
            AsyncCheckpointer::new(optim_files),
            AsyncCheckpointer::new(scheduler_files),
            Box::new(KeepLastNCheckpoints::new(1)),
        );
        let mut learner = checkpointer.load_checkpoint(learner, 1);
        learner.lr_step();
        let output = learner.train_step(());
        learner.optimizer_step(output.grads);

        for (actual, expected) in values(&learner.model()).iter().zip(&expected) {
            assert_close(&actual.0, &expected.0);
            assert_close(&actual.1, &expected.1);
        }
    }
}
