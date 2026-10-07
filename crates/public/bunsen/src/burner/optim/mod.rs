//! Parameter-group optimizers: several optimizers, and several learning
//! rates, over one model.
//!
//! A `burn` optimizer config builds one rule for every parameter of a module.
//! Training recipes often want more than that:
//!
//! - 2-D weight matrices train well with [`Muon`](burn::optim::Muon), while
//!   embeddings, norms and biases are better served by
//!   [`AdamW`](burn::optim::AdamW);
//! - groups want their own learning rates: the `NanoChat` recipe scales the
//!   `lm_head` and embedding rates separately, by a `d_model` factor;
//! - groups want their own weight decay, betas or gradient clipping.
//!
//! `burn`'s [`ModuleOptimizer`](burn::optim::ModuleOptimizer) and
//! [`ModuleLrScheduler`](burn::optim::lr_scheduler::module_lr_scheduler::ModuleLrScheduler)
//! do the stepping: each mounts several optimizers (or schedulers) over
//! [`ParamGroup`](burn::module::ParamGroup)s, and a parameter takes the last
//! group that matches it. [`GroupOptimizerPlan`] builds both from parameter
//! groups selected by `ParamId`, and adds what they leave out: a check that
//! the groups partition the module, groups keyed so that they survive a
//! checkpoint, and learning rates derived from one global schedule.
//!
//! This module is behind the `train` feature, which is on by default.
//!
//! # Lifecycle
//!
//! 1. **Select `ParamId`s.** Choose each group's parameters by structure with
//!    [`module::reflection`](crate::burner::module::reflection): build an
//!    [`XmlModuleTree`](crate::burner::module::reflection::XmlModuleTree) over
//!    the live module and select from it with `XPath`
//!    ([`select_param_ids`](crate::burner::module::reflection::XmlModuleTree::select_param_ids)).
//!    Any other way to name the ids works too: a visitor, or the fields
//!    themselves.
//! 2. **Build [`OptimizerGroup`]s.** A group is a set of `ParamId`s, the
//!    optimizer that steps them (any `burn`
//!    [`Optimizer`](burn::optim::Optimizer), such as
//!    `AdamWConfig::new().build()`), and optionally an [`LrSelector`] and a
//!    gradient clipping. [`OptimizerGroup::frozen`] builds a group whose
//!    parameters are never moved ([`FrozenOptimizer`]). Groups of different
//!    optimizer types go in one `Vec`.
//! 3. **Plan with [`GroupOptimizerPlan::try_new`].** It takes the module and
//!    the groups, checks that the groups partition the module's float
//!    parameters (none in two groups, none in no group), and keys each group by
//!    the module paths of its parameters.
//!    [`try_new_with_policy`](GroupOptimizerPlan::try_new_with_policy) takes an
//!    [`UnknownParamPolicy`]; `try_new` uses the default, `Panic`.
//! 4. **Build and train.** [`GroupOptimizerPlan::optimizer`] builds the
//!    `ModuleOptimizer`, and [`GroupOptimizerPlan::lr_scheduler`] the
//!    `ModuleLrScheduler` over a global schedule. Hand both to a
//!    [`Learner`](burn::train::Learner) (`Learner::new(model, plan.optimizer(),
//!    plan.lr_scheduler(schedule))`), or step them in a loop of your own.
//!
//! # Resuming from a checkpoint
//!
//! A plan keys its groups by module path, such as `body.weight`, not by the
//! `ParamId`s they were selected with. Paths are what survive a checkpoint:
//!
//! - A module record saves each parameter's id, and `Module::load_record` gives
//!   the parameter that id back. So a restored model has the saved run's ids,
//!   not the ones it was built with.
//! - An optimizer record saves each parameter's state keyed by that id, with
//!   its module path, and `ModuleOptimizer::load_record` hands each state to
//!   the group that matches the saved path.
//!
//! So a resumed run is built like a first run: build a fresh model, select
//! its groups, and plan them, with the same groups in the same order as the
//! run that saved the checkpoint. Its fresh ids select the same paths. Then
//! load the module record into the model and the optimizer record into
//! `plan.optimizer()`. A [`Learner`](burn::train::Learner) resuming from a
//! checkpoint does the loading.
//!
//! The scheduler's record is matched by position: build the resumed
//! scheduler from the same plan shape, over the same base schedule.
//!
//! ```rust
//! use bunsen::{
//!     burner::optim::{
//!         GroupOptimizerPlan,
//!         OptimizerGroup,
//!     },
//!     errors::BunsenResult,
//!     support::testing::cpu_device,
//! };
//! use burn::{
//!     module::Module,
//!     nn::{
//!         Linear,
//!         LinearConfig,
//!     },
//!     optim::{
//!         AdamWConfig,
//!         GradientsParams,
//!         ModuleOptimizer,
//!         SgdConfig,
//!     },
//!     store::ModuleRecord,
//!     tensor::{
//!         Device,
//!         Tensor,
//!     },
//! };
//!
//! /// Builds the model and its optimizer: the same code starts a run and
//! /// resumes one. SGD steps the weight, `AdamW` the bias.
//! fn build(device: &Device) -> BunsenResult<(Linear, ModuleOptimizer)> {
//!     let net: Linear = LinearConfig::new(4, 2).init(device);
//!     let plan = GroupOptimizerPlan::try_new(
//!         &net,
//!         vec![
//!             OptimizerGroup::new([net.weight.id], SgdConfig::new().build()),
//!             OptimizerGroup::new(
//!                 [net.bias.as_ref().unwrap().id],
//!                 AdamWConfig::new().build(),
//!             ),
//!         ],
//!     )?;
//!     Ok((net, plan.optimizer()))
//! }
//!
//! fn train_step(
//!     net: Linear,
//!     optim: &mut ModuleOptimizer,
//!     device: &Device,
//! ) -> Linear {
//!     let loss = net.forward(Tensor::<2>::ones([3, 4], device)).sum();
//!     let grads = GradientsParams::from_grads(loss.backward(), &net);
//!     optim.step(1e-2, net, grads)
//! }
//!
//! let device = cpu_device().autodiff();
//!
//! // The first run trains, and saves a checkpoint.
//! let (net, mut optim) = build(&device)?;
//! let net = train_step(net, &mut optim, &device);
//! let model_bytes = net.into_record().into_bytes()?;
//! let optim_bytes = optim.into_bytes()?;
//!
//! // The resumed run builds a fresh model, with fresh `ParamId`s, and a plan
//! // over it, then loads both records. The `AdamW` state lands on the bias.
//! let (net, optim) = build(&device)?;
//! let net = net.load_record(ModuleRecord::from_bytes(model_bytes)?);
//! let mut optim = optim.from_bytes(optim_bytes)?;
//!
//! let before = net.weight.val().into_data();
//! let net = train_step(net, &mut optim, &device);
//! assert_ne!(net.weight.val().into_data(), before);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Behaviour to know
//!
//! - **Duplicates are rejected.** A `ParamId` in two groups makes `try_new`
//!   return [`GroupOptimizerError::DuplicateParamId`].
//! - **Every float parameter needs a group.** A float parameter of the module
//!   in no group makes `try_new` return
//!   [`GroupOptimizerError::UnassignedParamIds`], listing each one with its
//!   path, under the default policy (below). This includes a parameter with
//!   `require_grad` off, which gets no gradient and so is not stepped whatever
//!   its group. Put parameters you mean to keep fixed in a frozen group
//!   ([`OptimizerGroup::frozen`]). Build a remnant group from the parameters no
//!   other group claims.
//! - **The plan is keyed by paths.** A parameter is stepped by the group that
//!   holds its module path. A model with the same structure as the planned one,
//!   such as a freshly built or a restored one, is stepped the same way. A
//!   parameter at a path the plan doesn't hold, such as a head under a new
//!   field, goes to the fallback, which the [`UnknownParamPolicy`] picks. Plan
//!   the groups after surgery.
//! - **The [`UnknownParamPolicy`] decides what a parameter in no group does.**
//!   `Panic`, the default, fails: `try_new` returns `UnassignedParamIds`, and
//!   `step` panics on the first such parameter, naming its rank and shape.
//!   `Freeze` accepts the module and leaves such parameters unchanged.
//! - **Empty groups are not errors.** An `XPath` selection that matches nothing
//!   gives an empty group, and the remnant group then claims the parameters it
//!   meant to, with the remnant's settings. Assert that each group is
//!   non-empty. A `ParamId` that is not a float parameter of the module is
//!   ignored.
//! - **Only float parameters are stepped.** Int and bool parameters keep their
//!   values, whatever group they are in, and need no group.
//! - **Gradient clipping is per group**
//!   ([`OptimizerGroup::with_grad_clipping`]), and applies to each parameter's
//!   gradient on its own, before its group's step: a norm clip bounds each
//!   tensor's norm, not a global norm.
//!
//! # Learning rates
//!
//! [`GroupOptimizerPlan::lr_scheduler`] takes the *global* schedule, any
//! `burn` [`LrScheduler`](burn::optim::lr_scheduler::LrScheduler) (a constant
//! `LearningRate` is one), and each group maps its rate through its
//! [`LrSelector`]:
//!
//! - no selector, or [`GlobalLrSelector`]: the global rate;
//! - [`OptimizerGroup::with_fixed_lr`] ([`FixedLrSelector`]): a constant that
//!   ignores the schedule;
//! - a closure, e.g. a per-group factor on the scheduled rate, as in the
//!   example below. A `Send + Sync` closure from the global rate to the group's
//!   rate is an `LrSelector`.
//!
//! A group with a selector gets a [`SelectedLr`] scheduler: the global
//! schedule, stepped in lockstep, mapped through the selector.
//!
//! # Example
//!
//! Muon for the matrices, `AdamW` at half the scheduled rate for the biases:
//!
//! ```rust
//! use bunsen::{
//!     burner::optim::{
//!         GroupOptimizerError,
//!         GroupOptimizerPlan,
//!         OptimizerGroup,
//!     },
//!     support::testing::cpu_device,
//! };
//! use burn::{
//!     module::{
//!         Module,
//!         ParamId,
//!     },
//!     nn::{
//!         Linear,
//!         LinearConfig,
//!     },
//!     optim::{
//!         AdamWConfig,
//!         GradientsParams,
//!         MuonConfig,
//!     },
//!     tensor::Tensor,
//! };
//!
//! #[derive(Module, Debug)]
//! struct Net {
//!     body: Linear,
//!     head: Linear,
//! }
//!
//! let device = cpu_device().autodiff();
//! let net: Net = Net {
//!     body: LinearConfig::new(4, 4).init(&device),
//!     head: LinearConfig::new(4, 2).init(&device),
//! };
//!
//! // 1. Select.
//! let matrices: Vec<ParamId> = vec![net.body.weight.id, net.head.weight.id];
//! let biases: Vec<ParamId> = [&net.body, &net.head]
//!     .map(|linear| linear.bias.as_ref().unwrap().id)
//!     .into();
//!
//! // 2. Groups.
//! let matrix_group = OptimizerGroup::new(matrices.clone(), MuonConfig::new().build());
//! let bias_group = OptimizerGroup::new(biases.clone(), AdamWConfig::new().build())
//!     .with_lr_selector(|lr| lr * 0.5);
//!
//! // 3. Plan: the module, then the groups.
//! let plan = GroupOptimizerPlan::try_new(&net, vec![matrix_group, bias_group])?;
//!
//! // 4. Build, and step; or hand both to a `Learner`.
//! let mut optim = plan.optimizer();
//! let mut scheduler = plan.lr_scheduler(1e-2);
//! let before = net.body.weight.val().into_data();
//! let x = Tensor::<2>::ones([3, 4], &device);
//! let loss = net.head.forward(net.body.forward(x)).sum();
//! let grads = GradientsParams::from_grads(loss.backward(), &net);
//! let net = optim.step(scheduler.step(), net, grads);
//! assert_ne!(net.body.weight.val().into_data(), before);
//!
//! // A parameter claimed by two groups is rejected.
//! let twice = GroupOptimizerPlan::try_new(
//!     &net,
//!     vec![
//!         OptimizerGroup::new(matrices.clone(), MuonConfig::new().build()),
//!         OptimizerGroup::new(matrices.clone(), AdamWConfig::new().build()),
//!     ],
//! );
//! assert!(matches!(
//!     twice.unwrap_err().find::<GroupOptimizerError>(),
//!     Some(GroupOptimizerError::DuplicateParamId { .. })
//! ));
//!
//! // So is a parameter claimed by none: here, the biases.
//! let partial = GroupOptimizerPlan::try_new(
//!     &net,
//!     vec![OptimizerGroup::new(matrices.clone(), MuonConfig::new().build())],
//! );
//! assert!(matches!(
//!     partial.unwrap_err().find::<GroupOptimizerError>(),
//!     Some(GroupOptimizerError::UnassignedParamIds { params }) if params.len() == 2
//! ));
//!
//! // To keep the biases fixed, freeze them in a group of their own.
//! let frozen = GroupOptimizerPlan::try_new(
//!     &net,
//!     vec![
//!         OptimizerGroup::new(matrices, MuonConfig::new().build()),
//!         OptimizerGroup::frozen(biases),
//!     ],
//! )?;
//! let bias = net.body.bias.as_ref().unwrap().val().into_data();
//! let x = Tensor::<2>::ones([3, 4], &device);
//! let loss = net.head.forward(net.body.forward(x)).sum();
//! let grads = GradientsParams::from_grads(loss.backward(), &net);
//! let net = frozen.optimizer().step(1e-2, net, grads);
//! assert_eq!(net.body.bias.as_ref().unwrap().val().into_data(), bias);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
mod frozen_optimizer;
mod group_optimizer;
mod lr_selectors;

pub use frozen_optimizer::*;
pub use group_optimizer::*;
pub use lr_selectors::*;
