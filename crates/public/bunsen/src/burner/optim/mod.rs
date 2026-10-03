//! Parameter-group optimizers: several optimizers, and several learning
//! rates, over one model.
//!
//! A `burn` [`Optimizer`](burn::optim::Optimizer) steps every parameter of a
//! module with one rule. Training recipes often want more than that:
//!
//! - 2-D weight matrices train well with [`Muon`](burn::optim::Muon), while
//!   embeddings, norms and biases are better served by
//!   [`AdamW`](burn::optim::AdamW);
//! - groups want their own learning rates: the `NanoChat` recipe scales the
//!   `lm_head` and embedding rates separately, by a `d_model` factor;
//! - groups want their own weight decay or betas.
//!
//! The `GroupOptimizerAdaptorN` family is a single
//! [`Optimizer`](burn::optim::Optimizer) that mounts `N` *types* of
//! optimizer, each with any number of parameter groups, and steps each
//! parameter with the group that claims it.
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
//! 2. **Build [`OptimizerGroup`]s.** A group is a set of `ParamId`s, the
//!    optimizer that steps them, and an optional [`LrSelector`].
//!    [`OptimizerGroup::from_adaptor`] takes the optimizer from what a `burn`
//!    optimizer config's `init()` returns. [`OptimizerGroup::frozen`] builds a
//!    group whose parameters are never moved ([`FrozenOptimizer`]).
//! 3. **Compose with `GroupOptimizerAdaptorN::new`.** Pass the module, then one
//!    `Vec<OptimizerGroup<B, Oi>>` per optimizer type; e.g.
//!    [`GroupOptimizerAdaptor2::new`] takes two. It checks that the groups
//!    partition the module's float parameters: none in two groups, none in no
//!    group. `new_with_policy` (e.g.
//!    [`GroupOptimizerAdaptor2::new_with_policy`]) takes an
//!    [`UnknownParamPolicy`] after the module; `new` uses the default, `Panic`.
//! 4. **Train.** The adaptor implements [`Optimizer`](burn::optim::Optimizer),
//!    so it goes wherever a single optimizer would: a
//!    [`Learner`](burn::train::Learner), or your own loop calling `step`. Step
//!    the module whose `ParamId`s the groups hold: `step` applies the policy to
//!    any float parameter it doesn't know.
//!
//! [`GroupOptimizerAdaptor1`] through [`GroupOptimizerAdaptor7`] exist. Pick
//! the smallest `N` that covers your optimizer *types*: more groups of one
//! type go in that type's `Vec`, and don't raise `N`.
//! `GroupOptimizerAdaptor1` is "one optimizer type, several learning-rate
//! groups".
//!
//! # Behaviour to know
//!
//! - **Duplicates are rejected.** A `ParamId` in two groups, of the same
//!   optimizer type or not, makes `new` return
//!   [`GroupOptimizerError::DuplicateParamId`].
//! - **Every float parameter needs a group.** A float parameter of the module
//!   in no group makes `new` return
//!   [`GroupOptimizerError::UnassignedParamIds`], listing them, under the
//!   default policy (below). This includes a parameter with `require_grad` off,
//!   which gets no gradient and so is not stepped whatever its group. Put
//!   parameters you mean to keep fixed in a frozen group
//!   ([`OptimizerGroup::frozen`]); a frozen group is an optimizer type of its
//!   own, so it takes one of the `N`. Build a remnant group from the parameters
//!   no other group claims.
//! - **`step` checks the module it is given.** The groups hold `ParamId`s, and
//!   a float parameter whose `ParamId` the adaptor doesn't know is never
//!   stepped: its gradient is dropped. That happens when the module changed
//!   after `new` (surgery, a fresh head), when `Module::load_record` gave its
//!   parameters the record's ids, or when `step` gets another module. Select
//!   the groups after surgery and after loading a record. A `Learner` that
//!   resumes from a checkpoint loads the model record after you built the
//!   adaptor, so load that record into the module yourself before selecting.
//! - **The [`UnknownParamPolicy`] decides what a parameter in no group does**,
//!   at `new` and at `step`. `Panic`, the default, fails: `new` returns
//!   `UnassignedParamIds`, and `step` panics before it steps anything, naming
//!   the ids. `Warn` logs each such `ParamId` once, through the `log` crate,
//!   and leaves the parameter unchanged; `Freeze` leaves it unchanged silently.
//!   Under both, `new` accepts the module.
//! - **Empty groups are not errors.** An `XPath` selection that matches nothing
//!   gives an empty group, and the remnant group then claims the parameters it
//!   meant to, with the remnant's settings. Assert that each group is
//!   non-empty.
//! - **Only float parameters are stepped.** Int and bool parameters keep their
//!   values, whatever group they are in, and need no group.
//! - **Gradient clipping**, set with `with_grad_clipping` (e.g.
//!   [`GroupOptimizerAdaptor2::with_grad_clipping`]), applies to each
//!   parameter's gradient on its own, before its group's step: a norm clip
//!   bounds each tensor's norm, not a global norm.
//! - **State is per parameter, per group.** The record is a tuple with one
//!   `Vec<OptimizerGroupRecord>` per optimizer type, one entry per group, each
//!   keyed by `ParamId`. Load a record into an adaptor built with the same
//!   groups, in the same order.
//!
//! # Learning rates
//!
//! Each step receives a *global* learning rate: the `lr` passed to `step`,
//! which a `Learner` takes from its scheduler. Each group maps it through its
//! [`LrSelector`]:
//!
//! - no selector, or [`GlobalLrSelector`]: the global rate;
//! - [`OptimizerGroup::with_fixed_lr`] ([`FixedLrSelector`]): a constant that
//!   ignores the schedule;
//! - a closure, e.g. a per-group factor on the scheduled rate, as in the
//!   example below. A `Send + Sync` closure from the global rate to the group's
//!   rate is an `LrSelector`.
//!
//! # Example
//!
//! Muon for the matrices, `AdamW` at half the scheduled rate for the biases:
//!
//! ```rust
//! use bunsen::{
//!     burner::{
//!         module::reflection::XmlModuleTree,
//!         optim::{
//!             FrozenOptimizer,
//!             GroupOptimizerAdaptor1,
//!             GroupOptimizerAdaptor2,
//!             GroupOptimizerError,
//!             OptimizerGroup,
//!         },
//!     },
//!     public::hashbrown::HashSet,
//! };
//! use burn::{
//!     backend::Autodiff,
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
//!         Optimizer,
//!     },
//!     prelude::Backend,
//!     tensor::Tensor,
//! };
//!
//! #[derive(Module, Debug)]
//! struct Net<B: Backend> {
//!     body: Linear<B>,
//!     head: Linear<B>,
//! }
//!
//! type B = Autodiff<bunsen::support::testing::CpuBackend>;
//! let device = bunsen::support::testing::default_device();
//! let net: Net<B> = Net {
//!     body: LinearConfig::new(4, 4).init(&device),
//!     head: LinearConfig::new(4, 2).init(&device),
//! };
//!
//! // 1. Select. Elements are type names (`Net`, `Linear`); fields are `@name`.
//! let mut mtree = XmlModuleTree::build(&net);
//! let matrices: HashSet<ParamId> = mtree
//!     .select_param_ids("Net/Linear/*[@name='weight'][@rank=2]")?
//!     .into_iter()
//!     .collect();
//! let rest: HashSet<ParamId> = mtree
//!     .param_ids()?
//!     .into_iter()
//!     .filter(|id| !matrices.contains(id))
//!     .collect();
//! assert_eq!((matrices.len(), rest.len()), (2, 2));
//!
//! // 2. Groups.
//! let muon = MuonConfig::new().init::<B, Net<B>>();
//! let adamw = AdamWConfig::new().init::<B, Net<B>>();
//! let matrix_group = OptimizerGroup::from_adaptor(matrices.clone(), &muon);
//! let rest_group = OptimizerGroup::from_adaptor(rest.clone(), &adamw)
//!     .with_lr_selector(|lr| lr * 0.5);
//!
//! // 3. Compose: the module, then one `Vec` of groups per optimizer type.
//! let mut optim: GroupOptimizerAdaptor2<_, _, Net<B>, B> =
//!     GroupOptimizerAdaptor2::new(&net, vec![matrix_group], vec![rest_group])?;
//!
//! // 4. Step it like any `burn` optimizer, or hand it to a `Learner`.
//! let before = net.body.weight.val().into_data();
//! let x = Tensor::<B, 2>::ones([3, 4], &device);
//! let loss = net.head.forward(net.body.forward(x)).sum();
//! let grads = GradientsParams::from_grads(loss.backward(), &net);
//! let net = optim.step(1e-2, net, grads);
//! assert_ne!(net.body.weight.val().into_data(), before);
//!
//! // A parameter claimed by two groups is rejected.
//! let twice: Result<GroupOptimizerAdaptor2<_, _, Net<B>, B>, _> =
//!     GroupOptimizerAdaptor2::new(
//!         &net,
//!         vec![OptimizerGroup::from_adaptor(matrices.clone(), &muon)],
//!         vec![OptimizerGroup::from_adaptor(mtree.param_ids()?, &adamw)],
//!     );
//! assert!(matches!(
//!     twice,
//!     Err(GroupOptimizerError::DuplicateParamId { .. })
//! ));
//!
//! // So is a parameter claimed by none: here, the biases.
//! let partial: Result<GroupOptimizerAdaptor1<_, Net<B>, B>, _> =
//!     GroupOptimizerAdaptor1::new(
//!         &net,
//!         vec![OptimizerGroup::from_adaptor(matrices.clone(), &muon)],
//!     );
//! assert!(matches!(
//!     partial,
//!     Err(GroupOptimizerError::UnassignedParamIds { param_ids }) if param_ids.len() == 2
//! ));
//!
//! // To keep the biases fixed, freeze them in a group of their own.
//! let mut frozen: GroupOptimizerAdaptor2<_, FrozenOptimizer, Net<B>, B> =
//!     GroupOptimizerAdaptor2::new(
//!         &net,
//!         vec![OptimizerGroup::from_adaptor(matrices, &muon)],
//!         vec![OptimizerGroup::frozen(rest)],
//!     )?;
//! let bias = net.body.bias.as_ref().unwrap().val().into_data();
//! let x = Tensor::<B, 2>::ones([3, 4], &device);
//! let loss = net.head.forward(net.body.forward(x)).sum();
//! let grads = GradientsParams::from_grads(loss.backward(), &net);
//! let net = frozen.step(1e-2, net, grads);
//! assert_eq!(net.body.bias.as_ref().unwrap().val().into_data(), bias);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
mod frozen_optimizer;
mod group_optimizer;
mod lr_selectors;

pub use frozen_optimizer::*;
pub use group_optimizer::*;
pub use lr_selectors::*;
