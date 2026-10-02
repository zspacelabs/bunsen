use burn::{
    module::Module,
    prelude::Backend,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        WithOkOrPanic,
    },
};

/// Lowers an upper *policy* config to the structure config that builds its
/// module.
///
/// The config-side twin of [`ModuleInit`]. Where `ModuleInit` turns a config
/// into a module, `ToStructureConfig` turns a policy config
/// (`FooContractConfig` and the like) into the unrolled `FooStructureConfig` of
/// a Stacked Config family. [`ModuleInit`'s Stacked Config
/// section](ModuleInit#stacked-config) describes the shape and carries a
/// compiled example of both pathways.
///
/// [`Structure`](Self::Structure) names the *lowest* structure config: the one
/// that implements `ModuleInit`. A policy that refines another policy still
/// names that structure config directly; the step from one policy to the next
/// stays an inherent method.
///
/// The trait is in [`crate::prelude`], next to `ModuleInit`.
///
/// # `init` for free
///
/// Every implementor gets `ModuleInit<B, M>` for each module `M` its structure
/// config builds, through a blanket impl whose `try_init` is
/// `self.try_to_structure()?.try_init(device)`. So `policy.init(&device)` and
/// `policy.to_structure().init(&device)` build the same module.
///
/// The blanket impl cannot be overridden: a type that implements both this
/// trait and `ModuleInit` is a conflicting-implementations error (E0119). That
/// is on purpose. A policy builds its module only through its structure
/// config, so the two pathways cannot drift apart. Per-policy logic, including
/// validation, belongs in [`try_to_structure`](Self::try_to_structure).
///
/// ```compile_fail,E0119
/// use bunsen::prelude::*;
/// use burn::{
///     config::Config,
///     module::Module,
///     nn::{
///         Linear,
///         LinearConfig,
///     },
///     prelude::Backend,
/// };
///
/// #[derive(Config, Debug)]
/// pub struct WideStructureConfig {
///     pub layer: LinearConfig,
/// }
///
/// impl<B: Backend> ModuleInit<B, Wide<B>> for WideStructureConfig {
///     fn try_init(
///         &self,
///         device: &B::Device,
///     ) -> BunsenResult<Wide<B>> {
///         Ok(Wide {
///             layer: self.layer.init(device),
///         })
///     }
/// }
///
/// #[derive(Module, Debug)]
/// pub struct Wide<B: Backend> {
///     layer: Linear<B>,
/// }
///
/// #[derive(Config, Debug)]
/// pub struct WideContractConfig {
///     pub width: usize,
/// }
///
/// impl ToStructureConfig for WideContractConfig {
///     type Structure = WideStructureConfig;
///
///     fn try_to_structure(&self) -> BunsenResult<WideStructureConfig> {
///         Ok(WideStructureConfig::new(LinearConfig::new(
///             self.width,
///             2 * self.width,
///         )))
///     }
/// }
///
/// // error[E0119]: conflicting implementations of trait `ModuleInit<_, Wide<_>>`
/// impl<B: Backend> ModuleInit<B, Wide<B>> for WideContractConfig {
///     fn try_init(
///         &self,
///         device: &B::Device,
///     ) -> BunsenResult<Wide<B>> {
///         self.try_to_structure()?.try_init(device)
///     }
/// }
/// ```
pub trait ToStructureConfig {
    /// The structure config this policy lowers to.
    type Structure;

    /// Lowers this policy to its structure config, or reports why the policy
    /// is invalid.
    fn try_to_structure(&self) -> BunsenResult<Self::Structure>;

    /// Lowers this policy to its structure config, panicking with the error's
    /// message if [`try_to_structure`](Self::try_to_structure) fails.
    fn to_structure(&self) -> Self::Structure {
        self.try_to_structure().ok_or_panic()
    }
}

/// A policy config inits through its structure config.
impl<B, M, C> ModuleInit<B, M> for C
where
    B: Backend,
    M: Module<B>,
    C: ToStructureConfig,
    C::Structure: ModuleInit<B, M>,
{
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<M> {
        self.try_to_structure()?.try_init(device)
    }
}
