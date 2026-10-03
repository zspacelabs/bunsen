use burn::{
    module::Module,
    prelude::Backend,
};

use crate::errors::{
    BunsenResult,
    WithOkOrPanic,
};

/// Builds a [`Module`] from a config: the Config → Module step of bunsen's
/// config lifecycle.
///
/// A config is plain, serializable data (a `#[derive(Config)]` struct); the
/// module it builds owns the tensors. The config implements
/// `ModuleInit<B, M>` once for each module type `M` it builds, generic over
/// the backend `B`, so one config builds its module on any backend.
///
/// The trait is in [`crate::prelude`]. Bring it into scope there, or from
/// `bunsen::burner::module`, to call `.init(&device)` on a bunsen config.
///
/// # Calling it
///
/// `B` and `M` are parameters of the trait, and a device does not name its
/// backend, so the module type comes from the binding:
///
/// ```
/// # use bunsen::{
/// #     blocks::transformers::mlp::{
/// #         Mlp,
/// #         MlpConfig,
/// #     },
/// #     prelude::*,
/// #     support::testing::default_device,
/// # };
/// # type B = burn::backend::Flex;
/// # let device = default_device();
/// # let config = MlpConfig::new(16);
/// let mlp: Mlp<B> = config.init(&device);
/// ```
///
/// # Why it is fallible
///
/// [`try_init`](Self::try_init) returns a [`BunsenResult`]. A config is data:
/// it is deserialized from files, assembled by loaders, and edited by users.
/// A config whose fields disagree (an embedding size that does not split into
/// its heads, a depth of zero) is an input error to report, not a bug, so an
/// implementation checks its config in `try_init` and returns
/// [`BunsenError::Invalid`](crate::errors::BunsenError::Invalid) rather than
/// panicking. [`init`](Self::init) is the convenience for a config you wrote
/// yourself: it panics with the error's message, through
/// [`WithOkOrPanic::ok_or_panic`]. This is the crate-wide pairing of a `try_x`
/// that returns a `BunsenResult` with an `x` that panics.
///
/// # Two config shapes
///
/// A module family picks one of two shapes: a Simple Config, or a Stacked
/// Config. Both put the config in the module's file, and both give the family
/// a narrow `FooMeta` trait: only the values a caller or a test needs to read
/// back, implemented by a config and by the module it builds, so a test can
/// check that a module agrees with the config that built it.
///
/// ## Simple Config
///
/// `FooConfig` builds `Foo`, and implements `ModuleInit` directly. `FooMeta`
/// is implemented by `FooConfig` and `Foo`. In the crate,
/// [`MlpConfig`](crate::blocks::transformers::mlp::MlpConfig) →
/// [`Mlp`](crate::blocks::transformers::mlp::Mlp) has this shape.
///
/// ```
/// use bunsen::{
///     prelude::*,
///     support::testing::default_device,
/// };
/// use burn::{
///     backend::Flex,
///     config::Config,
///     module::Module,
///     nn::{
///         Linear,
///         LinearConfig,
///     },
///     prelude::Backend,
/// };
///
/// /// The narrow view shared by [`SquareConfig`] and [`Square`].
/// pub trait SquareMeta {
///     /// The input and output width.
///     fn width(&self) -> usize;
/// }
///
/// /// Config for [`Square`].
/// #[derive(Config, Debug)]
/// pub struct SquareConfig {
///     /// The input and output width.
///     pub width: usize,
/// }
///
/// impl SquareMeta for SquareConfig {
///     fn width(&self) -> usize {
///         self.width
///     }
/// }
///
/// impl<B: Backend> ModuleInit<B, Square<B>> for SquareConfig {
///     fn try_init(
///         &self,
///         device: &B::Device,
///     ) -> BunsenResult<Square<B>> {
///         if self.width == 0 {
///             return Err(BunsenError::Invalid(
///                 "width must be > 0".to_string(),
///             ));
///         }
///         Ok(Square {
///             proj: LinearConfig::new(self.width, self.width).init(device),
///         })
///     }
/// }
///
/// /// A square linear projection.
/// #[derive(Module, Debug)]
/// pub struct Square<B: Backend> {
///     proj: Linear<B>,
/// }
///
/// impl<B: Backend> SquareMeta for Square<B> {
///     fn width(&self) -> usize {
///         self.proj.weight.dims()[0]
///     }
/// }
///
/// let device = default_device();
/// let config = SquareConfig::new(8);
///
/// // The binding names the module, and so the backend.
/// let square: Square<Flex> = config.init(&device);
/// assert_eq!(square.width(), config.width());
///
/// // A bad config is an error from `try_init`, and a panic from `init`.
/// let bad: BunsenResult<Square<Flex>> =
///     SquareConfig::new(0).try_init(&device);
/// assert!(bad.is_err());
/// ```
///
/// ## Stacked Config
///
/// When the knobs a user turns are not the parameters the implementation
/// needs (a few sizes against a per-layer tree of sub-configs), the family
/// splits its config in two levels:
///
/// - `FooStructureConfig` is the unrolled tree: one field per sub-module
///   config. It implements `ModuleInit` directly, and it is the config that
///   loaders and tooling work with.
/// - At least one upper *policy* config computes that tree from the user's
///   knobs. A policy config is named for the policy it encodes
///   (`FooContractConfig`, `FooApiConfig`, `FooSignalConfig`), never bare
///   `FooConfig`, which means the Simple shape. ("Contract" here names the
///   user-facing knobs; it is unrelated to [`crate::contracts`].) Several
///   policies may build the same structure.
/// - Each policy implements
///   [`ToStructureConfig`](crate::burner::module::ToStructureConfig), lowering
///   itself to `FooStructureConfig`, and gets `ModuleInit` from that trait's
///   blanket impl. A policy never implements `ModuleInit` itself; the compiler
///   rejects it (E0119), so the two pathways below cannot drift apart.
/// - `FooMeta` is required on `FooStructureConfig` and `Foo`, and optional on
///   the policies.
///
/// A policy builds its module by either pathway, with the same result:
///
/// 1. `policy.to_structure()`, then `.init(&device)` on the structure: lower
///    first, to inspect or edit the tree before building;
/// 2. `policy.init(&device)`: the blanket impl's short circuit, which is
///    `policy.try_to_structure()?.try_init(device)`.
///
/// The crate's Stacked families show the variations:
///
/// - [`ResNetContractConfig`] is a single policy over [`ResNetStructureConfig`]
///   and [`ResNet`].
/// - [`SileroVadSignalConfig`] and [`SileroVadStftConfig`] are two chained
///   policies: [`to_stft`] refines the signal policy into the STFT one, and
///   both lower straight to [`SileroVadStructureConfig`].
/// - [`SwinTransformerV2ContractConfig`] has a fallible lowering: its
///   `try_to_structure` checks that the stages fit the input and the window, so
///   `try_init` on the policy returns
///   [`BunsenError::Invalid`](crate::errors::BunsenError::Invalid) for one that
///   does not.
///
/// ```
/// use bunsen::{
///     prelude::*,
///     support::testing::default_device,
/// };
/// use burn::{
///     backend::Flex,
///     config::Config,
///     module::Module,
///     nn::{
///         Linear,
///         LinearConfig,
///     },
///     prelude::Backend,
/// };
///
/// /// The narrow view shared by [`TowerStructureConfig`] and [`Tower`].
/// pub trait TowerMeta {
///     /// The output width of each layer, in order.
///     fn widths(&self) -> Vec<usize>;
/// }
///
/// /// The policy: a width, a depth, and how fast the tower narrows.
/// #[derive(Config, Debug)]
/// pub struct TowerContractConfig {
///     /// The input width.
///     pub width: usize,
///
///     /// The number of layers.
///     pub depth: usize,
///
///     /// Each layer divides the width by this.
///     #[config(default = "2")]
///     pub taper: usize,
/// }
///
/// impl ToStructureConfig for TowerContractConfig {
///     type Structure = TowerStructureConfig;
///
///     fn try_to_structure(&self) -> BunsenResult<TowerStructureConfig> {
///         if self.depth == 0 || self.taper == 0 {
///             return Err(BunsenError::Invalid(
///                 "depth and taper must be > 0".to_string(),
///             ));
///         }
///         let mut layers = Vec::new();
///         let mut d_input = self.width;
///         for _ in 0..self.depth {
///             let d_output = (d_input / self.taper).max(1);
///             layers.push(LinearConfig::new(d_input, d_output));
///             d_input = d_output;
///         }
///         Ok(TowerStructureConfig::new(layers))
///     }
/// }
///
/// /// The structure: one sub-config per layer.
/// #[derive(Config, Debug)]
/// pub struct TowerStructureConfig {
///     /// The layer configs, in order.
///     pub layers: Vec<LinearConfig>,
/// }
///
/// impl TowerMeta for TowerStructureConfig {
///     fn widths(&self) -> Vec<usize> {
///         self.layers.iter().map(|c| c.d_output).collect()
///     }
/// }
///
/// impl<B: Backend> ModuleInit<B, Tower<B>> for TowerStructureConfig {
///     fn try_init(
///         &self,
///         device: &B::Device,
///     ) -> BunsenResult<Tower<B>> {
///         Ok(Tower {
///             layers: self.layers.iter().map(|c| c.init(device)).collect(),
///         })
///     }
/// }
///
/// /// A stack of narrowing linear layers.
/// #[derive(Module, Debug)]
/// pub struct Tower<B: Backend> {
///     layers: Vec<Linear<B>>,
/// }
///
/// impl<B: Backend> TowerMeta for Tower<B> {
///     fn widths(&self) -> Vec<usize> {
///         self.layers.iter().map(|l| l.weight.dims()[1]).collect()
///     }
/// }
///
/// let device = default_device();
/// let policy = TowerContractConfig::new(64, 3);
///
/// // Pathway 1: lower to the structure, then build it.
/// let structure = policy.to_structure();
/// assert_eq!(structure.widths(), vec![32, 16, 8]);
/// let lowered: Tower<Flex> = structure.init(&device);
///
/// // Pathway 2: build from the policy; `init` comes from the blanket impl.
/// let direct: Tower<Flex> = policy.init(&device);
///
/// assert_eq!(direct.widths(), lowered.widths());
/// assert_eq!(direct.widths(), structure.widths());
///
/// // A bad policy fails in `try_to_structure`; `try_init` passes it on.
/// let bad: BunsenResult<Tower<Flex>> =
///     TowerContractConfig::new(64, 0).try_init(&device);
/// assert!(bad.is_err());
/// ```
///
/// # Hand-written `init`
///
/// A constructor that needs more than a device keeps an inherent `init` (and
/// `try_init`) instead of this trait:
///
/// - a layer built for its place in a stack:
///   [`CausalSelfAttentionConfig::init`] and [`NanoChatGptBlockConfig::init`]
///   take a `layer_index`;
/// - per-stream state that serves a model rather than standing alone:
///   [`KVCacheConfig::init`] takes no device, and
///   [`SileroVadContextConfig::init`] takes the model it serves;
/// - simulation state seeded with a value: [`LBMD2Q9Config::init`] takes the
///   initial density `rho`.
///
/// [`CausalSelfAttentionConfig::init`]: crate::blocks::transformers::attention::csa::CausalSelfAttentionConfig::init
/// [`NanoChatGptBlockConfig::init`]: crate::kits::gpts::nanochat::blocks::NanoChatGptBlockConfig::init
/// [`KVCacheConfig::init`]: crate::ops::transformers::attention::KVCacheConfig::init
/// [`SileroVadContextConfig::init`]: crate::kits::speech::silero_vad::blocks::SileroVadContextConfig::init
/// [`LBMD2Q9Config::init`]: crate::kits::sims::lbm::d2q9::LBMD2Q9Config::init
/// [`ResNetContractConfig`]: crate::kits::images::resnet::ResNetContractConfig
/// [`ResNetStructureConfig`]: crate::kits::images::resnet::ResNetStructureConfig
/// [`ResNet`]: crate::kits::images::resnet::ResNet
/// [`SileroVadSignalConfig`]: crate::kits::speech::silero_vad::blocks::SileroVadSignalConfig
/// [`SileroVadStftConfig`]: crate::kits::speech::silero_vad::blocks::SileroVadStftConfig
/// [`SileroVadStructureConfig`]: crate::kits::speech::silero_vad::blocks::SileroVadStructureConfig
/// [`to_stft`]: crate::kits::speech::silero_vad::blocks::SileroVadSignalConfig::to_stft
/// [`SwinTransformerV2ContractConfig`]: crate::kits::images::swin::v2::SwinTransformerV2ContractConfig
pub trait ModuleInit<B: Backend, M: Module<B>> {
    /// Builds the module on `device`, or reports why the config cannot build
    /// it.
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<M>;

    /// Builds the module on `device`, panicking with the error's message if
    /// [`try_init`](Self::try_init) fails.
    fn init(
        &self,
        device: &B::Device,
    ) -> M {
        self.try_init(device).ok_or_panic()
    }
}
