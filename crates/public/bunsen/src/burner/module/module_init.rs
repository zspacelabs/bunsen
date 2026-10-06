use burn::{
    module::Module,
    tensor::Device,
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
/// `ModuleInit<M>` once for each module type `M` it builds, generic over
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
/// #     support::testing::cpu_device,
/// # };
/// # let device = cpu_device();
/// # let config = MlpConfig::new(16);
/// let mlp: Mlp = config.init(&device);
/// ```
///
/// # Why it is fallible
///
/// [`try_init`](Self::try_init) returns a [`BunsenResult`]. A config is data:
/// it is deserialized from files, assembled by loaders, and edited by users.
/// A config whose fields disagree (an embedding size that does not split into
/// its heads, a depth of zero) is an error to report, so an implementation
/// checks its config in `try_init` and returns an
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) error, usually a
/// [`ConstraintError`](crate::errors::ConstraintError) naming the config and
/// the field, rather than panicking. Code that reads a config from outside
/// the program re-marks the error
/// [`as_policy`](crate::errors::ResultContext::as_policy).
/// [`init`](Self::init) is the convenience for a config you wrote yourself:
/// it panics with the error's report, through
/// [`WithOkOrPanic::ok_or_panic`]. This is the crate-wide pairing of a `try_x`
/// that returns a `BunsenResult` with an `x` that panics.
///
/// # Two config shapes
///
/// A module family's config takes one of two shapes: a Simple Config, or a
/// Stacked Config. Either way the family has a narrow `FooMeta` trait: the
/// values a caller or a test reads back, answered alike by a config and by
/// the module it builds. Code that holds either form asks the same question,
/// and a test can check that a module agrees with the config that built it.
///
/// What the author of a family must do is set by [STYLE.md, "Module
/// design"][style-module-design]: how the configs are named, when to promote
/// a Simple family to Stacked, which types implement `FooMeta`, and the tests
/// each shape requires.
///
/// ## Simple Config
///
/// One config builds the module: `FooConfig` builds `Foo`, and implements
/// `ModuleInit` directly. In the crate,
/// [`MlpConfig`](crate::blocks::transformers::mlp::MlpConfig) →
/// [`Mlp`](crate::blocks::transformers::mlp::Mlp) has this shape.
///
/// ```
/// use bunsen::{
///     errors::ConstraintError,
///     prelude::*,
///     support::testing::cpu_device,
/// };
/// use burn::{
///     config::Config,
///     module::Module,
///     nn::{
///         Linear,
///         LinearConfig,
///     },
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
/// impl ModuleInit<Square> for SquareConfig {
///     fn try_init(
///         &self,
///         device: &Device,
///     ) -> BunsenResult<Square> {
///         if self.width == 0 {
///             return Err(ConstraintError::zero_or_empty(
///                 "SquareConfig",
///                 "width",
///             )
///             .into());
///         }
///         Ok(Square {
///             proj: LinearConfig::new(self.width, self.width).init(device),
///         })
///     }
/// }
///
/// /// A square linear projection.
/// #[derive(Module, Debug)]
/// pub struct Square {
///     proj: Linear,
/// }
///
/// impl SquareMeta for Square {
///     fn width(&self) -> usize {
///         self.proj.weight.dims()[0]
///     }
/// }
///
/// let device = cpu_device();
/// let config = SquareConfig::new(8);
///
/// // The binding names the module, and so the backend.
/// let square: Square = config.init(&device);
/// assert_eq!(square.width(), config.width());
///
/// // A bad config is an error from `try_init`, and a panic from `init`.
/// let bad: BunsenResult<Square> = SquareConfig::new(0).try_init(&device);
/// assert_eq!(bad.unwrap_err().kind(), BunsenErrorKind::Illegal);
/// ```
///
/// ## Stacked Config
///
/// A Stacked Config keeps the knobs a user turns apart from the parameters
/// the implementation needs (a few sizes against a per-layer tree of
/// sub-configs), in two levels:
///
/// - `FooStructureConfig` is the unrolled tree: one field per sub-module
///   config. It implements `ModuleInit` directly, and it is the config that
///   loaders and tooling work with.
/// - At least one upper *policy* config, such as `FooContractConfig`, computes
///   that tree from the user's knobs. ("Contract" here names the user-facing
///   knobs; it is unrelated to [`crate::contracts`].) Several policies may
///   build the same structure, or refine one another.
/// - Each policy implements
///   [`ToStructureConfig`](crate::burner::module::ToStructureConfig), lowering
///   itself to `FooStructureConfig`, and gets `ModuleInit` from that trait's
///   blanket impl. A policy never implements `ModuleInit` itself; the compiler
///   rejects it (E0119), so the two pathways below cannot drift apart.
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
///   `try_init` on the policy returns an
///   [`Illegal`](crate::errors::BunsenErrorKind::Illegal) error for one that
///   does not.
///
/// ```
/// use bunsen::{
///     errors::ConstraintError,
///     prelude::*,
///     support::testing::cpu_device,
/// };
/// use burn::{
///     config::Config,
///     module::Module,
///     nn::{
///         Linear,
///         LinearConfig,
///     },
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
///         if self.depth == 0 {
///             return Err(ConstraintError::zero_or_empty(
///                 "TowerContractConfig",
///                 "depth",
///             )
///             .into());
///         }
///         if self.taper == 0 {
///             return Err(ConstraintError::zero_or_empty(
///                 "TowerContractConfig",
///                 "taper",
///             )
///             .into());
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
/// impl ModuleInit<Tower> for TowerStructureConfig {
///     fn try_init(
///         &self,
///         device: &Device,
///     ) -> BunsenResult<Tower> {
///         Ok(Tower {
///             layers: self.layers.iter().map(|c| c.init(device)).collect(),
///         })
///     }
/// }
///
/// /// A stack of narrowing linear layers.
/// #[derive(Module, Debug)]
/// pub struct Tower {
///     layers: Vec<Linear>,
/// }
///
/// impl TowerMeta for Tower {
///     fn widths(&self) -> Vec<usize> {
///         self.layers.iter().map(|l| l.weight.dims()[1]).collect()
///     }
/// }
///
/// let device = cpu_device();
/// let policy = TowerContractConfig::new(64, 3);
///
/// // Pathway 1: lower to the structure, then build it.
/// let structure = policy.to_structure();
/// assert_eq!(structure.widths(), vec![32, 16, 8]);
/// let lowered: Tower = structure.init(&device);
///
/// // Pathway 2: build from the policy; `init` comes from the blanket impl.
/// let direct: Tower = policy.init(&device);
///
/// assert_eq!(direct.widths(), lowered.widths());
/// assert_eq!(direct.widths(), structure.widths());
///
/// // A bad policy fails in `try_to_structure`; `try_init` passes it on.
/// let bad: BunsenResult<Tower> =
///     TowerContractConfig::new(64, 0).try_init(&device);
/// assert_eq!(
///     bad.unwrap_err().to_string(),
///     "TowerContractConfig.depth: must not be zero or empty"
/// );
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
/// [style-module-design]: https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#module-design
pub trait ModuleInit<M: Module> {
    /// Builds the module on `device`, or reports why the config cannot build
    /// it.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<M>;

    /// Builds the module on `device`, panicking with the error's message if
    /// [`try_init`](Self::try_init) fails.
    fn init(
        &self,
        device: &Device,
    ) -> M {
        self.try_init(device).ok_or_panic()
    }
}
