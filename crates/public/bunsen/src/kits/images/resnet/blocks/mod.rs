//! # `ResNet` Blocks
//!
//! The parts a [`ResNet`] is built from, outermost first:
//!
//! - [`ResNet`]: a stem (conv, norm, activation, max-pool), a run of
//!   [`LayerBlock`] stages, and a classifier head (average pool, linear).
//! - [`LayerBlock`]: one stage, a sequence of [`ResidualBlock`]s. The first
//!   block of a stage changes the width or the stride.
//! - [`ResidualBlock`]: one residual unit, an enum over [`BasicBlock`] (two
//!   `3x3` convs) and [`BottleneckBlock`] (a `1x1` pinch, a `3x3` conv, a `1x1`
//!   expand), so that a stage holds one list whichever unit it uses.
//! - [`ResNetDownsampleConfig`]: the projection on a block's identity path,
//!   when the block changes the width or the stride.
//!
//! The three outer levels are [Stacked
//! configs](crate::burner::module::ModuleInit#stacked-config). A
//! `*ContractConfig` policy says what the level means (depths, widths,
//! dilation, bottleneck or not) and lowers through
//! [`ToStructureConfig`](crate::burner::module::ToStructureConfig) to a
//! `*StructureConfig`, the unrolled tree, which builds the module through
//! [`ModuleInit`](crate::burner::module::ModuleInit). The trees nest:
//! [`ResNetContractConfig`] lowers to a [`ResNetStructureConfig`], whose
//! stages are [`LayerBlockStructureConfig`]s, whose blocks are
//! [`ResidualBlockStructureConfig`]s, each a [`BasicBlockConfig`] or a
//! [`BottleneckBlockConfig`]. The two unit blocks are Simple: their configs
//! build them directly.
//!
//! Each level has a narrow Meta trait, implemented by the config that
//! builds it and by the module: [`ResNetMeta`], [`LayerBlockMeta`],
//! [`ResidualBlockMeta`], [`BasicBlockMeta`] and [`BottleneckBlockMeta`].

pub(crate) mod resnet_model;

mod basic_block;
mod bottleneck_block;
mod downsample;
mod layer_block;
mod residual_block;

pub use basic_block::*;
pub use bottleneck_block::*;
pub use downsample::*;
pub use layer_block::*;
pub use residual_block::*;
pub use resnet_model::*;
