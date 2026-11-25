//! # The generic conv net
//!
//! [`ConvNetModel`]: a [`ConvStem`], [`ConvStage`]s of [`ResidualUnit`]s,
//! and a [`ConvHead`]. It is the one model type of the residual conv-net
//! families: a `ResNet` is a `ConvNetModel`, built from
//! `ResNetContractConfig`. The idea is timm's `byobnet` (a model is a stem,
//! stage specs and a head, with the layer choices stated once), in bunsen's
//! own types: a unit's kind and every slot are enums of configs and modules,
//! never strings or generics.
//!
//! ## Lifecycle
//!
//! It is a [Stacked config](crate::burner::module::ModuleInit#stacked-config):
//!
//! 1. [`ConvNetContractConfig`] is the policy: a [`ConvStemContractConfig`],
//!    one [`ConvStageContractConfig`] per stage (a unit kind, a depth, widths,
//!    groups, a stride), the [`ConvLayersConfig`] (norm, activation, shortcut)
//!    stated once and replaced per stage, the output stride, and the head's
//!    classes and pool. A family's policy lowers to it, in its own terms.
//! 2. [`to_structure`](crate::burner::module::ToStructureConfig::to_structure)
//!    resolves every unit, once: strides past the output stride become
//!    dilation, each unit kind becomes its conv path. The result is a
//!    [`ConvNetStructureConfig`]: the stem's convs, every unit's
//!    [`ResidualUnitConfig`], the head.
//! 3. `init` builds the [`ConvNetModel`], from the structure or the policy
//!    (through the blanket [`ModuleInit`](crate::burner::module::ModuleInit)),
//!    checking that the tree chains.
//!
//! [`ConvNetMeta`] answers the same questions of the structure and the
//! model, as [`ConvChainMeta`] (every link: the stem, a stage, a unit and its
//! parts) and [`ConvStageMeta`] do
//! of their parts.
//!
//! ## The residual unit
//!
//! Every unit kind ([`ConvUnitKind`]) is one [`ResidualUnit`]: a path of
//! [`ResidualConv`]s, one of them the spatial mixer (the conv that strides;
//! a drop block follows its norm), a channel attention over the path's output
//! when the policy names one, a [`ConvShortcut`] when the path changes the
//! channels or the size, and an activation after the add. The path's
//! last conv has none of its own. A basic unit's path is two `3x3` convs, the
//! first the mixer; a bottleneck's a `1x1`, a `3x3` mixer and a `1x1`.
//!
//! ## Parameter paths
//!
//! The paths are the model's own, not an upstream's; each family's interop
//! rekeys its checkpoints onto them. An enum's variant is not part of a
//! path.
//!
//! | part | path |
//! |---|---|
//! | the stem's `k`th conv and its norm | `stem.convs.{k}.conv`, `stem.convs.{k}.norm` |
//! | stage `i`'s unit `j`: its path's `k`th conv | `stages.{i}.units.{j}.convs.{k}.conv`, `….norm` |
//! | its shortcut | `stages.{i}.units.{j}.shortcut.conv`, `….norm` |
//! | its channel attention | `stages.{i}.units.{j}.attn` (`.fc1`, `.fc2`; `.conv`) |
//! | the classifier | `head.fc` |
//!
//! ## Seams
//!
//! Each slot is an enum config and an enum module, a kind a module of its
//! own ([`blocks`' rule](crate::blocks#design-rules)); a kind not built yet is
//! added as a variant whose lowering is
//! [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported):
//!
//! - the unit kind: [`ConvUnitKind`] (basic, bottleneck);
//! - a path's conv: [`ResidualConvConfig`] (a conv block);
//! - the shortcut: [`ConvShortcutKind`] (a conv and a norm; an average pool,
//!   then a conv and a norm), lowering to one [`ConvShortcutConfig`], an
//!   optional pool before a conv block;
//! - the channel attention: [`ConvAttentionKind`] /
//!   [`ChannelAttentionConfig`](crate::blocks::images::attention::ChannelAttentionConfig)
//!   (squeeze-excite, ECA);
//! - the stem's pool: [`ConvStemPoolKind`] / [`ConvStemPoolConfig`] (max,
//!   none);
//! - the head's pool: [`GlobalPool2dConfig`] (average).

mod head;
mod model;
mod policy;
mod stage;
mod stem;
mod unit;

pub use head::*;
pub use model::*;
pub use policy::*;
pub use stage::*;
pub use stem::*;
pub use unit::*;

#[cfg(test)]
mod tests;
