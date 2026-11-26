//! # The generic conv net
//!
//! [`ConvNetModel`]: a [`ConvStem`], [`ConvStage`]s of [`ResidualUnit`]s, a
//! [`GlobalPool2d`], and the image models'
//! [`ClassifierHead`](crate::blocks::images::models::classifier_head::ClassifierHead).
//! It is the one model type of the residual conv-net
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
//!    [`ResidualUnitConfig`], the pool and the head.
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
//! a drop block follows its norm), a channel attention when the policy names
//! one (over the path's output, or after the mixer: [`ConvAttentionPlace`]), a
//! [`ConvShortcut`] when the path changes the channels or the size, and an
//! activation after the add. The path's last conv has none of its own. A basic
//! unit's path is two `3x3` convs, the first the mixer; a bottleneck's a `1x1`,
//! a `3x3` mixer and a `1x1`. A pre-activation unit normalizes and activates
//! its input once ([`NormAct`]), read by the path and the shortcut, and each
//! later conv after its own norm ([`PreActConv`]); it has no activation after
//! the add. A unit with no residual is its path alone. After the last stage,
//! the net may have a final layer before the pool ([`ConvFinal`]: an
//! activation, or a conv block).
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
//! | its pre-activation | `stages.{i}.units.{j}.pre.norm` |
//! | its shortcut | `stages.{i}.units.{j}.shortcut.conv`, `….norm` |
//! | its channel attention | `stages.{i}.units.{j}.attn` (`.fc1`, `.fc2`; `.conv`) |
//! | the final conv | `final_conv.conv`, `final_conv.norm` |
//! | the classifier | `head.fc` |
//!
//! ## Seams
//!
//! Each slot is an enum config and an enum module, a kind a module of its
//! own ([`blocks`' rule](crate::blocks#design-rules)); a kind not built yet is
//! added as a variant whose lowering is
//! [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported):
//!
//! - the unit kind: [`ConvUnitKind`] (basic, bottleneck, and their
//!   pre-activation forms), each conv optionally weight-standardized
//!   ([`ConvLayersConfig::weight_std`]) and SAME-padded
//!   ([`ConvLayersConfig::same_pad`]);
//! - a path's conv: [`ResidualConvConfig`] (a conv block, a pre-activation
//!   conv);
//! - the shortcut: [`ConvShortcutKind`] (a conv and a norm; an average pool,
//!   then a conv and a norm; none, and no residual), lowering to one
//!   [`ConvShortcutConfig`], an optional pool before a conv block;
//! - the channel attention: [`ConvAttentionKind`] /
//!   [`ChannelAttentionConfig`](crate::blocks::images::attention::ChannelAttentionConfig)
//!   (squeeze-excite, ECA, `RegNet`'s squeeze-excite of the unit's input), and
//!   its [`ConvAttentionPlace`];
//! - the stem's pool: [`ConvStemPoolKind`] / [`ConvStemPoolConfig`] (max,
//!   average, max then blur, a strided conv, a zero pad then max, a SAME pad
//!   then max, none);
//! - the final layer: [`ConvFinalKind`] / [`ConvFinalConfig`] (an activation, a
//!   norm and activation, a conv block);
//! - the global pool: [`GlobalPool2dConfig`] (average, max, their mean, their
//!   concatenation).

mod model;
mod policy;
mod pool;
mod preact;
mod stage;
mod stem;
mod unit;

pub use model::*;
pub use policy::*;
pub use pool::*;
pub use preact::*;
pub use stage::*;
pub use stem::*;
pub use unit::*;

#[cfg(test)]
mod tests;
