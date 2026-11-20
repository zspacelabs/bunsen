//! # Model config: what a model is configured from
//!
//! The config side of a kit, apart from building and from reading an
//! upstream's files: presets and generation policy. Reading a checkpoint's
//! own configs is [`interop`](super::interop)'s; the catalog's probes of its
//! files are [`pretrained`](super::pretrained)'s (feature `store`).
//!
//! - [`resolve_presets`]: a config resolved through a pretrained's presets
//!   ([`ResolvedPresetSet`](crate::data::catalog::presets::ResolvedPresetSet))
//!   as a [`PresetSelection`](crate::data::catalog::presets::PresetSelection)
//!   asks: the family's config, the row's default presets, the selected
//!   presets, then a final patch, each a [merge
//!   patch](crate::support::merge_patch) over the one before.
//! - [`GenerationPolicy`]: how a chat model generates (stop tokens, sampling, a
//!   token cap, the template's kwargs), the call-time half of a chat family's
//!   config, carried for a driver to read.

mod generation;
mod presets;

pub use generation::*;
pub use presets::*;
