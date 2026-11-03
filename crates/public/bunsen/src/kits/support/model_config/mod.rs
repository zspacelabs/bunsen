//! # Model config: what a model is configured from
//!
//! The config side of a kit, apart from building: reading the configs a
//! checkpoint ships with into a kit's own. Building a model from its config
//! is `construct`'s (feature `cache`).
//!
//! - [`HfConfigJson`]: a `transformers` `config.json`, or one section of it,
//!   read field by field, each error naming the field's dotted path. The kits'
//!   `hf:` readers map it onto their configs.
//! - [`resolve_presets`]: a config resolved through a pretrained's presets
//!   ([`PresetSet`](crate::data::pretrained::PresetSet)) as a
//!   [`PresetSelection`](crate::data::pretrained::PresetSelection) asks: the
//!   family's config, the row's default presets, the selected presets, then a
//!   final patch, each a [merge patch](crate::support::merge_patch) over the
//!   one before.
//! - [`GenerationPolicy`]: how a chat model generates (stop tokens, sampling, a
//!   token cap, the template's kwargs), the call-time half of a chat family's
//!   config, carried for a driver to read.
//! - [`Probe`]s: readers of a pretrained's files that find its presets' values
//!   ([`GenerationConfigProbe`], [`TokenizerConfigProbe`]), and
//!   [`apply_probes`], which lays a catalog row's own presets over what its
//!   probes find. The catalog tool runs them when it builds a catalog.

mod generation;
mod hf_config;
mod presets;
mod probes;

pub use generation::*;
pub use hf_config::*;
pub use presets::*;
pub use probes::*;
