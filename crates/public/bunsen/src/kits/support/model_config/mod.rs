//! # Model config: what a model is configured from
//!
//! The config side of a kit, apart from building: reading the configs a
//! checkpoint ships with into a kit's own. Building a model from its config
//! is `construct`'s (feature `cache`).
//!
//! - [`TransformersConfigJson`]: a `transformers` `config.json`, or one section
//!   of it, read field by field, each error naming the field's dotted path. The
//!   kits' `hf:` readers map it onto their configs.
//! - [`resolve_presets`]: a config resolved through a pretrained's presets
//!   ([`ResolvedPresetSet`](crate::data::catalog::presets::ResolvedPresetSet))
//!   as a [`PresetSelection`](crate::data::catalog::presets::PresetSelection)
//!   asks: the family's config, the row's default presets, the selected
//!   presets, then a final patch, each a [merge
//!   patch](crate::support::merge_patch) over the one before.
//! - [`GenerationPolicy`]: how a chat model generates (stop tokens, sampling, a
//!   token cap, the template's kwargs), the call-time half of a chat family's
//!   config, carried for a driver to read.
//! - [`Probe`]s: readers of a pretrained's files that find its presets' values
//!   (`generation_config`, `tokenizer_config`, `chat_template`:
//!   [`builtin_probes`]). What they find accumulates ([`accumulate_probed`]).
//!   The catalog builder (`cargo catalog`) runs them when it builds the
//!   built-in catalog, and [`probe_loaded`] runs them over a checkpoint no
//!   catalog build describes.

mod generation;
mod presets;
mod probes;
mod transformers_config;
mod transformers_decoder_section;

pub use generation::*;
pub use presets::*;
pub use probes::*;
pub use transformers_config::*;
pub use transformers_decoder_section::*;
