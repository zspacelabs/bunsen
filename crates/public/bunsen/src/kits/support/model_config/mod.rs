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
//!   [`PresetSelection`] asks: the family's config, the row's default presets,
//!   the selected presets, then a final patch, each a [merge
//!   patch](crate::support::merge_patch) over the one before.

mod hf_config;
mod presets;

pub use hf_config::*;
pub use presets::*;
