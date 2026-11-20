//! # Interop: what an external ecosystem's files and names mean, for any kit
//!
//! The shared tier of the kits' interop: what more than one kit reads the
//! same way from an ecosystem's files. One module per ecosystem, named for
//! the schema's owner, never the organization. A kit's own mapping of its
//! model onto an ecosystem is the kit's `interop::{ecosystem}`, which builds
//! on this one.
//!
//! - [`transformers`]: a `transformers` `config.json`, read field by field
//!   ([`TransformersConfigJson`](transformers::TransformersConfigJson)), its
//!   decoder fields
//!   ([`TransformersDecoderSection`](transformers::TransformersDecoderSection)),
//!   and the helpers a kit's `_from_str` / `_from_path` entry points wrap.
//!
//! ## The ladder
//!
//! A kit's interop module offers, for each config it reads, entry points
//! with predictable names: `{target}_config` (a conversion, from the parsed
//! form), `{target}_config_from_str` and `{target}_config_from_path` (for a
//! config a caller starts from, through this tier's helpers),
//! `load_{target}_tensors` (a rekey and a load), and `load_{target}` (a
//! loaded component). A conversion is a free function, never a
//! `From` / `TryFrom` impl or a method on the bunsen config.

pub mod transformers;
