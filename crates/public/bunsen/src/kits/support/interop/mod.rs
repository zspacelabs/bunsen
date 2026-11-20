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
//! - [`nemo`] (feature `speech`): `NeMo`'s YAML config sections, the bunsen
//!   configs they give, and its state-dict names.
//!
//! ## The ladder
//!
//! A kit's interop module offers, for each config it reads, entry points
//! with predictable names: `{target}_config` (a conversion, from the parsed
//! form), `{target}_config_from_str` and `{target}_config_from_path` (for a
//! config a caller starts from, through this tier's helpers),
//! `load_{target}_tensors` (a rekey and a load), and `load_{target}` (a
//! loaded component).
//!
//! A conversion lives on the upstream side, in an interop module. When it
//! reads one mirror (a typed record of an upstream file's fields, named
//! `*Mirror`), it is a method on that mirror, named for its target
//! ([`NemoPreprocessorMirror::mel_front_end_config`](nemo::NemoPreprocessorMirror::mel_front_end_config));
//! otherwise (several sources, or a parsed document such as
//! [`TransformersConfigJson`](transformers::TransformersConfigJson)) a free
//! function, named the same way. It is never a method on the bunsen config,
//! nor a `From` / `TryFrom` impl.

#[cfg(feature = "speech")]
pub mod nemo;
pub mod transformers;
