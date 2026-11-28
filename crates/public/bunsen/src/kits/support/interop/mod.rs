//! # Interop: the ecosystem-neutral helpers of the kits' interop
//!
//! What bunsen knows of an external ecosystem lives in that ecosystem's
//! module, by use: the shared tiers are ecosystem first
//! ([`kits::support::transformers::interop`](crate::kits::support::transformers::interop),
//! [`kits::support::torch::interop`](crate::kits::support::torch::interop),
//! [`kits::support::mistral_common::interop`](crate::kits::support::mistral_common::interop)
//! for any kit; `kits::images::support::timm::interop` and
//! `kits::speech::support::nemo::interop` for one area's), and a kit's own
//! mapping of its model onto an ecosystem is the kit's `interop::{ecosystem}`,
//! which builds on them. This module holds what no ecosystem owns:
//!
//! - [`json_mirror_from_path`]: a JSON file a kit reads whole, as its typed
//!   mirror.
//! - `load_lm_rekeyed` (feature `store`): a checkpoint's tensors into a
//!   language model holder's
//!   [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel),
//!   under the kit's rekey.
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
//! ([`NemoPreprocessorMirror::mel_front_end_config`](crate::kits::speech::support::nemo::interop::NemoPreprocessorMirror::mel_front_end_config));
//! otherwise (several sources, or a parsed document such as
//! [`TransformersConfigJson`](crate::kits::support::transformers::interop::TransformersConfigJson)) a free
//! function, named the same way. It is never a method on the bunsen config,
//! nor a `From` / `TryFrom` impl.

#[cfg(feature = "store")]
mod checkpoint;
#[cfg(feature = "store")]
mod weights;

#[cfg(feature = "store")]
pub use checkpoint::*;
#[cfg(feature = "store")]
pub use weights::*;

mod mirror;

pub use mirror::*;
