//! # Pretrained Whisper models
//!
//! From a model's name to a loaded model. `default_whisper_factory()` is
//! the index: `well-known:openai/base`, `openai/base`, `large`.
//! `factory.load_bundle::<B>(name, &cache, device)` is the whole pathway
//! through the kit's hook: every resource of the row's map into the cache,
//! the checkpoint through the reader its `kind` names and past a check
//! that it has the geometry its prefab promised, the vocabulary through
//! the rank parser, and out as a
//! [`WhisperBundle`](crate::kits::speech::whisper::driver::WhisperBundle)
//! behind an `Arc`. A checkpoint on disk is not the factory's: it is a
//! given map, read through the same hook with
//! [`PretrainedRef::load`](crate::data::pretrained::PretrainedRef::load).
//! A caller with a provider of its own adds it to the default factory; the
//! hook is the kit's, never the caller's.

#[cfg(all(feature = "store_pytorch", feature = "cache"))]
mod construct;
#[cfg(all(feature = "store_pytorch", feature = "cache"))]
mod factory;
mod maps;
mod prefabs;
mod providers;
#[cfg(feature = "cache")]
mod vocab;

#[cfg(all(feature = "store_pytorch", feature = "cache"))]
#[doc(inline)]
pub use construct::*;
#[cfg(all(feature = "store_pytorch", feature = "cache"))]
#[doc(inline)]
pub use factory::*;
#[doc(inline)]
pub use maps::*;
#[doc(inline)]
pub use prefabs::*;
#[doc(inline)]
pub use providers::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use vocab::*;

#[cfg(feature = "store_pytorch")]
mod pytorch_utils;

#[cfg(feature = "store_pytorch")]
#[doc(inline)]
pub use pytorch_utils::*;
