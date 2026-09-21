//! # Pretrained Whisper models
//!
//! From a model's name to a loaded model. `default_whisper_factory()` is
//! the index: `well-known:openai/base`, `openai/base`, `large`, and
//! `hf:openai/whisper-large-v3` for a Hugging Face repo, through the
//! generic [`HfProvider`](crate::data::pretrained::HfProvider).
//! `factory.load_bundle::<B>(name, &cache, device)` is the whole pathway
//! through the kit's hook: every resource of the row's map into the cache,
//! the checkpoint through the reader its `kind` names (`OpenAI`'s `.pt`,
//! or `transformers`' `model.safetensors`, one file or shards) and past a check
//! that it has the geometry its prefab promised, the vocabulary through
//! the rank parser, and out as a
//! [`WhisperBundle`](crate::kits::speech::whisper::driver::WhisperBundle)
//! behind an `Arc`. What the factory resolves is a
//! [`Deferred`](crate::data::pretrained::Deferred) model carrying the hook
//! its map calls for; a checkpoint on disk is not the factory's, but
//! becomes one the same way through `Deferred::from_map`. A caller with a
//! provider of its own adds it to the default factory; the hook is the
//! kit's, chosen by the row, never the caller's.

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

#[cfg(feature = "store_safetensors")]
mod safetensors_utils;

#[cfg(feature = "store_safetensors")]
#[doc(inline)]
pub use safetensors_utils::*;
