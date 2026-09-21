//! # Pretrained Whisper models
//!
//! From a model's name to a loaded model. [`default_whisper_factory`] is
//! the index: `well-known:openai/base`, `openai/base`, `large`, or a path
//! to a checkpoint. `factory.resolve_for::<WhisperConstruct>(spec)` is a
//! [`PretrainedRef`](crate::data::pretrained::PretrainedRef), and
//! `factory.load::<B, WhisperConstruct>(spec, &cache, &hook, device)` takes
//! it the rest of the way: every resource of its map into the cache, the
//! checkpoint through the scanner and past a check that it has the
//! geometry its prefab promised, the vocabulary through the rank parser,
//! and out as a
//! [`WhisperBundle`](crate::kits::speech::whisper::driver::WhisperBundle)
//! behind an `Arc`. A caller with its own providers builds its own factory
//! over [`default_whisper_providers`].

#[cfg(all(feature = "store_pytorch", feature = "cache"))]
mod construct;
mod maps;
mod prefabs;
mod providers;
#[cfg(feature = "cache")]
mod vocab;

#[cfg(all(feature = "store_pytorch", feature = "cache"))]
#[doc(inline)]
pub use construct::*;
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
