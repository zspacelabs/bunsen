//! # Pretrained Whisper models
//!
//! From a model's name to a loaded model. `default_whisper_factory()` is
//! the index: `well-known:openai/base`, `openai/base`, `large`, or a path
//! to a checkpoint. `factory.resolve(spec)` is a
//! [`PretrainedRef`](crate::data::pretrained::PretrainedRef), and
//! `factory.load_bundle::<B>(spec, &cache, device)` takes it the rest of
//! the way through the factory's hook: every resource of its map into the
//! cache, the checkpoint through the scanner and past a check that it has
//! the geometry its prefab promised, the vocabulary through the rank
//! parser, and out as a
//! [`WhisperBundle`](crate::kits::speech::whisper::driver::WhisperBundle)
//! behind an `Arc`. A caller with a provider of its own adds it to the
//! default factory; the hook is the kit's, never the caller's.

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
