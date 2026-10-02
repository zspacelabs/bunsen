//! # Pretrained Whisper models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The tables and maps are always here; the factory and the hook need the
//! `store_pytorch` and `cache` features.
//!
//! # The index
//!
//! [`default_whisper_factory`] is the index a caller holds, over
//! [`default_whisper_providers`] in search order:
//!
//! - [`WELL_KNOWN_TABLE`]: the checkpoints bunsen knows by name, as
//!   `well-known:openai/base`, `openai/base`, or `base`. Its one group,
//!   [`OPENAI`], is upstream's `_MODELS`, with upstream's aliases (`large`,
//!   `turbo`).
//! - With the `whisper-weights` feature, the bundled table:
//!   `bundled:openai/base`, served in place from `bunsen-bundled-whisper`'s
//!   build directory.
//! - The data layer's [`HfProvider`](crate::data::pretrained::HfProvider):
//!   `hf:openai/whisper-large-v3` for a Hugging Face repo, its safetensors
//!   checkpoint under [`CHECKPOINT`]. It lists nothing.
//!
//! A caller with a provider of its own, a mirror say, adds it to the
//! default factory with
//! [`with_provider`](crate::data::pretrained::PretrainedFactory::with_provider);
//! the hook stays the kit's, chosen by the row, never the caller's.
//!
//! # Rows, prefabs and maps
//!
//! A row names its prefab in [`WHISPER_PREFABS`], the geometry its
//! checkpoint must have, and fuses a checkpoint map from
//! [`OPENAI_CHECKPOINTS`] with the vocabulary map its token layout
//! selects ([`WhisperVocabulary`]). A prefab is a shape without weights,
//! so a name means a shape before any bytes are fetched; a map is where
//! the bytes come from and the digest that pins them. A path arrives as a
//! one-resource map with no prefab, and gets its vocabulary from the same
//! rule, [`WhisperVocabulary::for_layout`].
//!
//! # Loading
//!
//! [`load_bundle`](crate::data::pretrained::PretrainedFactory::load_bundle)
//! is the whole pathway from a name to a
//! [`WhisperBundle`](crate::kits::speech::whisper::driver::WhisperBundle)
//! behind an `Arc`. The factory resolves the name to a
//! [`Deferred`](crate::data::pretrained::Deferred) model carrying the
//! kit's hook, [`WhisperConstruct`], with the [`WhisperReader`] its
//! checkpoint's `kind` names (`OpenAI`'s `.pt`, or `transformers`'
//! `model.safetensors`, one file or shards). The hook's plan brings the
//! checkpoint local, scans it, checks it against the geometry the prefab
//! promised, and settles the vocabulary; the cache brings the rest local;
//! and the hook's construct reads the checkpoint into a model at the
//! precision it ships in and the vocabulary into ranks. A checkpoint on
//! disk is not the factory's, but becomes a deferred model the same way,
//! through [`Deferred::from_map`](crate::data::pretrained::Deferred::from_map).
//!
//! # Example
//!
//! The index answers without loading anything:
//!
//! ```rust
//! # #[cfg(all(feature = "store_pytorch", feature = "cache"))] {
//! use bunsen::kits::speech::whisper::{
//!     driver::WhisperSpecialIds,
//!     pretrained::{
//!         VOCABULARY,
//!         WHISPER_PREFABS,
//!         WhisperVocabulary,
//!         default_whisper_factory,
//!     },
//! };
//!
//! let factory = default_whisper_factory()?;
//!
//! // `large` is upstream's alias for `large-v3`, and the row names its
//! // prefab: the geometry the checkpoint must scan to.
//! let (_, row) = factory.lookup("large")?;
//! assert_eq!(row.name, "openai/large-v3");
//! let prefab =
//!     WHISPER_PREFABS.try_lookup_prefab(row.prefab.as_deref().unwrap())?;
//! let geometry = prefab.to_config().geometry();
//! assert_eq!((geometry.n_mels, geometry.vocab_size), (128, 51866));
//!
//! // The row's vocabulary is the one that geometry's token layout selects.
//! let ids = WhisperSpecialIds::from_vocab_size(geometry.vocab_size)?;
//! let rule = WhisperVocabulary::for_layout(&ids).map().to_map();
//! assert_eq!(
//!     row.resources.try_get(VOCABULARY)?.file,
//!     rule.try_get(VOCABULARY)?.file,
//! );
//! # }
//! # Ok::<(), bunsen::errors::BunsenError>(())
//! ```

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
pub use construct::*;
#[cfg(all(feature = "store_pytorch", feature = "cache"))]
pub use factory::*;
pub use maps::*;
pub use prefabs::*;
pub use providers::*;
#[cfg(feature = "cache")]
pub use vocab::*;

#[cfg(feature = "store_pytorch")]
mod pytorch_utils;

#[cfg(feature = "store_pytorch")]
pub use pytorch_utils::*;

#[cfg(feature = "store_safetensors")]
mod safetensors_utils;

#[cfg(feature = "store_safetensors")]
pub use safetensors_utils::*;
