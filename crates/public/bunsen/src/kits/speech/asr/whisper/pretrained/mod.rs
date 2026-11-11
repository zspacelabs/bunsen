//! # Pretrained Whisper models
//!
//! From a model's name to a loaded model, through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)).
//! The prefab map is always here: a geometry needs no features. The tables,
//! the maps, the hook and the factory need the `store` and `cache`
//! features.
//!
//! # The index
//!
//! [`default_whisper_factory`] is the index a caller holds, over
//! [`default_whisper_providers`] in search order:
//!
//! - [`WHISPER_CATALOG`]: the checkpoints bunsen knows by name, as
//!   `well-known:openai/base`, `openai/base`, or `base`. Its one group,
//!   `openai`, is upstream's `_MODELS`, with upstream's aliases (`large`,
//!   `turbo`).
//! - The data layer's [`HfProvider`](crate::data::pretrained::HfProvider):
//!   `hf:openai/whisper-large-v3` for a Hugging Face repo, its safetensors
//!   checkpoint under [`CHECKPOINT`]. It lists nothing.
//!
//! # Presets
//!
//! A construct carries the [`WhisperPolicy`](super::driver::WhisperPolicy)
//! its presets resolved to, which a driver takes with
//! [`with_policy`](super::driver::WhisperStreamDriverConfig::with_policy).
//! The multilingual checkpoints that translate (every one but the `.en`
//! checkpoints and `large-v3-turbo`) are the catalog's `multilingual` template
//! class, with a `task` axis: `[transcribe]`, the default, and `[translate]`.
//! Every other field is the lookup's patch:
//! `'openai/small[translate]{beam_size: 1}'`. A multilingual checkpoint no row
//! describes, a file or an `hf:` repo, takes the class's presets
//! ([`MULTILINGUAL_TEMPLATE_CLASS`]).
//!
//! A caller with a provider of its own, a mirror say, adds it to the
//! default factory with
//! [`with_provider`](crate::kits::support::construct::PretrainedFactory::with_provider);
//! the hook stays the kit's, chosen by the row, never the caller's.
//!
//! # Rows, prefabs and bundles
//!
//! A row names its prefab in [`WHISPER_PREFABS`], the geometry its
//! checkpoint must have, and fuses a checkpoint bundle of
//! [`WHISPER_CATALOG`] with the vocabulary bundle its token layout
//! selects ([`WhisperVocabulary`]). A prefab is a shape without weights,
//! so a name means a shape before any bytes are fetched; a bundle is where
//! the bytes come from and the digest that pins them. A path arrives as a
//! one-resource bundle with no prefab, and gets its vocabulary from the same
//! rule, [`WhisperVocabulary::for_layout`].
//!
//! # Loading
//!
//! [`load_construct`](crate::kits::support::construct::PretrainedFactory::load_construct)
//! is the whole pathway from a name to a
//! [`WhisperConstruct`](crate::kits::speech::asr::whisper::driver::WhisperConstruct)
//! behind an `Arc`. The factory resolves the name to a
//! [`Deferred`](crate::kits::support::construct::Deferred) model carrying the
//! kit's hook, [`WhisperConstructConfig`], with the [`WhisperReader`] its
//! checkpoint's `kind` names (`OpenAI`'s `.pt`, or `transformers`'
//! `model.safetensors`, one file or shards). The hook's plan brings the
//! checkpoint local, scans it, checks it against the geometry the prefab
//! promised, and settles the vocabulary; the cache brings the rest local;
//! and the hook's construct reads the checkpoint into a model at the
//! precision it ships in and the vocabulary into ranks. A checkpoint on
//! disk is not the factory's, but becomes a deferred model the same way,
//! through [`Deferred::from_resource_bundle`](crate::kits::support::construct::Deferred::from_resource_bundle).
//!
//! # Example
//!
//! The index answers without loading anything:
//!
//! ```rust
//! # #[cfg(feature = "store")] {
//! use bunsen::kits::speech::asr::whisper::{
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
//! // `large` is upstream's alias for `large-v3`; its prefab documents the
//! // geometry its checkpoint carries.
//! let (_, row) = factory.lookup("large")?;
//! assert_eq!(row.name, "openai/whisper/large/large-v3");
//! let prefab = WHISPER_PREFABS.try_lookup_prefab("large-v3")?;
//! let geometry = prefab.to_config().geometry();
//! assert_eq!((geometry.n_mels, geometry.vocab_size), (128, 51866));
//!
//! // The row's vocabulary is the one that geometry's token layout selects.
//! let ids = WhisperSpecialIds::from_vocab_size(geometry.vocab_size)?;
//! let rule = WhisperVocabulary::for_layout(&ids).resource_bundle();
//! assert_eq!(
//!     row.resources.try_get(VOCABULARY)?.file,
//!     rule.try_get(VOCABULARY)?.file,
//! );
//! # }
//! # Ok::<(), bunsen::errors::BunsenError>(())
//! ```

#[cfg(feature = "store")]
mod construct;
#[cfg(feature = "store")]
mod factory;
mod prefabs;
#[cfg(feature = "store")]
mod providers;
#[cfg(feature = "store")]
mod resource_bundles;
#[cfg(feature = "store")]
mod vocab;

#[cfg(feature = "store")]
pub use construct::*;
#[cfg(feature = "store")]
pub use factory::*;
pub use prefabs::*;
#[cfg(feature = "store")]
pub use providers::*;
#[cfg(feature = "store")]
pub use resource_bundles::*;
#[cfg(feature = "store")]
pub use vocab::*;

#[cfg(feature = "store")]
mod pytorch_utils;

#[cfg(feature = "store")]
pub use pytorch_utils::*;

#[cfg(feature = "store")]
mod safetensors_utils;

#[cfg(feature = "store")]
pub use safetensors_utils::*;
