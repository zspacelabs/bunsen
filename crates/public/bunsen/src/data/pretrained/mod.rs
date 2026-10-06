//! # Pretrained models
//!
//! From a name to a model. A kit's [`PretrainedFactory`] takes a spec,
//! `provider:ref` or a bare name, and asks its providers for the row the
//! spec names. Loading the row, the [`PretrainedCache`] brings its files
//! local and the kit's [`Construct`] hook builds from them, behind an
//! `Arc`. The [`data`](crate::data#lifecycle) overview draws the whole
//! pathway; this page says what the pieces are, and why they are split
//! the way they are.
//!
//! Most of this module needs the `cache` feature (default, through
//! `store`): the factory, the cache, [`Deferred`], [`Construct`] and
//! [`Loaded`]. The rows, maps, resources, prefabs and providers are plain
//! data and are always here.
//!
//! ## Prefabs, rows and resources
//!
//! Three things are kept apart, because they vary independently:
//!
//! - A **prefab** is a geometry by name, with no weights: what a name means as
//!   a shape before any bytes are fetched. A kit keeps its prefabs in a
//!   [`StaticPreFabMap`].
//! - A **row**, a [`Pretrained`], is a name in a provider: its aliases and
//!   listing fields, the prefab it instantiates when the kit keeps one, and the
//!   resource maps it is made of, fused strictly into one [`ResourceMap`]. A
//!   file many rows share is a map of its own that each row references
//!   (Whisper's two vocabularies serve its twelve checkpoints); a one-file
//!   model is a row of one map. One prefab may have many rows, in many
//!   providers.
//! - A **resource**, a [`Resource`], is one file: its key in the map
//!   (`checkpoint`, `vocabulary`: the kit's constants), the file name it lands
//!   as, the SHA-256 that pins it, a `kind` label, and the [`Source`]s it can
//!   be had from, in order.
//!
//! None of these knows a hook, or which resource is the model: the
//! function that loads a kit's models is the kit's, and it supplies the
//! hook. Rows, maps and resources are plain serde data, so a manifest can
//! carry them. Each has a static twin for compiled-in tables
//! ([`StaticPretrained`], [`StaticResourceMap`], [`StaticResource`],
//! [`StaticSource`]), which a `to_*` method turns into the owned one. An
//! owned resource carries its map's bases appended to its own sources, so
//! it stands alone, and two maps [fuse](ResourceMap::fuse) by key with no
//! memory of where a resource came from: strictly for authoring, where a
//! repeated key is a mistake, or as an overlay for an override, where the
//! right-hand map wins.
//!
//! ## Providers
//!
//! A [`PretrainedProvider`] is a namespace over rows: the `provider` of
//! `provider:ref`. It looks a name up, and lists what it has when it can.
//! Three are in use:
//!
//! - `well-known:` ([`WELL_KNOWN`]) holds the compiled-in rows, a
//!   [`PretrainedTable`] of [`PretrainedGroup`]s. `well-known:openai/tiny` is
//!   the `tiny` row of the `openai` group; a bare `openai/tiny`, or `tiny`,
//!   finds it too.
//! - `bundled:` ([`BUNDLED`]) holds the rows a build ships with: a table a kit
//!   registers under its `*-weights` feature.
//! - `hf:` ([`HF`]) is an [`HfProvider`]: a Hugging Face repo, by `org/repo`.
//!   It lists nothing and never answers a bare name.
//!
//! A [`PretrainedFactory`] holds a kit's providers in search order and
//! offers a spec to them by its [dispatch rules](PretrainedFactory#dispatch).
//! A provider's `Ok(None)` means "not here", and an error aborts the
//! lookup.
//!
//! ## The cache and bundles
//!
//! The [`PretrainedCache`] keeps a pinned file at
//! `<cache>/pretrained/<kit>/<namespace>/<sha256>/<file>`. The digest in
//! the path is the pin: a file there was verified when it was written, and
//! is trusted without re-hashing (the
//! [trust model](PretrainedCache#trust-model)). The cache looks there
//! before it tries any of a resource's sources (the
//! [source order](PretrainedCache#source-order)).
//!
//! A bundle takes one of two forms:
//!
//! - Bytes linked into the binary are a [`Source::Bundled`], written into the
//!   cache under their digest on first use. `bunsen-bundled-silero` ships its
//!   burnpack this way, as `bundled:silero/vad`.
//! - Files fetched at build time are a populated cache directory.
//!   `bunsen-bundled-whisper` lays its `OUT_DIR` out in this layout, so a
//!   [`PretrainedCache`] rooted there finds the files without knowing they were
//!   bundled. (Its `bundled:openai/base` row serves the same files in place, as
//!   local-dir sources.)
//!
//! The second form is the general one: bundling a fetched asset means
//! populating a cache directory, not adding a kind of source. A deployment
//! that has to run offline does the same thing at install time: it fills
//! the cache directory ahead of time (`whisper-cli models fetch`) and
//! points the cache at it.
//!
//! ## Example
//!
//! A toy kit whose model is the text of one file, a table with one row
//! over a directory another tool keeps, and an offline cache in a
//! temporary directory. Nothing here reaches the network.
//!
//! ```rust
//! # #[cfg(feature = "cache")] {
//! use std::sync::Arc;
//!
//! use bunsen::{
//!     data::{
//!         cache::{
//!             BunsenDiskCacheOptions,
//!             sha256_of,
//!         },
//!         pretrained::{
//!             Construct,
//!             Deferred,
//!             LoadedResources,
//!             Pretrained,
//!             PretrainedCache,
//!             PretrainedCacheOptions,
//!             PretrainedFactory,
//!             PretrainedGroup,
//!             PretrainedRef,
//!             PretrainedTable,
//!             Provenance,
//!             Resource,
//!             ResourceMap,
//!             Source,
//!             WELL_KNOWN,
//!         },
//!     },
//!     errors::{
//!         BunsenResult,
//!         sys_at,
//!     },
//! };
//!
//! /// The toy kit's hook: it builds the text of the `text` resource.
//! struct Greeting;
//!
//! impl Construct for Greeting {
//!     type Built = String;
//!
//!     const KIT: &'static str = "greeting";
//!
//!     fn for_map(_map: &ResourceMap) -> BunsenResult<Self> {
//!         Ok(Greeting)
//!     }
//!
//!     fn construct(
//!         &self,
//!         _model: &PretrainedRef,
//!         loaded: &LoadedResources,
//!         _device: &Device,
//!     ) -> BunsenResult<Arc<String>> {
//!         let path = loaded.expect("text")?;
//!         let text =
//!             std::fs::read_to_string(path).map_err(sys_at("read", path))?;
//!         Ok(Arc::new(text))
//!     }
//! }
//!
//! // A directory another tool keeps the file in.
//! let tmp = tempfile::tempdir()?;
//! let upstream = tmp.path().join("upstream");
//! std::fs::create_dir_all(&upstream)?;
//! std::fs::write(upstream.join("hello.txt"), "hello")?;
//!
//! // A row of one resource, pinned, with that directory as its source.
//! let hello = Pretrained {
//!     name: "hello".to_string(),
//!     aliases: vec!["hi".to_string()],
//!     description: "a greeting".to_string(),
//!     license: None,
//!     origin: None,
//!     prefab: None,
//!     resources: ResourceMap::new("hello").with_resource(Resource {
//!         key: "text".to_string(),
//!         file: "hello.txt".to_string(),
//!         sha256: Some(sha256_of(&upstream.join("hello.txt"))?),
//!         kind: Some("text".to_string()),
//!         namespace: "toy".to_string(),
//!         sources: vec![Source::LocalDir {
//!             name: "upstream".to_string(),
//!             dir: Some(upstream.clone()),
//!         }],
//!     }),
//! };
//!
//! // The kit's index: one provider, a table of one group.
//! let table = PretrainedTable {
//!     name: WELL_KNOWN.to_string(),
//!     description: "the toy kit's rows".to_string(),
//!     groups: vec![PretrainedGroup {
//!         name: "toy".to_string(),
//!         description: "toy rows".to_string(),
//!         license: None,
//!         origin: None,
//!         items: vec![hello],
//!     }],
//! };
//! let factory =
//!     PretrainedFactory::<Greeting>::new().with_provider(Arc::new(table))?;
//! assert_eq!(factory.ids(), ["well-known:toy/hello"]);
//!
//! // A cache rooted in the temporary directory, and offline.
//! let cache = PretrainedCache::new(
//!     PretrainedCacheOptions::default()
//!         .with_disk(
//!             BunsenDiskCacheOptions::default()
//!                 .with_cache_dir(Some(tmp.path().join("cache")))
//!                 .without_transfer_observers(),
//!         )
//!         .with_offline(true),
//! )?;
//!
//! // The whole pathway: the alias resolves to the row, the file is used in
//! // place, and the hook builds.
//! let loaded = factory.load("hi", &cache, &Default::default())?;
//! assert_eq!(loaded.name, "well-known:toy/hello");
//! assert_eq!(*loaded.handle, "hello");
//! let text = loaded.resources.get("text").expect("the text");
//! assert_eq!(text.provenance, Provenance::LocalDir);
//! assert!(!tmp.path().join("cache").exists(), "nothing was copied in");
//!
//! // A path is not the factory's business: it is a given map.
//! let given = Deferred::<Greeting>::from_map(ResourceMap::given(
//!     "mine",
//!     "text",
//!     upstream.join("hello.txt"),
//! ))?;
//! let loaded = given.load(&cache, &Default::default())?;
//! assert_eq!(
//!     (loaded.name.as_str(), loaded.handle.as_str()),
//!     ("mine", "hello")
//! );
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The kits' factories are the full-size versions:
//! [`default_resnet_factory`](crate::kits::images::resnet::pretrained::default_resnet_factory),
//! [`default_whisper_factory`](crate::kits::speech::whisper::pretrained::default_whisper_factory)
//! and
//! [`default_silero_factory`](crate::kits::speech::silero_vad::pretrained::default_silero_factory).

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
mod construct;
#[cfg(feature = "cache")]
mod deferred;
#[cfg(feature = "cache")]
mod factory;
mod hf;
#[cfg(feature = "cache")]
mod loaded;
mod prefabs;
mod pretrained_ref;
mod providers;
mod resource;
mod resource_map;
mod rows;
#[cfg(feature = "store_safetensors")]
mod safetensors;

#[cfg(feature = "cache")]
pub use cache::*;
#[cfg(feature = "cache")]
pub use construct::*;
#[cfg(feature = "cache")]
pub use deferred::*;
#[cfg(feature = "cache")]
pub use factory::*;
pub use hf::*;
#[cfg(feature = "cache")]
pub use loaded::*;
pub use prefabs::*;
pub use pretrained_ref::*;
pub use providers::*;
pub use resource::*;
pub use resource_map::*;
pub use rows::*;
#[cfg(feature = "store_safetensors")]
pub use safetensors::*;

/// A [`Lookup`](crate::errors::BunsenErrorKind::Lookup) error for a name a
/// table does not have: a [`LookupError`](crate::errors::LookupError) for
/// `name` among the `kind`s, with the names the table has as candidates, under
/// a frame naming the table when there is one: `"<table>: no <kind> "x";
/// there are: a, b"`.
#[track_caller]
pub(crate) fn not_found(
    table: Option<&str>,
    kind: &'static str,
    name: &str,
    names: &[&str],
) -> crate::errors::BunsenError {
    let error = crate::errors::BunsenError::lookup(
        crate::errors::LookupError::missing(kind, name).with_candidates(names.iter().copied()),
    );
    match table {
        Some(table) => error.context(table),
        None => error,
    }
}
