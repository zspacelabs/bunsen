//! # Pretrained models
//!
//! From a name to a model. A kit's
//! [`PretrainedFactory`](crate::kits::support::construct::PretrainedFactory)
//! takes a spec, `provider:ref` or a bare name, and asks its providers for the
//! row the spec names. Loading the row, the [`PretrainedCache`] brings its
//! files local and the kit's
//! [`ConstructInit`](crate::kits::support::construct::ConstructInit) hook
//! builds from them, behind an `Arc`. The [`data`](crate::data#lifecycle)
//! overview draws the whole pathway; this page says what the pieces are, and
//! why they are split the way they are.
//!
//! Most of this module needs the `cache` feature (default, through
//! `store`): the factory, the cache,
//! [`Deferred`](crate::kits::support::construct::Deferred),
//! [`ConstructInit`](crate::kits::support::construct::ConstructInit) and
//! [`Loaded`](crate::kits::support::construct::Loaded). The rows, resource
//! bundles, resources, prefabs and providers are plain data and are always
//! here.
//!
//! A hook reads its checkpoint's own format with the readers here: the
//! safetensors checkpoint and header ([`SafetensorsCheckpoint`]), MLX's
//! affine-quantized tensors ([`mlx`]), `FermionResearch`'s Phonon formats
//! ([`fermion`]), and [`apply_named_tensors`], which loads any of them into a
//! module under a kit's name resource bundle, or [`apply_rekeyed_tensors`],
//! under a [`Rekey`](crate::burner::module::Rekey) whose targets may also
//! change a tensor's data on the way.
//!
//! ## Prefabs, rows and resources
//!
//! Three things are kept apart, because they vary independently:
//!
//! - A **prefab** is a geometry by name, with no weights: what a name means as
//!   a shape before any bytes are fetched. A kit keeps its prefabs in a
//!   [`StaticPreFabMap`].
//! - A **row**, a [`Pretrained`], is a name in a provider: its aliases and
//!   listing fields, its [`tags`] (facts about the model, for listings and for
//!   choosing among models, [`selection`]), the prefab it instantiates when the
//!   kit keeps one, and the resource bundles it is made of, fused strictly into
//!   one [`ResourceBundle`]. A file many rows share is a resource bundle of its
//!   own that each row references (Whisper's two vocabularies serve its twelve
//!   checkpoints); a one-file model is a row of one resource bundle. One prefab
//!   may have many rows, in many providers.
//! - A **resource**, a [`Resource`], is one file: its key in the resource
//!   bundle (`checkpoint`, `vocabulary`: the kit's constants), the file name it
//!   lands as, the SHA-256 that pins it, a `kind` label, and the [`Source`]s it
//!   can be had from, in order.
//!
//! None of these knows a hook, or which resource is the model: the
//! function that loads a kit's models is the kit's, and it supplies the
//! hook. Rows, resource bundles and resources are plain serde data, so a
//! manifest can carry them. An owned resource carries its resource bundle's
//! bases appended to its own sources, so it stands alone, and two resource
//! bundles [fuse](ResourceBundle::fuse) by key with no memory of where a
//! resource came from: strictly for authoring, where a repeated key is a
//! mistake, or as an overlay for an override, where the right-hand resource
//! bundle wins.
//!
//! A kit's table is data: a [`PretrainedCatalog`], the `catalog.json` the
//! catalog tool (`cargo validate catalog`) generates from a hand-written
//! source, embedded in the kit as an [`EmbeddedCatalog`]. Its rows name their
//! resource bundles, a bundle shared by many rows is listed once, its files
//! carry their sizes and dtypes, and a directory base names where another
//! tool keeps its files as path templates ([`resolve_dir_template`]).
//! [`PretrainedCatalog::to_table`] loads it into the owned
//! [`PretrainedTable`].
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
//! - `bundled:` ([`BUNDLED`]) holds the rows a build ships with: a table a
//!   caller registers over files it lays out or bytes it links in.
//! - `hf:` ([`HF`]) is an [`HfProvider`]: a Hugging Face repo, by `org/repo`.
//!   It lists nothing and never answers a bare name.
//!
//! A [`PretrainedFactory`](crate::kits::support::construct::PretrainedFactory)
//! holds a kit's providers in search order and offers a spec to them by its
//! [dispatch rules]. A provider's `Ok(None)` means "not here", and an error
//! aborts the lookup.
//!
//! [dispatch rules]: crate::kits::support::construct::PretrainedFactory#dispatch
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
//! [`cached_files`](PretrainedCache::cached_files) lists what a cache holds,
//! a [`CachedFile`] each, by kit and namespace ([`CacheSelector`]), and
//! [`remove_cached`](PretrainedCache::remove_cached) clears them.
//!
//! A bundle takes one of two forms:
//!
//! - Bytes linked into the binary are a [`Source::Bundled`], written into the
//!   cache under their digest on first use.
//! - Files fetched at build time are a populated cache directory.
//!   `bind-whisper-onnx` lays its `OUT_DIR` out in this layout, so a
//!   [`PretrainedCache`] rooted there finds the files without knowing they were
//!   bundled. (`whisper-model-validation` also registers a
//!   `bundled:openai/base` row over the same files, served in place as
//!   local-dir sources.)
//!
//! The second form is the general one: bundling a fetched asset means
//! populating a cache directory, not adding a kind of source. A deployment
//! that has to run offline does the same thing at install time: it fills
//! the cache directory ahead of time (`bunsen-tool asr family <family> fetch`)
//! and points the cache at it.
//!
//! ## Example
//!
//! A toy kit whose model is the text of one file, a table with one row
//! over a directory another tool keeps, and an offline cache in a
//! temporary directory. Nothing here reaches the network.
//!
//! ```rust
//! # #[cfg(all(feature = "cache", feature = "kits"))] {
//! use std::sync::Arc;
//!
//! use bunsen::{
//!     data::{
//!         cache::{
//!             BunsenDiskCacheOptions,
//!             sha256_of,
//!         },
//!         pretrained::{
//!             LoadedResources,
//!             Pretrained,
//!             PretrainedCache,
//!             PretrainedCacheOptions,
//!             PretrainedGroup,
//!             PretrainedRef,
//!             PretrainedTable,
//!             Provenance,
//!             Resource,
//!             ResourceBundle,
//!             Source,
//!             WELL_KNOWN,
//!         },
//!     },
//!     errors::{
//!         BunsenResult,
//!         sys_at,
//!     },
//!     kits::support::construct::{
//!         ConstructInit,
//!         Deferred,
//!         PretrainedFactory,
//!     },
//! };
//! use burn::tensor::Device;
//!
//! /// The toy kit's hook: it builds the text of the `text` resource.
//! struct Greeting;
//!
//! impl ConstructInit for Greeting {
//!     type Construct = String;
//!
//!     const KIT: &'static str = "greeting";
//!
//!     fn for_resource_bundle(_map: &ResourceBundle) -> BunsenResult<Self> {
//!         Ok(Greeting)
//!     }
//!
//!     fn try_init(
//!         &self,
//!         _model: &PretrainedRef,
//!         loaded: &LoadedResources,
//!         _device: &Device,
//!         _policy: bunsen::kits::support::construct::LoadPolicy,
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
//!     resources: ResourceBundle::new("hello").with_resource(Resource {
//!         key: "text".to_string(),
//!         file: "hello.txt".to_string(),
//!         sha256: Some(sha256_of(&upstream.join("hello.txt"))?),
//!         kind: Some("text".to_string()),
//!         size: None,
//!         dtype: None,
//!         namespace: "toy".to_string(),
//!         sources: vec![Source::LocalDir {
//!             name: "upstream".to_string(),
//!             dir: Some(upstream.clone()),
//!         }],
//!     }),
//!     presets: Default::default(),
//!     tags: Default::default(),
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
//!     case_sensitive: false,
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
//! let loaded = factory.load(
//!     "hi",
//!     &cache,
//!     &Default::default(),
//!     bunsen::kits::support::construct::LoadPolicy::as_stored(),
//! )?;
//! assert_eq!(loaded.name, "well-known:toy/hello");
//! assert_eq!(*loaded.handle, "hello");
//! let text = loaded.resources.get("text").expect("the text");
//! assert_eq!(text.provenance, Provenance::LocalDir);
//! assert!(!tmp.path().join("cache").exists(), "nothing was copied in");
//!
//! // A path is not the factory's business: it is a given resource bundle.
//! let given = Deferred::<Greeting>::from_resource_bundle(
//!     ResourceBundle::given("mine", "text", upstream.join("hello.txt")),
//! )?;
//! let loaded = given.load(
//!     &cache,
//!     &Default::default(),
//!     bunsen::kits::support::construct::LoadPolicy::as_stored(),
//! )?;
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
//! [`default_whisper_factory`](crate::kits::speech::asr::whisper::pretrained::default_whisper_factory)
//! and
//! [`default_silero_factory`](crate::kits::speech::vad::silero_vad::pretrained::default_silero_factory).

#[cfg(feature = "store")]
mod apply_named;
pub mod build_catalog;
#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
mod cached_files;
mod catalog;
#[cfg(feature = "store")]
pub mod fermion;
mod hf;
#[cfg(feature = "cache")]
mod loaded;
pub mod mlx;
mod prefabs;
mod presets;
mod pretrained_ref;
mod providers;
#[cfg(feature = "store")]
mod pytorch;
mod resource;
mod resource_bundle;
mod rows;
#[cfg(feature = "store")]
mod safetensors;
pub mod selection;
pub mod tags;

#[cfg(feature = "store")]
pub use apply_named::*;
pub use build_catalog::presets::ResolvedPresetSet;
#[cfg(feature = "cache")]
pub use cache::*;
#[cfg(feature = "cache")]
pub use cached_files::*;
pub use catalog::*;
pub use hf::*;
#[cfg(feature = "cache")]
pub use loaded::*;
pub use prefabs::*;
pub use presets::*;
pub use pretrained_ref::*;
pub use providers::*;
#[cfg(feature = "store")]
pub use pytorch::*;
pub use resource::*;
pub use resource_bundle::*;
pub use rows::*;
#[cfg(feature = "store")]
pub use safetensors::*;
pub use tags::Tags;

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
