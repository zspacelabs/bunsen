//! # Builds: from a name to a model
//!
//! From a name to a model. A kit's
//! [`BuildFactory`]
//! takes a spec, `provider:ref` or a bare name, and asks its providers for the
//! row the spec names. Loading the row, the [`ResourceCache`] brings its
//! files local and the kit's
//! [`ConstructInit`] hook
//! builds from them, behind an `Arc`. The [`data`](crate::data#lifecycle)
//! overview draws the whole pathway; this page says what the pieces are, and
//! why they are split the way they are.
//!
//! The pieces:
//!
//! - [`BuildProvider`]: a namespace over rows, the `provider` of
//!   `provider:ref`.
//! - [`ConstructInit`]: the kit's hook, which completes a resolved ref's
//!   resource set before it is loaded and builds from the loaded parts.
//! - [`Deferred`]: a model not yet loaded, its ref paired with the hook that
//!   will build it.
//! - [`BuildFactory`]: a kit's providers in search order, resolving lookup
//!   expressions to [`Deferred`] models.
//! - [`LookupExpr`]: a lookup expression, `[provider:]name[preset, …]{patch}`,
//!   split into the name and what it asks of the row's presets.
//! - [`Loaded`]: the handle the pathway hands back.
//!
//! The factory, [`Deferred`], [`ConstructInit`], [`Loaded`] and
//! [`LookupExpr`] need the `cache` feature (default, through `store`), and
//! the factory the `kits` feature too, for the kits' preset resolution. The
//! rows, refs and providers are plain data and are always here.
//!
//! A hook reads its checkpoint's own format with the readers in
//! [`formats`](crate::data::formats).
//!
//! ## Prefabs, rows and resources
//!
//! Three things are kept apart, because they vary independently:
//!
//! - A **prefab** is a geometry by name, with no weights: what a name means as
//!   a shape before any bytes are fetched. A kit keeps its prefabs in a
//!   [`StaticPreFabMap`](crate::data::pretrained::StaticPreFabMap).
//! - A **row**, a [`Pretrained`], is a name in a provider: its aliases and
//!   listing fields, its [`tags`](crate::data::pretrained::tags) (facts about
//!   the model, for listings and for choosing among models,
//!   [`catalog::select`](crate::data::catalog::select)), the prefab it
//!   instantiates when the kit keeps one, and the resource sets it is made of,
//!   fused strictly into one [`ResourceSet`]. A file many rows share is a
//!   resource set of its own that each row references (Whisper's two
//!   vocabularies serve its twelve checkpoints); a one-file model is a row of
//!   one resource set. One prefab may have many rows, in many providers.
//! - A **resource**, a [`Resource`], is one file: the slot it fills in its set
//!   (`checkpoint`, `vocabulary`: the kit's constants), the file name it lands
//!   as, the SHA-256 that pins it, its format, and the [`Source`]s it can be
//!   had from, in order.
//!
//! None of these knows a hook, or which resource is the model: the
//! function that loads a kit's models is the kit's, and it supplies the
//! hook. Rows, resource sets and resources are plain serde data, so a
//! manifest can carry them. An owned resource carries its resource set's
//! bases appended to its own sources, so it stands alone, and two resource
//! sets [fuse](crate::data::resources::ResourceSet::fuse) by key with no
//! memory of where a resource came from: strictly for authoring, where a
//! repeated key is a mistake, or as an overlay for an override, where the
//! right-hand resource set wins.
//!
//! A kit's rows are data: the builds of the kit's contract in the built-in
//! build catalog ([`catalog`](crate::data::catalog)), which a
//! [`CatalogBuildProvider`] serves as rows named in full
//! (`qwen/qwen3/qwen3-0.6b`), with the presets and arguments the catalog
//! resolves. Its files carry their sizes and dtypes, and a local directory
//! names where another tool keeps its files as path
//! templates ([`resolve_dir_template`]).
//!
//! ## Providers
//!
//! A [`BuildProvider`] is a namespace over rows: the `provider` of
//! `provider:ref`. It looks a name up, and lists what it has when it can.
//! Three are in use:
//!
//! - `catalog:` ([`CATALOG`]) holds the compiled-in rows, a [`PretrainedTable`]
//!   of [`PretrainedGroup`]s. `catalog:openai/tiny` is the `tiny` row of the
//!   `openai` group; a bare `openai/tiny`, or `tiny`, finds it too.
//! - `bundled:` ([`BUNDLED`]) holds the rows a build ships with: a table a
//!   caller registers over files it lays out or bytes it links in.
//! - `hf:` ([`HF`]) is an [`HfProvider`]: a Hugging Face repo, by `org/repo`.
//!   It lists nothing and never answers a bare name.
//!
//! A [`BuildFactory`]
//! holds a kit's providers in search order and offers a spec to them by its
//! [dispatch rules]. A provider's `Ok(None)` means "not here", and an error
//! aborts the lookup.
//!
//! [dispatch rules]: crate::data::builds::BuildFactory#dispatch
//!
//! Files, the pinned cache and loaded resource sets are
//! [`resources`](crate::data::resources).
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
//!         builds::{
//!             BuildFactory,
//!             BuildRef,
//!             CATALOG,
//!             ConstructInit,
//!             Deferred,
//!             Pretrained,
//!             PretrainedGroup,
//!             PretrainedTable,
//!         },
//!         cache::{
//!             BunsenDiskCacheOptions,
//!             sha256_of,
//!         },
//!         resources::{
//!             LoadedResourceSet,
//!             Provenance,
//!             Resource,
//!             ResourceCache,
//!             ResourceCacheOptions,
//!             ResourceSet,
//!             Source,
//!         },
//!     },
//!     errors::{
//!         BunsenResult,
//!         sys_at,
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
//!     fn for_resource_set(_map: &ResourceSet) -> BunsenResult<Self> {
//!         Ok(Greeting)
//!     }
//!
//!     fn try_init(
//!         &self,
//!         _model: &BuildRef,
//!         loaded: &LoadedResourceSet,
//!         _device: &Device,
//!         _policy: bunsen::data::formats::LoadPolicy,
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
//!     resources: ResourceSet::new("hello").with_resource(Resource {
//!         slot: "text".to_string(),
//!         file: "hello.txt".to_string(),
//!         sha256: Some(sha256_of(&upstream.join("hello.txt"))?),
//!         format: Some("text".to_string()),
//!         size: None,
//!         dtype: None,
//!         name: "toy".to_string(),
//!         sources: vec![Source::LocalDir {
//!             name: "upstream".to_string(),
//!             dir: Some(upstream.clone()),
//!         }],
//!     }),
//!     presets: Default::default(),
//!     args: Default::default(),
//!     tags: Default::default(),
//! };
//!
//! // The kit's index: one provider, a table of one group.
//! let table = PretrainedTable {
//!     name: CATALOG.to_string(),
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
//!     BuildFactory::<Greeting>::new().with_provider(Arc::new(table))?;
//! assert_eq!(factory.ids(), ["catalog:toy/hello"]);
//!
//! // A cache rooted in the temporary directory, and offline.
//! let cache = ResourceCache::new(
//!     ResourceCacheOptions::default()
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
//!     bunsen::data::formats::LoadPolicy::as_stored(),
//! )?;
//! assert_eq!(loaded.name, "catalog:toy/hello");
//! assert_eq!(*loaded.handle, "hello");
//! let text = loaded.resources.get("text").expect("the text");
//! assert_eq!(text.provenance, Provenance::LocalDir);
//! assert!(!tmp.path().join("cache").exists(), "nothing was copied in");
//!
//! // A path is not the factory's business: it is a given resource set.
//! let given = Deferred::<Greeting>::from_resource_set(ResourceSet::given(
//!     "mine",
//!     "text",
//!     upstream.join("hello.txt"),
//! ))?;
//! let loaded = given.load(
//!     &cache,
//!     &Default::default(),
//!     bunsen::data::formats::LoadPolicy::as_stored(),
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
//!
//! [`ResourceCache`]: crate::data::resources::ResourceCache
//! [`Resource`]: crate::data::resources::Resource
//! [`ResourceSet`]: crate::data::resources::ResourceSet
//! [`ResourceSet::fuse`]: crate::data::resources::ResourceSet::fuse
//! [`Source`]: crate::data::resources::Source
//! [`resolve_dir_template`]: crate::data::resources::resolve_dir_template

mod build_ref;
mod catalog_provider;
#[cfg(feature = "cache")]
mod deferred;
#[cfg(all(feature = "cache", feature = "kits"))]
mod factory;
mod hf;
#[cfg(feature = "cache")]
mod init;
#[cfg(feature = "cache")]
mod lookup;
mod providers;
mod rows;

pub use build_ref::*;
pub use catalog_provider::*;
#[cfg(feature = "cache")]
pub use deferred::*;
#[cfg(all(feature = "cache", feature = "kits"))]
pub use factory::*;
pub use hf::*;
#[cfg(feature = "cache")]
pub use init::*;
#[cfg(feature = "cache")]
pub use lookup::*;
pub use providers::*;
pub use rows::*;
