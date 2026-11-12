//! # Builds: from a name to a model
//!
//! A kit's [`BuildFactory`] takes a spec, `provider:ref` or a bare name, and
//! asks its providers for the build the spec names. Loading the build, the
//! [`ResourceCache`] brings its files local and the kit's [`ConstructInit`]
//! hook builds from them, behind an `Arc`. The [`data`](crate::data#lifecycle)
//! overview draws the whole pathway; this page says what the pieces are, and
//! why they are split the way they are.
//!
//! The pieces:
//!
//! - [`BuildProvider`]: a namespace over builds, the `provider` of
//!   `provider:ref`, answering with the catalog's [`ResolvedBuild`].
//! - [`BuildRef`]: what a spec resolved to, a provider's build or a given
//!   resource set.
//! - [`ConstructInit`]: the kit's hook, which completes a resolved ref's
//!   resource set before it is loaded and builds from the loaded parts.
//! - [`DeferredBuild`]: a model not yet loaded, its ref paired with the hook
//!   that will build it.
//! - [`BuildFactory`]: a kit's providers in search order, resolving lookup
//!   expressions to [`DeferredBuild`] models.
//! - [`LookupExpr`]: a lookup expression, `[provider:]name[preset, …]{patch}`,
//!   split into the name and what it asks of the build's presets.
//! - [`Loaded`]: the handle the pathway hands back.
//!
//! The factory, [`DeferredBuild`], [`ConstructInit`], [`Loaded`] and
//! [`LookupExpr`] need the `cache` feature (default, through `store`), and
//! the factory the `kits` feature too, for the kits' preset resolution. The
//! refs and providers are plain data and are always here.
//!
//! A hook reads its checkpoint's own format with the readers in
//! [`formats`](crate::data::formats).
//!
//! ## Builds and resources
//!
//! A **build** is what the catalog says one buildable thing is
//! ([`ResolvedBuild`]): its full name and aliases, its listing fields, its
//! tags (facts about the model, for listings and for choosing among models,
//! [`catalog::select`](crate::data::catalog::select)), its presets, the
//! arguments its contract's slots are bound to (such as the geometry it
//! states, [`GEOMETRY`]), and the resources its slots are bound to, in the
//! catalog's form. A file many builds share is one resource that each binds
//! (Whisper's two vocabularies serve its twelve checkpoints).
//!
//! What a loader reads is the build's [`ResourceSet`]
//! ([`ResolvedBuild::resource_set`]), made when it is loaded: each slot's
//! [`Resource`] (the file name it lands as, the SHA-256 that pins it, its
//! format, its full name, and the [`Source`]s it can be had from, in order),
//! a local directory resolved from its path templates then
//! ([`resolve_dir_template`]).
//!
//! None of these knows a hook, or which resource is the model: the function
//! that loads a kit's models is the kit's, and it supplies the hook. A kit's
//! builds are data: the builds of the kit's contract in the built-in catalog
//! ([`catalog`](crate::data::catalog)), which a [`CatalogBuildProvider`]
//! serves named in full (`qwen/qwen3/qwen3-0.6b`).
//!
//! ## Providers
//!
//! A [`BuildProvider`] is a namespace over builds: the `provider` of
//! `provider:ref`. It looks a name up, and lists what it has when it can.
//! Two are in use:
//!
//! - `catalog:` ([`CATALOG`]) is a [`CatalogBuildProvider`]: the builds of the
//!   kit's contract in the built-in catalog, and in any extension layer
//!   imported over it. `catalog:openai/whisper/tiny` is the build; a bare
//!   `openai/tiny`, or `tiny`, finds it too. Files a deployment ships are such
//!   a layer: an update of the builds' resources that adds the files as
//!   local-dir sources, so the same name loads them.
//! - `hf:` ([`HF`]) is an [`HfProvider`]: a Hugging Face repo, by `org/repo`,
//!   answered with a build named `hf/<org>/<repo>`. It lists nothing and never
//!   answers a bare name.
//!
//! A [`BuildFactory`] holds a kit's providers in search order and offers a
//! spec to them by its [dispatch rules]. A provider's `Ok(None)` means "not
//! here", and an error aborts the lookup.
//!
//! [dispatch rules]: crate::data::builds::BuildFactory#dispatch
//!
//! Files, the pinned cache and loaded resource sets are
//! [`resources`](crate::data::resources).
//!
//! ## Example
//!
//! A toy kit whose model is the text of one file: a catalog of one build over
//! a directory another tool keeps, and an offline cache in a temporary
//! directory. Nothing here reaches the network.
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
//!             CatalogBuildProvider,
//!             ConstructInit,
//!             DeferredBuild,
//!         },
//!         cache::BunsenDiskCacheOptions,
//!         catalog::BunsenCatalogStack,
//!         formats::LoadPolicy,
//!         resources::{
//!             LoadedResourceSet,
//!             Provenance,
//!             ResourceCache,
//!             ResourceCacheOptions,
//!             ResourceSet,
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
//!         _policy: LoadPolicy,
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
//! // The kit's contract, and one build of it, `toy/hello` (alias `hi`),
//! // whose one file is had from that directory.
//! let mut stack = BunsenCatalogStack::new();
//! stack.push_yaml(
//!     "prelude",
//!     r#"
//! $contracts:
//!   greeting:
//!     $kind: contract
//!     $description: "A greeting, as text."
//!     $resources: { text: { $format: [text] } }
//! template_class_root: { $kind: template_class, $description: "The root." }
//! "#,
//! )?;
//! stack.push_yaml(
//!     "toy",
//!     &format!(
//!         r#"
//! toy:
//!   $kind: namespace
//!   $description: "Toy builds."
//!   hi: {{ $kind: alias, $ref: {{ $path: hello, $kind: build }} }}
//!   hello:
//!     $kind: build
//!     $description: "A greeting."
//!     $contract: greeting
//!     $resources: {{ text: text }}
//!     text:
//!       $kind: resource
//!       $file: hello.txt
//!       $format: text
//!       $sources: [{{ $local_dir: {{ $name: upstream, $default: ["{}"] }} }}]
//! "#,
//!         upstream.display()
//!     ),
//! )?;
//! let catalog = CatalogBuildProvider::new(
//!     CATALOG,
//!     "the toy builds",
//!     &stack,
//!     "greeting",
//! )?;
//! let factory =
//!     BuildFactory::<Greeting>::new().with_provider(Arc::new(catalog))?;
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
//! // The whole pathway: the alias resolves to the build, the file is used in
//! // place, and the hook builds.
//! let loaded = factory.load(
//!     "hi",
//!     &cache,
//!     &Default::default(),
//!     LoadPolicy::load_as_stored(),
//! )?;
//! assert_eq!(loaded.name, "catalog:toy/hello");
//! assert_eq!(*loaded.handle, "hello");
//! let text = loaded.resources.get("text").expect("the text");
//! assert_eq!(text.provenance, Provenance::LocalDir);
//! assert!(!tmp.path().join("cache").exists(), "nothing was copied in");
//!
//! // A path is not the factory's business: it is a given resource set.
//! let given = DeferredBuild::<Greeting>::from_resource_set(
//!     ResourceSet::given("mine", "text", upstream.join("hello.txt")),
//! )?;
//! let loaded = given.load(
//!     &cache,
//!     &Default::default(),
//!     LoadPolicy::load_as_stored(),
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
//! [`ResolvedBuild`]: crate::data::catalog::ResolvedBuild
//! [`ResolvedBuild::resource_set`]: crate::data::catalog::ResolvedBuild::resource_set
//! [`ResourceCache`]: crate::data::resources::ResourceCache
//! [`ResourceSet`]: crate::data::resources::ResourceSet
//! [`Resource`]: crate::data::resources::Resource
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
