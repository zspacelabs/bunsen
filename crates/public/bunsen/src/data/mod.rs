//! # Files, caches and pretrained models
//!
//! Where files land, what a name means, and how a name becomes a model.
//! Seven submodules, each built on the one before:
//!
//! - [`cache`]: where files land. A [`BunsenDiskCache`] picks a cache directory
//!   and a data directory, and reports every transfer to its observers.
//!   Digests, links and `.partial` files are here too. With the `fetch`
//!   feature, so are the downloads.
//! - [`resources`]: pinned files. A *resource* is one pinned file with the
//!   places it can be had from, a [`ResourceSet`] a named set of them, and a
//!   [`ResourceCache`] brings a bundle's files local.
//! - [`catalog`]: what there is to build, as data. A layered document of
//!   contracts, builds, resources, tags and presets, the built-in one compiled
//!   in, which resolves a build's name to its files and arguments and chooses
//!   among builds by their tags.
//! - [`formats`]: reading the files: safetensors, `PyTorch`, MLX and the Phonon
//!   formats, named tensors applied to a module, and the
//!   [`LoadPolicy`](formats::LoadPolicy) a load places its floats by.
//! - [`builds`]: what a name means, and how it becomes a model. A *row* is a
//!   name over a resource set; providers answer names with rows, a
//!   [`BuildFactory`] dispatches a spec to them, and the kit's
//!   [`ConstructInit`] hook builds from the files the cache brought local.
//! - [`pretrained`]: the kits' prefabs (a geometry by name) and tags.
//! - [`datasets`]: dataset shard sets, numbered files. A [`ShardSetDescriptor`]
//!   names a family of files (`shard_00000.parquet`, ...), and a [`ShardSet`]
//!   binds one to a directory and brings shards in through the disk cache.
//!
//! ## Lifecycle
//!
//! From a name to a model, as [`BuildFactory::load`] runs it:
//!
//! ```text
//! spec: "provider:ref" or a bare name         a path, a manifest
//!   │                                            │
//!   │ BuildFactory::resolve(spec, &cache)        │ ResourceSet::given,
//!   │   provider:ref -> that provider only       │ or deserialized
//!   │   bare -> each provider that answers       │
//!   │     bare names, in order; first wins       │
//!   │   BuildProvider::resolve:                  │
//!   │     a catalog answers from its builds;     │
//!   │     HfProvider asks the hub once and       │
//!   │     keeps the listing in the cache         │
//!   ▼                                            │
//! ResolvedBuild: name, aliases, tags, presets,   │
//!   args, resources in the catalog's form        │
//!   │ BuildRef::Named                            │ BuildRef::Given
//!   ▼                                            ▼
//! Deferred<H>: the ref, and the hook H::for_resource_set(its files) picks
//!   │ [with_overlay(resource set)]: the caller's resources win, by slot
//!   │ load(&cache, &device):
//!   │   H::plan(ref, cache)              -> the completed ResourceSet
//!   │   ResourceCache::load(KIT, resource set)  -> LoadedResourceSet
//!   │     per resource: the cache path, else a local dir (used in
//!   │     place) or bundled bytes (written in), else its URLs,
//!   │     fetched together under the FetchPolicy [fetch]
//!   │   H::try_init(ref, loaded, device) -> Arc<H::Construct>
//!   ▼
//! Loaded<H::Construct>: name, handle (an Arc), resources
//! ```
//!
//! - **Spec to build.** [`BuildFactory::resolve`] offers a spec to its
//!   [`BuildProvider`]s by the [dispatch rules]. A [`CatalogBuildProvider`]
//!   answers from a catalog's builds, and an [`HfProvider`] from the hub's file
//!   listing. The answer is a [`ResolvedBuild`], whose files in the loader's
//!   form are a [`ResourceSet`] of [`Resource`]s. A path or a manifest skips
//!   this step, since it is a resource set already.
//! - **Build to deferred model.** The build, or the given resource set, becomes
//!   a [`BuildRef`]. [`Deferred::new`] or [`Deferred::from_resource_set`] pairs
//!   it with the kit's hook, chosen by [`ConstructInit::for_resource_set`] from
//!   the resources' `format`s. [`Deferred::with_overlay`] lays the caller's
//!   resources over the build's.
//! - **Deferred to loaded.** [`Deferred::load`] runs the hook's
//!   [`plan`](crate::data::builds::ConstructInit::plan), then
//!   [`ResourceCache::load`] (in its [source order], into
//!   [`LoadedResourceSet`]), then the hook's
//!   [`try_init`](crate::data::builds::ConstructInit::try_init). The result is
//!   a [`Loaded`] handle.
//!
//! Prefabs sit beside this pathway. A kit's config names its geometries
//! ([`Prefabs`]), and a build states the geometry it is built in
//! ([`BuildRef::stated`]), which a kit's tests check against its prefabs.
//!
//! Shards take a shorter path: a dataset's build becomes a
//! [`ShardSetDescriptor`], a [`ShardSet`] binds it to a directory, and its
//! `locate` / `fetch` / `fetch_many` give paths to a data loader. See
//! [`datasets`].
//!
//! ## The kits' factories
//!
//! Each kit with pretrained models ships a `default_{kit}_factory()` over
//! its compiled-in providers. There is no process-wide registry.
//!
//! - [`default_resnet_factory`] is the simple case: a catalog provider of
//!   `torchvision` and `timm` checkpoints, each row naming its prefab.
//! - [`default_whisper_factory`] is the full one: the catalog provider, then
//!   `hf:org/repo`.
//! - [`default_silero_factory`] has one catalog row, `silero/vad-16k`,
//!   upstream's safetensors.
//!
//! ## Features
//!
//! | Feature | Default | What it adds here |
//! |---------|---------|-------------------|
//! | `cache` | yes, through `store` | the [`cache`] module; [`ResourceCache`], [`BuildFactory`], [`Deferred`], [`ConstructInit`], [`Loaded`], [`LoadedResourceSet`]; [`ShardSet`]. All local. |
//! | `fetch` | no | the network at run time: `fetch_file`, `fetch_many` and its `FetchPolicy`, the URL sources of [`ResourceCache`], the download in `ShardSet::fetch`, and the [`HfProvider`] listing. Without it, a file that only a URL can supply is an [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) error. |
//! | `indicatif` | no | implies `fetch`: a byte progress bar per transfer (`IndicatifObserver`, in the default observer stack). |
//! | `store` | yes | the checkpoint readers: [`SafetensorsCheckpoint`] (the reader for `hf:` rows), the `PyTorch` reader kits read `.pt` checkpoints with, the burnpack, `.nemo` and Phonon ([`fermion`](formats::fermion)) readers. |
//!
//! No bunsen feature reaches the network at build time. `reference` is a
//! conventional name (see `STYLE.md`), not a bunsen feature: it means "the
//! build may reach the network", and only the validation crates offer it,
//! turning on the `bind-*-onnx` crates' asset features. `fetch` is the
//! run-time counterpart, and neither implies the other.
//!
//! [`BunsenDiskCache`]: cache::BunsenDiskCache
//! [`ConstructInit`]: crate::data::builds::ConstructInit
//! [`ConstructInit::for_resource_set`]: crate::data::builds::ConstructInit::for_resource_set
//! [`Deferred`]: crate::data::builds::Deferred
//! [`Deferred::from_resource_set`]: crate::data::builds::Deferred::from_resource_set
//! [`Deferred::load`]: crate::data::builds::Deferred::load
//! [`Deferred::new`]: crate::data::builds::Deferred::new
//! [`Deferred::with_overlay`]: crate::data::builds::Deferred::with_overlay
//! [`HfProvider`]: crate::data::builds::HfProvider
//! [`Loaded`]: crate::data::builds::Loaded
//! [`LoadedResourceSet`]: resources::LoadedResourceSet
//! [`ResourceCache`]: resources::ResourceCache
//! [`ResourceCache::load`]: resources::ResourceCache::load
//! [`BuildFactory`]: crate::data::builds::BuildFactory
//! [`BuildRef::stated`]: crate::data::builds::BuildRef::stated
//! [`CatalogBuildProvider`]: crate::data::builds::CatalogBuildProvider
//! [`ResolvedBuild`]: crate::data::catalog::ResolvedBuild
//! [`BuildFactory::load`]: crate::data::builds::BuildFactory::load
//! [`BuildFactory::resolve`]: crate::data::builds::BuildFactory::resolve
//! [`BuildProvider`]: crate::data::builds::BuildProvider
//! [`BuildRef`]: crate::data::builds::BuildRef
//! [`Resource`]: resources::Resource
//! [`ResourceSet`]: resources::ResourceSet
//! [`SafetensorsCheckpoint`]: crate::data::formats::SafetensorsCheckpoint
//! [`ShardSet`]: datasets::ShardSet
//! [`ShardSetDescriptor`]: datasets::ShardSetDescriptor
//! [`Prefabs`]: pretrained::Prefabs
//! [`default_resnet_factory`]: crate::kits::images::resnet::pretrained::default_resnet_factory
//! [`default_silero_factory`]: crate::kits::speech::vad::silero_vad::pretrained::default_silero_factory
//! [`default_whisper_factory`]: crate::kits::speech::asr::whisper::pretrained::default_whisper_factory
//! [dispatch rules]: crate::data::builds::BuildFactory#dispatch
//! [source order]: resources::ResourceCache#source-order

#[cfg(feature = "cache")]
pub mod cache;

pub mod builds;

pub mod catalog;

pub mod formats;

pub mod pretrained;

pub mod resources;

pub mod datasets;

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
