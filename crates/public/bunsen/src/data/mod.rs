//! # Files, caches and pretrained models
//!
//! Where files land, what a name means, and how a name becomes a model.
//! Six submodules, each built on the one before:
//!
//! - [`cache`]: where files land. A [`BunsenDiskCache`] picks a cache directory
//!   and a data directory, and reports every transfer to its observers.
//!   Digests, links and `.partial` files are here too. With the `fetch`
//!   feature, so are the downloads.
//! - [`resources`]: pinned files. A *resource* is one pinned file with the
//!   places it can be had from, a [`ResourceBundle`] a named set of them, and a
//!   [`PretrainedCache`] brings a bundle's files local.
//! - [`catalog`]: what there is to build, as data. A layered document of
//!   contracts, builds, resources, tags and presets, the built-in one compiled
//!   in, which resolves a build's name to its files and arguments and chooses
//!   among builds by their tags.
//! - [`formats`]: reading the files: safetensors, `PyTorch`, MLX and the Phonon
//!   formats, named tensors applied to a module, and the
//!   [`LoadPolicy`](formats::LoadPolicy) a load places its floats by.
//! - [`pretrained`]: what a name means, and how it becomes a model. A *prefab*
//!   is a geometry by name, and a *row* is a name over a resource bundle.
//!   Providers answer names with rows, a [`PretrainedFactory`] dispatches a
//!   spec to them, and the kit's [`ConstructInit`] hook builds from the files
//!   the cache brought local.
//! - [`shards`]: numbered dataset files. A [`ShardSetDescriptor`] names a
//!   family of files (`shard_00000.parquet`, ...), and a [`ShardSet`] binds one
//!   to a directory and brings shards in through the disk cache.
//!
//! ## Lifecycle
//!
//! From a name to a model, as [`PretrainedFactory::load`] runs it:
//!
//! ```text
//! spec: "provider:ref" or a bare name         a path, a manifest
//!   │                                            │
//!   │ PretrainedFactory::resolve(spec, &cache)   │ ResourceBundle::given,
//!   │   provider:ref -> that provider only       │ or deserialized
//!   │   bare -> each provider that answers       │
//!   │     bare names, in order; first row wins   │
//!   │   PretrainedProvider::resolve:             │
//!   │     a table answers from its rows;         │
//!   │     HfProvider asks the hub once and       │
//!   │     keeps the listing in the cache         │
//!   ▼                                            │
//! Pretrained row: name, aliases, prefab?,        │
//!   one ResourceBundle (its resource bundles fused strictly)    │
//!   │ PretrainedRef::Named                       │ PretrainedRef::Given
//!   ▼                                            ▼
//! Deferred<H>: the ref, and the hook H::for_resource_bundle(resource bundle) picks for it
//!   │ [with_overlay(resource bundle)]: the caller's resources win, by key
//!   │ load(&cache, &device):
//!   │   H::plan(ref, cache)              -> the completed ResourceBundle
//!   │   PretrainedCache::load(KIT, resource bundle)  -> LoadedResources
//!   │     per resource: the cache path, else a local dir (used in
//!   │     place) or bundled bytes (written in), else its URLs,
//!   │     fetched together under the FetchPolicy [fetch]
//!   │   H::try_init(ref, loaded, device) -> Arc<H::Construct>
//!   ▼
//! Loaded<H::Construct>: name, handle (an Arc), resources
//! ```
//!
//! - **Spec to row.** [`PretrainedFactory::resolve`] offers a spec to its
//!   [`PretrainedProvider`]s by the [dispatch rules]. A [`PretrainedTable`]
//!   answers from compiled-in rows, and an [`HfProvider`] from the hub's file
//!   listing. The answer is a [`Pretrained`] row over a [`ResourceBundle`] of
//!   [`Resource`]s. A path or a manifest skips this step, since it is a
//!   resource bundle already.
//! - **Row to deferred model.** The row, or the given resource bundle, becomes
//!   a [`PretrainedRef`]. [`Deferred::new`] or
//!   [`Deferred::from_resource_bundle`] pairs it with the kit's hook, chosen by
//!   [`ConstructInit::for_resource_bundle`] from the resources' `kind`s.
//!   [`Deferred::with_overlay`] lays the caller's resources over the row's.
//! - **Deferred to loaded.** [`Deferred::load`] runs the hook's
//!   [`plan`](crate::kits::support::construct::ConstructInit::plan), then
//!   [`PretrainedCache::load`] (in its [source order], into
//!   [`LoadedResources`]), then the hook's
//!   [`try_init`](crate::kits::support::construct::ConstructInit::try_init).
//!   The result is a [`Loaded`] handle.
//!
//! Prefabs sit beside this pathway. A kit's [`StaticPreFabMap`] names its
//! geometries, and a row names the prefab it instantiates;
//! [`PretrainedRef::prefab`] and [`PretrainedFactory::for_prefab`] join the
//! two.
//!
//! Shards take a shorter path: a [`StaticShardSetMap`] entry becomes a
//! [`ShardSetDescriptor`], a [`ShardSet`] binds it to a directory, and its
//! `locate` / `fetch` / `fetch_many` give paths to a data loader. See
//! [`shards`].
//!
//! ## The kits' factories
//!
//! Each kit with pretrained models ships a `default_{kit}_factory()` over
//! its compiled-in providers. There is no process-wide registry.
//!
//! - [`default_resnet_factory`] is the simple case: a well-known table of
//!   `torchvision` and `timm` checkpoints, each row naming its prefab.
//! - [`default_whisper_factory`] is the full one: the well-known table, then
//!   `hf:org/repo`.
//! - [`default_silero_factory`] has one well-known row, `silero/vad-16k`,
//!   upstream's safetensors.
//!
//! ## Features
//!
//! | Feature | Default | What it adds here |
//! |---------|---------|-------------------|
//! | `cache` | yes, through `store` | the [`cache`] module; [`PretrainedCache`], [`PretrainedFactory`], [`Deferred`], [`ConstructInit`], [`Loaded`], [`LoadedResources`]; [`ShardSet`]. All local. |
//! | `fetch` | no | the network at run time: `fetch_file`, `fetch_many` and its `FetchPolicy`, the URL sources of [`PretrainedCache`], the download in `ShardSet::fetch`, and the [`HfProvider`] listing. Without it, a file that only a URL can supply is an [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) error. |
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
//! [`ConstructInit`]: crate::kits::support::construct::ConstructInit
//! [`ConstructInit::for_resource_bundle`]: crate::kits::support::construct::ConstructInit::for_resource_bundle
//! [`Deferred`]: crate::kits::support::construct::Deferred
//! [`Deferred::from_resource_bundle`]: crate::kits::support::construct::Deferred::from_resource_bundle
//! [`Deferred::load`]: crate::kits::support::construct::Deferred::load
//! [`Deferred::new`]: crate::kits::support::construct::Deferred::new
//! [`Deferred::with_overlay`]: crate::kits::support::construct::Deferred::with_overlay
//! [`HfProvider`]: pretrained::HfProvider
//! [`Loaded`]: crate::kits::support::construct::Loaded
//! [`LoadedResources`]: resources::LoadedResources
//! [`Pretrained`]: pretrained::Pretrained
//! [`PretrainedCache`]: resources::PretrainedCache
//! [`PretrainedCache::load`]: resources::PretrainedCache::load
//! [`PretrainedFactory`]: crate::kits::support::construct::PretrainedFactory
//! [`PretrainedFactory::for_prefab`]: crate::kits::support::construct::PretrainedFactory::for_prefab
//! [`PretrainedFactory::load`]: crate::kits::support::construct::PretrainedFactory::load
//! [`PretrainedFactory::resolve`]: crate::kits::support::construct::PretrainedFactory::resolve
//! [`PretrainedProvider`]: pretrained::PretrainedProvider
//! [`PretrainedRef`]: pretrained::PretrainedRef
//! [`PretrainedRef::prefab`]: pretrained::PretrainedRef::prefab
//! [`PretrainedTable`]: pretrained::PretrainedTable
//! [`Resource`]: resources::Resource
//! [`ResourceBundle`]: resources::ResourceBundle
//! [`SafetensorsCheckpoint`]: crate::data::formats::SafetensorsCheckpoint
//! [`ShardSet`]: shards::ShardSet
//! [`ShardSetDescriptor`]: shards::ShardSetDescriptor
//! [`StaticPreFabMap`]: pretrained::StaticPreFabMap
//! [`StaticShardSetMap`]: shards::StaticShardSetMap
//! [`default_resnet_factory`]: crate::kits::images::resnet::pretrained::default_resnet_factory
//! [`default_silero_factory`]: crate::kits::speech::vad::silero_vad::pretrained::default_silero_factory
//! [`default_whisper_factory`]: crate::kits::speech::asr::whisper::pretrained::default_whisper_factory
//! [dispatch rules]: crate::kits::support::construct::PretrainedFactory#dispatch
//! [source order]: resources::PretrainedCache#source-order

#[cfg(feature = "cache")]
pub mod cache;

pub mod catalog;

pub mod formats;

pub mod pretrained;

pub mod resources;

pub mod shards;

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
