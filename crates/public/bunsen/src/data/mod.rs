//! # Files, caches and pretrained models
//!
//! Where files land, what a name means, and how a name becomes a model.
//! Three submodules, each built on the one before:
//!
//! - [`cache`]: where files land. A [`BunsenDiskCache`] picks a cache directory
//!   and a data directory, and reports every transfer to its observers.
//!   Digests, links and `.partial` files are here too. With the `fetch`
//!   feature, so are the downloads.
//! - [`pretrained`]: what a name means, and how it becomes a model. A *prefab*
//!   is a geometry by name, a *row* is a name over a resource map, and a
//!   *resource* is one pinned file with the places it can be had from.
//!   Providers answer names with rows, a [`PretrainedFactory`] dispatches a
//!   spec to them, and a [`PretrainedCache`] brings the files local for the
//!   kit's [`Construct`] hook to build from.
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
//!   │ PretrainedFactory::resolve(spec, &cache)   │ ResourceMap::given,
//!   │   provider:ref -> that provider only       │ or deserialized
//!   │   bare -> each provider that answers       │
//!   │     bare names, in order; first row wins   │
//!   │   PretrainedProvider::resolve:             │
//!   │     a table answers from its rows;         │
//!   │     HfProvider asks the hub once and       │
//!   │     keeps the listing in the cache         │
//!   ▼                                            │
//! Pretrained row: name, aliases, prefab?,        │
//!   one ResourceMap (its maps fused strictly)    │
//!   │ PretrainedRef::Named                       │ PretrainedRef::Given
//!   ▼                                            ▼
//! Deferred<H>: the ref, and the hook H::for_map(map) picks for it
//!   │ [with_overlay(map)]: the caller's resources win, by key
//!   │ load(&cache, &device):
//!   │   H::plan(ref, cache)              -> the completed ResourceMap
//!   │   PretrainedCache::load(KIT, map)  -> LoadedResources
//!   │     per resource: the cache path, else a local dir (used in
//!   │     place) or bundled bytes (written in), else its URLs,
//!   │     fetched together under the FetchPolicy [fetch]
//!   │   H::construct(ref, loaded, device) -> Arc<H::Built<B>>
//!   ▼
//! Loaded<H::Built<B>>: name, handle (an Arc), resources
//! ```
//!
//! - **Spec to row.** [`PretrainedFactory::resolve`] offers a spec to its
//!   [`PretrainedProvider`]s by the [dispatch rules]. A [`PretrainedTable`]
//!   answers from compiled-in rows, and an [`HfProvider`] from the hub's file
//!   listing. The answer is a [`Pretrained`] row over a [`ResourceMap`] of
//!   [`Resource`]s. A path or a manifest skips this step, since it is a map
//!   already.
//! - **Row to deferred model.** The row, or the given map, becomes a
//!   [`PretrainedRef`]. [`Deferred::new`] or [`Deferred::from_map`] pairs it
//!   with the kit's hook, chosen by [`Construct::for_map`] from the resources'
//!   `kind`s. [`Deferred::with_overlay`] lays the caller's resources over the
//!   row's.
//! - **Deferred to loaded.** [`Deferred::load`] runs the hook's
//!   [`plan`](pretrained::Construct::plan), then [`PretrainedCache::load`] (in
//!   its [source order], into [`LoadedResources`]), then the hook's
//!   [`construct`](pretrained::Construct::construct). The result is a
//!   [`Loaded`] handle.
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
//! - [`default_whisper_factory`] is the full one: the well-known table, the
//!   bundled row with `whisper-weights`, then `hf:org/repo`.
//! - [`default_silero_factory`] is bundled-only: one row, linked into the
//!   binary with `silero-weights`, and no providers without it.
//!
//! ## Features
//!
//! | Feature | Default | What it adds here |
//! |---------|---------|-------------------|
//! | `cache` | yes, through `store` | the [`cache`] module; [`PretrainedCache`], [`PretrainedFactory`], [`Deferred`], [`Construct`], [`Loaded`], [`LoadedResources`]; [`ShardSet`]. All local. |
//! | `fetch` | no | the network at run time: `fetch_file`, `fetch_many` and its `FetchPolicy`, the URL sources of [`PretrainedCache`], the download in `ShardSet::fetch`, and the [`HfProvider`] listing. Without it, a file that only a URL can supply is `ResourceNotFound`. |
//! | `indicatif` | no | implies `fetch`: a byte progress bar per transfer (`IndicatifObserver`, in the default observer stack). |
//! | `store_safetensors` | yes | [`SafetensorsCheckpoint`], the reader for `hf:` rows. `store_pytorch` (also default) is what kits read `.pt` checkpoints with. |
//! | `silero-weights` | no | `bundled:silero/vad`: the Silero burnpack, linked into the binary. |
//! | `whisper-weights` | no | `bundled:openai/base`, served in place from `bunsen-bundled-whisper`'s build directory. **The build reaches the network** on a cold cache. |
//!
//! `download` is a conventional name (see `STYLE.md`), not a bunsen
//! feature: it means "the build may reach the network". A crate that
//! offers it turns on the asset features, as `whisper-model-validation`'s
//! `download` turns on `whisper-weights`. `fetch` is the run-time
//! counterpart, and neither implies the other.
//!
//! [`BunsenDiskCache`]: cache::BunsenDiskCache
//! [`Construct`]: pretrained::Construct
//! [`Construct::for_map`]: pretrained::Construct::for_map
//! [`Deferred`]: pretrained::Deferred
//! [`Deferred::from_map`]: pretrained::Deferred::from_map
//! [`Deferred::load`]: pretrained::Deferred::load
//! [`Deferred::new`]: pretrained::Deferred::new
//! [`Deferred::with_overlay`]: pretrained::Deferred::with_overlay
//! [`HfProvider`]: pretrained::HfProvider
//! [`Loaded`]: pretrained::Loaded
//! [`LoadedResources`]: pretrained::LoadedResources
//! [`Pretrained`]: pretrained::Pretrained
//! [`PretrainedCache`]: pretrained::PretrainedCache
//! [`PretrainedCache::load`]: pretrained::PretrainedCache::load
//! [`PretrainedFactory`]: pretrained::PretrainedFactory
//! [`PretrainedFactory::for_prefab`]: pretrained::PretrainedFactory::for_prefab
//! [`PretrainedFactory::load`]: pretrained::PretrainedFactory::load
//! [`PretrainedFactory::resolve`]: pretrained::PretrainedFactory::resolve
//! [`PretrainedProvider`]: pretrained::PretrainedProvider
//! [`PretrainedRef`]: pretrained::PretrainedRef
//! [`PretrainedRef::prefab`]: pretrained::PretrainedRef::prefab
//! [`PretrainedTable`]: pretrained::PretrainedTable
//! [`Resource`]: pretrained::Resource
//! [`ResourceMap`]: pretrained::ResourceMap
//! [`SafetensorsCheckpoint`]: pretrained::SafetensorsCheckpoint
//! [`ShardSet`]: shards::ShardSet
//! [`ShardSetDescriptor`]: shards::ShardSetDescriptor
//! [`StaticPreFabMap`]: pretrained::StaticPreFabMap
//! [`StaticShardSetMap`]: shards::StaticShardSetMap
//! [`default_resnet_factory`]: crate::kits::images::resnet::pretrained::default_resnet_factory
//! [`default_silero_factory`]: crate::kits::speech::silero_vad::pretrained::default_silero_factory
//! [`default_whisper_factory`]: crate::kits::speech::whisper::pretrained::default_whisper_factory
//! [dispatch rules]: pretrained::PretrainedFactory#dispatch
//! [source order]: pretrained::PretrainedCache#source-order

#[cfg(feature = "cache")]
pub mod cache;

pub mod pretrained;

pub mod shards;
