//! # Resources: pinned files, the places they come from, and their cache
//!
//! A [`Resource`] is one file: the slot it fills in a set (`checkpoint`,
//! `vocabulary`: the kits' constants), the file name it lands as, the SHA-256
//! that pins it, its format, its full name in the catalog (or `given`), and
//! the [`Source`]s it can be had from, in order. A [`ResourceSet`] is a named
//! set of them, and two sets [fuse](ResourceSet::fuse) by slot: strictly for
//! authoring, where a repeated slot is a mistake, or as an overlay, where the
//! right-hand set wins. A [`ResourceCache`] brings a set's files local, into
//! a [`LoadedResourceSet`].
//!
//! None of this is about models: the kits' builds and the dataset shard sets
//! both load through it. Everything but the plain data (the resources,
//! sources and sets) needs the `cache` feature.
//!
//! ## The cache and bundles
//!
//! The [`ResourceCache`] keeps a pinned file at
//! `<cache>/resources/<name>/<sha256>/<file>`, `<name>` a catalog
//! resource's full name (`hf/<org>/<repo>/<slot>` for a Hub repo's, `given`
//! for a path's). The digest in
//! the path is the pin: a file there was verified when it was written, and
//! is trusted without re-hashing (the
//! [trust model](ResourceCache#trust-model)). The cache looks there
//! before it tries any of a resource's sources (the
//! [source order](ResourceCache#source-order)).
//! [`cached_files`](ResourceCache::cached_files) lists what a cache holds,
//! a [`CachedFile`] each, by kit and namespace ([`CacheSelector`]), and
//! [`remove_cached`](ResourceCache::remove_cached) clears them.
//!
//! A bundle takes one of two forms:
//!
//! - Bytes linked into the binary are a [`Source::Bundled`], written into the
//!   cache under their digest on first use.
//! - Files fetched at build time are a populated cache directory.
//!   `bind-whisper-onnx` lays its `OUT_DIR` out in this layout, so a
//!   [`ResourceCache`] rooted there finds the files without knowing they were
//!   bundled. (`whisper-model-validation` also imports a catalog layer that
//!   adds the same files to `openai/whisper/base` as local-dir sources, served
//!   in place.)
//!
//! The second form is the general one: bundling a fetched asset means
//! populating a cache directory, not adding a kind of source. A deployment
//! that has to run offline does the same thing at install time: it fills
//! the cache directory ahead of time (`bunsen-tool asr family <family> fetch`)
//! and points the cache at it.

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
mod cached_files;
pub mod hub;
#[cfg(feature = "cache")]
mod loaded;
mod resource;
mod resource_set;

#[cfg(feature = "cache")]
pub use cache::*;
#[cfg(feature = "cache")]
pub use cached_files::*;
#[cfg(feature = "cache")]
pub use loaded::*;
pub use resource::*;
pub use resource_set::*;
