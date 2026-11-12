//! # Resources: pinned files, the places they come from, and their cache
//!
//! A [`Resource`] is one file: its key in a set (`checkpoint`, `vocabulary`:
//! the kits' constants), the file name it lands as, the SHA-256 that pins
//! it, a `kind` label, its full name in the catalog (or `given`), and the
//! [`Source`]s it can be had from, in order. A [`ResourceBundle`] is a named
//! set of them, and two bundles [fuse](ResourceBundle::fuse) by key: strictly
//! for authoring, where a repeated key is a mistake, or as an overlay, where
//! the right-hand bundle wins. A [`PretrainedCache`] brings a bundle's files
//! local, into [`LoadedResources`].
//!
//! None of this is about models: the kits' builds and the dataset shard sets
//! both load through it. Everything but the plain data (the resources,
//! sources and bundles) needs the `cache` feature.
//!
//! ## The cache and bundles
//!
//! The [`PretrainedCache`] keeps a pinned file at
//! `<cache>/pretrained/<namespace>/<sha256>/<file>`, `<namespace>` a catalog
//! resource's full name (`hf` or `given` off the catalog). The digest in
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

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
mod cached_files;
#[cfg(feature = "cache")]
mod loaded;
mod resource;
mod resource_bundle;

#[cfg(feature = "cache")]
pub use cache::*;
#[cfg(feature = "cache")]
pub use cached_files::*;
#[cfg(feature = "cache")]
pub use loaded::*;
pub use resource::*;
pub use resource_bundle::*;
