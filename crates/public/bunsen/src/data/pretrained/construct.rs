//! # Construction
//!
//! From a resolved ref to what a kit builds. A [`Construct`] hook is the
//! kit's: it says which cache segment its files live under, what it builds
//! for a backend, which hook a map calls for (by what the map says about
//! its resources, their `kind`s), how a ref's map is completed before it is
//! loaded, and how the loaded parts become the built thing, behind an
//! `Arc`. A [`Deferred`](super::Deferred) model pairs a ref with its hook
//! and runs the whole way: plan, load, construct. Nothing here opens a
//! file.

use std::sync::Arc;

use burn::prelude::Backend;

use super::{
    LoadedResources,
    PretrainedCache,
    PretrainedRef,
    ResourceMap,
};
use crate::errors::BunsenResult;

/// What a kit builds from a resolved ref, and how.
///
/// Not object-safe: [`construct`](Self::construct) is generic over the
/// backend so that [`Built`](Self::Built) is a real type. A hook is a
/// value, chosen for a map by [`for_map`](Self::for_map) and carried by the
/// deferred model it builds: different providers' rows may be read by
/// different mechanisms, and the map's resources, their `kind`s, say which.
/// A caller never builds one; it may adjust the one it was given.
///
/// Both steps see the ref, not just its map: what a row promises (its
/// prefab) is checked in [`plan`](Self::plan) before bytes are fetched,
/// and a kit whose checkpoint does not describe itself takes its config
/// from the row in [`construct`](Self::construct).
pub trait Construct: Sized {
    /// The cache segment the kit's files live under: the `<kit>` of
    /// `pretrained/<kit>/<namespace>/<sha256>/<file>`.
    const KIT: &'static str;

    /// What construction yields, for a backend.
    type Built<B: Backend>;

    /// The hook for `map`: the reader its resources' `kind`s call for.
    ///
    /// # Errors
    /// The kit's: a resource of a kind it has no reader for, refused by
    /// name rather than misread.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self>;

    /// Completes the ref's map before it is loaded: a kit rule that names
    /// a resource from another, or checks a declared one against the rule,
    /// or checks the checkpoint against what the row promised. The default
    /// is the map as it stands. It may resolve a resource through the
    /// cache to look at it; a scan that reads shapes only is the intended
    /// use.
    ///
    /// # Errors
    /// The kit's: a map the rule cannot complete, or one that contradicts
    /// it.
    fn plan(
        &self,
        model: &PretrainedRef,
        _cache: &PretrainedCache,
    ) -> BunsenResult<ResourceMap> {
        Ok(model.to_map())
    }

    /// Builds from every part local. Never fetches. `model` is the ref
    /// [`plan`](Self::plan) saw.
    ///
    /// # Errors
    /// The kit's: a part the map lacks
    /// ([`LoadedResources::expect`]), or one that does not read as what its
    /// key says.
    fn construct<B: Backend>(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &B::Device,
    ) -> BunsenResult<Arc<Self::Built<B>>>;
}

/// The handle a pretrained hands back: bound by what was built.
///
/// Cloning shares the handle; the built thing is constructed once.
#[derive(Debug)]
pub struct Loaded<T> {
    /// The map's name: the pretrained's id, or the caller's for a given
    /// map.
    pub name: String,

    /// What was built, shared.
    pub handle: Arc<T>,

    /// Where every part came from.
    pub resources: LoadedResources,
}

impl<T> Clone for Loaded<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            handle: Arc::clone(&self.handle),
            resources: self.resources.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
    };

    use super::*;
    use crate::{
        data::{
            cache::BunsenDiskCacheOptions,
            pretrained::{
                Deferred,
                Fuse,
                PretrainedCacheOptions,
                Provenance,
            },
        },
        errors::BunsenError,
        support::testing::{
            CpuBackend,
            default_device,
        },
    };

    /// A hook that builds the list of its parts' paths, in key order, and
    /// completes a map with a vocabulary when one is missing.
    struct Paths {
        vocabulary: Option<PathBuf>,
    }

    impl Construct for Paths {
        type Built<B: Backend> = Vec<PathBuf>;

        const KIT: &'static str = "paths";

        fn for_map(_map: &ResourceMap) -> BunsenResult<Self> {
            Ok(Paths { vocabulary: None })
        }

        fn plan(
            &self,
            model: &PretrainedRef,
            _cache: &PretrainedCache,
        ) -> BunsenResult<ResourceMap> {
            let map = model.to_map();
            match &self.vocabulary {
                Some(path) if map.get("vocabulary").is_none() => map.fuse(
                    ResourceMap::given("derived", "vocabulary", path),
                    Fuse::Strict,
                ),
                _ => Ok(map),
            }
        }

        fn construct<B: Backend>(
            &self,
            _model: &PretrainedRef,
            loaded: &LoadedResources,
            _device: &B::Device,
        ) -> BunsenResult<Arc<Vec<PathBuf>>> {
            loaded.expect("checkpoint")?;
            Ok(Arc::new(
                loaded.iter().map(|(_, r)| r.path.clone()).collect(),
            ))
        }
    }

    /// A hook that asks for a key no map of these tests has, and takes
    /// the default plan.
    struct NeedsConfig;

    impl Construct for NeedsConfig {
        type Built<B: Backend> = ();

        const KIT: &'static str = "paths";

        fn for_map(_map: &ResourceMap) -> BunsenResult<Self> {
            Ok(NeedsConfig)
        }

        fn construct<B: Backend>(
            &self,
            _model: &PretrainedRef,
            loaded: &LoadedResources,
            _device: &B::Device,
        ) -> BunsenResult<Arc<()>> {
            loaded.expect("config")?;
            Ok(Arc::new(()))
        }
    }

    fn cache_in(dir: &std::path::Path) -> PretrainedCache {
        PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap()
    }

    /// Plan completes the map, load brings every part local, construct
    /// builds; the handle is shared by clones and the resources say where
    /// each part came from.
    #[test]
    fn test_load_plans_loads_and_constructs() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.pt");
        let vocabulary = dir.path().join("gpt2.tiktoken");
        fs::write(&checkpoint, b"x").unwrap();
        fs::write(&vocabulary, b"y").unwrap();
        let cache = cache_in(dir.path());
        let hook = Paths {
            vocabulary: Some(vocabulary.clone()),
        };

        let loaded = Deferred {
            model: PretrainedRef::from(ResourceMap::given("mine", "checkpoint", &checkpoint)),
            hook,
        }
        .load::<CpuBackend>(&cache, &default_device())
        .unwrap();

        assert_eq!(loaded.name, "mine");
        assert_eq!(*loaded.handle, vec![checkpoint, vocabulary]);
        assert_eq!(loaded.resources.keys(), ["checkpoint", "vocabulary"]);
        for (_, part) in loaded.resources.iter() {
            assert_eq!(part.provenance, Provenance::LocalDir);
        }
        assert!(!dir.path().join("cache").exists(), "nothing was written");

        let again = loaded.clone();
        assert_eq!(Arc::strong_count(&loaded.handle), 2);
        assert!(Arc::ptr_eq(&loaded.handle, &again.handle));
    }

    /// With nothing to derive, the default plan is the ref's map, and a
    /// map that already has the key is left alone.
    #[test]
    fn test_plan_leaves_a_complete_map_alone() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.pt");
        let vocabulary = dir.path().join("mine.tiktoken");
        fs::write(&checkpoint, b"x").unwrap();
        fs::write(&vocabulary, b"y").unwrap();
        let cache = cache_in(dir.path());
        let model = PretrainedRef::from(
            ResourceMap::given("mine", "checkpoint", &checkpoint)
                .fuse(
                    ResourceMap::given("flags", "vocabulary", &vocabulary),
                    Fuse::Strict,
                )
                .unwrap(),
        );

        let hook = Paths {
            vocabulary: Some(dir.path().join("never-read.tiktoken")),
        };
        let loaded = Deferred {
            model: model.clone(),
            hook,
        }
        .load::<CpuBackend>(&cache, &default_device())
        .unwrap();
        assert_eq!(*loaded.handle, vec![checkpoint.clone(), vocabulary.clone()]);

        // The hook a map calls for, with no rule of its own.
        let no_rule = Deferred::<Paths>::new(model.clone()).unwrap();
        assert!(no_rule.hook.vocabulary.is_none());
        let loaded = no_rule
            .load::<CpuBackend>(&cache, &default_device())
            .unwrap();
        assert_eq!(*loaded.handle, vec![checkpoint, vocabulary]);

        assert_eq!(NeedsConfig.plan(&model, &cache).unwrap(), model.to_map());
    }

    /// A hook that asks for a part the map lacks gets `ResourceNotFound`
    /// naming the keys there are; a map with a part that is not on disk
    /// fails at the load, before the hook runs.
    #[test]
    fn test_a_missing_part_is_named() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.pt");
        fs::write(&checkpoint, b"x").unwrap();
        let cache = cache_in(dir.path());

        let err = Deferred::<NeedsConfig>::from_map(ResourceMap::given(
            "mine",
            "checkpoint",
            &checkpoint,
        ))
        .unwrap()
        .load::<CpuBackend>(&cache, &default_device())
        .unwrap_err();
        assert!(
            matches!(&err, BunsenError::ResourceNotFound(m) if m.contains("\"config\"") && m.contains("checkpoint")),
            "{err}"
        );

        let err = Deferred::<Paths>::from_map(ResourceMap::given(
            "mine",
            "checkpoint",
            dir.path().join("absent.pt"),
        ))
        .unwrap()
        .load::<CpuBackend>(&cache, &default_device())
        .unwrap_err();
        assert!(matches!(&err, BunsenError::ResourceNotFound(_)), "{err}");
    }
}
