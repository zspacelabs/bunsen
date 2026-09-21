//! # Construction
//!
//! From a loaded map to what a kit builds. A [`Construct`] hook is the
//! kit's: it says which cache segment its files live under, what it builds
//! for a backend, how a map is completed before it is loaded, and how the
//! loaded parts become the built thing, behind an `Arc`. [`load_map`] runs
//! the whole way: plan, load, construct. Nothing here opens a file.

use std::sync::Arc;

use burn::prelude::Backend;

use super::{
    LoadedResources,
    PretrainedCache,
    ResourceMap,
};
use crate::errors::BunsenResult;

/// What a kit builds from a loaded map, and how.
///
/// Not object-safe: [`construct`](Self::construct) is generic over the
/// backend so that [`Built`](Self::Built) is a real type. A kit's loading
/// function supplies its hook as a value, which carries the hook's options:
/// a scanner, a key remapping, a geometry to expect.
pub trait Construct {
    /// The cache segment the kit's files live under: the `<kit>` of
    /// `pretrained/<kit>/<namespace>/<sha256>/<file>`.
    const KIT: &'static str;

    /// What construction yields, for a backend.
    type Built<B: Backend>;

    /// Completes a map before it is loaded: a kit rule that names a
    /// resource from another, or checks a declared one against the rule.
    /// The default is identity. It may resolve a resource through the
    /// cache to look at it; a scan that reads shapes only is the intended
    /// use.
    ///
    /// # Errors
    /// The kit's: a map the rule cannot complete, or one that contradicts
    /// it.
    fn plan(
        &self,
        map: ResourceMap,
        _cache: &PretrainedCache,
    ) -> BunsenResult<ResourceMap> {
        Ok(map)
    }

    /// Builds from every part local. Never fetches.
    ///
    /// # Errors
    /// The kit's: a part the map lacks
    /// ([`LoadedResources::expect`]), or one that does not read as what its
    /// key says.
    fn construct<B: Backend>(
        &self,
        loaded: &LoadedResources,
        device: &B::Device,
    ) -> BunsenResult<Arc<Self::Built<B>>>;
}

/// The handle a pretrained hands back: bound by what was built.
///
/// Cloning shares the handle; the built thing is constructed once.
#[derive(Debug)]
pub struct Loaded<T> {
    /// The map's name: the pretrained's, or the caller's for a given map.
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

/// A map from anywhere, through a hook: plan, load, construct.
///
/// This is the whole pathway once a map is in hand, whether it came from a
/// row, a path, or a manifest.
///
/// # Errors
/// As [`Construct::plan`], [`PretrainedCache::load`] and
/// [`Construct::construct`].
pub fn load_map<B: Backend, H: Construct>(
    map: ResourceMap,
    cache: &PretrainedCache,
    hook: &H,
    device: &B::Device,
) -> BunsenResult<Loaded<H::Built<B>>> {
    let planned = hook.plan(map, cache)?;
    let resources = cache.load(H::KIT, &planned)?;
    let handle = hook.construct::<B>(&resources, device)?;
    Ok(Loaded {
        name: resources.map.name.clone(),
        handle,
        resources,
    })
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

        fn plan(
            &self,
            map: ResourceMap,
            _cache: &PretrainedCache,
        ) -> BunsenResult<ResourceMap> {
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
            loaded: &LoadedResources,
            _device: &B::Device,
        ) -> BunsenResult<Arc<Vec<PathBuf>>> {
            loaded.expect("checkpoint")?;
            Ok(Arc::new(
                loaded.iter().map(|(_, r)| r.path.clone()).collect(),
            ))
        }
    }

    /// A hook that asks for a key no map of these tests has.
    struct NeedsConfig;

    impl Construct for NeedsConfig {
        type Built<B: Backend> = ();

        const KIT: &'static str = "paths";

        fn construct<B: Backend>(
            &self,
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
    fn test_load_map_plans_loads_and_constructs() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.pt");
        let vocabulary = dir.path().join("gpt2.tiktoken");
        fs::write(&checkpoint, b"x").unwrap();
        fs::write(&vocabulary, b"y").unwrap();
        let cache = cache_in(dir.path());
        let hook = Paths {
            vocabulary: Some(vocabulary.clone()),
        };

        let loaded = load_map::<CpuBackend, _>(
            ResourceMap::given("mine", "checkpoint", &checkpoint),
            &cache,
            &hook,
            &default_device(),
        )
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

    /// With nothing to derive, the default plan is identity, and a map
    /// that already has the key is left alone.
    #[test]
    fn test_plan_leaves_a_complete_map_alone() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.pt");
        let vocabulary = dir.path().join("mine.tiktoken");
        fs::write(&checkpoint, b"x").unwrap();
        fs::write(&vocabulary, b"y").unwrap();
        let cache = cache_in(dir.path());
        let map = ResourceMap::given("mine", "checkpoint", &checkpoint)
            .fuse(
                ResourceMap::given("flags", "vocabulary", &vocabulary),
                Fuse::Strict,
            )
            .unwrap();

        let hook = Paths {
            vocabulary: Some(dir.path().join("never-read.tiktoken")),
        };
        let loaded =
            load_map::<CpuBackend, _>(map.clone(), &cache, &hook, &default_device()).unwrap();
        assert_eq!(*loaded.handle, vec![checkpoint.clone(), vocabulary.clone()]);

        let no_rule = Paths { vocabulary: None };
        let loaded = load_map::<CpuBackend, _>(map, &cache, &no_rule, &default_device()).unwrap();
        assert_eq!(*loaded.handle, vec![checkpoint, vocabulary]);
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

        let err = load_map::<CpuBackend, _>(
            ResourceMap::given("mine", "checkpoint", &checkpoint),
            &cache,
            &NeedsConfig,
            &default_device(),
        )
        .unwrap_err();
        assert!(
            matches!(&err, BunsenError::ResourceNotFound(m) if m.contains("\"config\"") && m.contains("checkpoint")),
            "{err}"
        );

        let err = load_map::<CpuBackend, _>(
            ResourceMap::given("mine", "checkpoint", dir.path().join("absent.pt")),
            &cache,
            &Paths { vocabulary: None },
            &default_device(),
        )
        .unwrap_err();
        assert!(matches!(&err, BunsenError::ResourceNotFound(_)), "{err}");
    }
}
