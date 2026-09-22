//! # Deferred models
//!
//! A [`Deferred`] model is what a name resolved to, not yet loaded: the
//! resource map, a row's or one given from elsewhere, and the kit's hook
//! for it, chosen by what the map says about its resources, their `kind`s.
//! A factory hands one back from a name; a map from elsewhere, a checkpoint
//! on disk say, becomes one through [`Deferred::from_map`]. Loading needs
//! nothing more from the caller: different providers' rows may be read by
//! different mechanisms, and the hook attached is the one for this map.

use std::collections::BTreeMap;

use burn::prelude::Backend;

use super::{
    CacheStatus,
    Construct,
    Loaded,
    PretrainedCache,
    PretrainedRef,
    ResourceMap,
};
use crate::errors::BunsenResult;

/// A model not yet loaded: its map, and the hook that will build it.
#[derive(Clone, Debug)]
pub struct Deferred<H: Construct> {
    /// What resolved: a row of a provider, or a given map.
    pub model: PretrainedRef,

    /// The kit's hook for it, chosen by the map. Public, so that a caller
    /// doing surgery of its own, a config the model is built from say, can
    /// adjust what it was given; it never has to build one.
    pub hook: H,
}

impl<H: Construct> Deferred<H> {
    /// `model` with the hook its map calls for.
    ///
    /// # Errors
    /// As [`Construct::for_map`]: a resource of a kind the kit cannot read.
    pub fn new(model: PretrainedRef) -> BunsenResult<Self> {
        let hook = H::for_map(&model.to_map())?;
        Ok(Self { model, hook })
    }

    /// A map from elsewhere, a checkpoint on disk say, as a deferred model.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn from_map(map: ResourceMap) -> BunsenResult<Self> {
        Self::new(PretrainedRef::from(map))
    }

    /// The qualified id, or the given map's name.
    pub fn id(&self) -> String {
        self.model.id()
    }

    /// The map as it stands.
    pub fn to_map(&self) -> ResourceMap {
        self.model.to_map()
    }

    /// The caller's resources over the model's, by key: a `--vocab` over a
    /// row. The hook stays; it was chosen for the model's own resources.
    ///
    /// # Errors
    /// As [`PretrainedRef::with_overlay`].
    pub fn with_overlay(
        self,
        map: ResourceMap,
    ) -> BunsenResult<Self> {
        Ok(Self {
            model: self.model.with_overlay(map)?,
            hook: self.hook,
        })
    }

    /// Where every resource stands, by key, without touching bytes.
    pub fn status(
        &self,
        cache: &PretrainedCache,
    ) -> BTreeMap<String, CacheStatus> {
        cache.map_status(H::KIT, &self.to_map())
    }

    /// The hook's plan: the map completed, before anything but what the
    /// plan itself looks at is fetched.
    ///
    /// # Errors
    /// As [`Construct::plan`].
    pub fn plan(
        &self,
        cache: &PretrainedCache,
    ) -> BunsenResult<ResourceMap> {
        self.hook.plan(&self.model, cache)
    }

    /// Loads: [`Construct::plan`], [`PretrainedCache::load`],
    /// [`Construct::construct`]. The whole pathway once a model is in hand,
    /// whether it came from a factory, a path, or a manifest.
    ///
    /// # Errors
    /// As [`Construct::plan`], [`PretrainedCache::load`] and
    /// [`Construct::construct`].
    pub fn load<B: Backend>(
        &self,
        cache: &PretrainedCache,
        device: &B::Device,
    ) -> BunsenResult<Loaded<H::Built<B>>> {
        let planned = self.plan(cache)?;
        let resources = cache.load(H::KIT, &planned)?;
        let handle = self.hook.construct::<B>(&self.model, &resources, device)?;
        Ok(Loaded {
            name: resources.map.name.clone(),
            handle,
            resources,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{
        data::{
            cache::BunsenDiskCacheOptions,
            pretrained::{
                Pretrained,
                PretrainedCacheOptions,
                Provenance,
                Resource,
                Source,
                testing::CheckpointPath,
            },
        },
        errors::BunsenError,
        support::testing::{
            CpuBackend,
            default_device,
        },
    };

    fn offline_cache(dir: &std::path::Path) -> PretrainedCache {
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

    /// `well-known:a/small`, a row whose one resource is only remote.
    fn named() -> PretrainedRef {
        PretrainedRef::Named {
            provider: "well-known".to_string(),
            pretrained: Pretrained {
                name: "a/small".to_string(),
                aliases: Vec::new(),
                description: "small weights".to_string(),
                license: None,
                origin: None,
                prefab: Some("small".to_string()),
                resources: ResourceMap::new("small").with_resource(Resource {
                    key: "checkpoint".to_string(),
                    file: "small.pt".to_string(),
                    sha256: None,
                    kind: Some("pytorch".to_string()),
                    namespace: "a".to_string(),
                    sources: vec![Source::Url("https://a.example/small.pt".to_string())],
                }),
            },
        }
    }

    /// A given path is local already and loads in place through the hook
    /// its map called for; a name goes through the cache, which is offline
    /// here and has nothing local.
    #[test]
    fn test_status_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());
        let file = dir.path().join("ckpt.pt");
        fs::write(&file, b"x").unwrap();
        let spec = file.to_str().unwrap();

        let given =
            Deferred::<CheckpointPath>::from_map(ResourceMap::given(spec, "checkpoint", &file))
                .unwrap();
        assert_eq!(given.id(), spec);
        assert_eq!(given.status(&cache)["checkpoint"], CacheStatus::LocalDir);
        assert_eq!(given.plan(&cache).unwrap(), given.to_map());
        let loaded = given.load::<CpuBackend>(&cache, &default_device()).unwrap();
        assert_eq!(*loaded.handle, file);
        assert_eq!(loaded.name, spec);
        assert_eq!(
            loaded.resources.get("checkpoint").unwrap().provenance,
            Provenance::LocalDir
        );
        assert!(format!("{given:?}").contains("CheckpointPath"));

        let named = Deferred::<CheckpointPath>::new(named()).unwrap();
        assert_eq!(named.id(), "well-known:a/small");
        assert_eq!(named.status(&cache)["checkpoint"], CacheStatus::Remote);
        assert!(matches!(
            named.load::<CpuBackend>(&cache, &default_device()),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }

    /// An overlay replaces by key and keeps the hook the model was given.
    #[test]
    fn test_with_overlay_keeps_the_hook() {
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());
        let other = dir.path().join("other.pt");
        fs::write(&other, b"y").unwrap();

        let model = Deferred::<CheckpointPath>::new(named())
            .unwrap()
            .with_overlay(ResourceMap::given("mine", "checkpoint", &other))
            .unwrap();
        assert_eq!(model.id(), "well-known:a/small");
        assert_eq!(model.to_map().get("checkpoint").unwrap().file, "other.pt");
        let loaded = model.load::<CpuBackend>(&cache, &default_device()).unwrap();
        assert_eq!(*loaded.handle, other);
        assert_eq!(loaded.name, "well-known:a/small");
    }
}
