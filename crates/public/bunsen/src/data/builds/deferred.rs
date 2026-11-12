//! Deferred models: a resolved ref and its hook, not yet loaded.

use alloc::collections::BTreeMap;

use burn::tensor::Device;

use crate::{
    data::{
        builds::{
            ConstructInit,
            Loaded,
            PretrainedRef,
        },
        resources::{
            CacheStatus,
            ResourceCache,
            ResourceSet,
        },
    },
    errors::{
        BunsenResult,
        ResultContext,
    },
};

/// A model not yet loaded: its resource set, and the hook that will build
/// it.
///
/// What a name resolved to. [`PretrainedFactory::resolve`] hands one back
/// from a spec; a resource set from elsewhere, a checkpoint on disk say,
/// becomes one through [`from_resource_set`](Self::from_resource_set),
/// and any [`PretrainedRef`] through [`new`](Self::new). Either way the hook is
/// [`ConstructInit::for_resource_set`]'s choice, made from what the resource
/// set says about its resources (their `format`s): different providers' rows
/// may be read by different mechanisms, and the hook attached is the one for
/// this resource set.
///
/// Before loading, a caller may lay its own resources over the model's
/// ([`with_overlay`](Self::with_overlay)), ask where each resource stands
/// ([`status`](Self::status)), or adjust the public `hook`. Loading needs
/// nothing more: [`load`](Self::load) runs the hook's plan, the
/// [`ResourceCache`]'s load, and the hook's construct, and returns a
/// [`Loaded`] handle.
///
/// [`PretrainedFactory::resolve`]: super::PretrainedFactory::resolve
#[derive(Clone, Debug)]
pub struct Deferred<H: ConstructInit> {
    /// What resolved: a row of a provider, or a given resource set.
    pub model: PretrainedRef,

    /// The kit's hook for it, chosen by the resource set. Public, so that a
    /// caller doing surgery of its own, a config the model is built from say,
    /// can adjust what it was given; it never has to build one.
    pub config: H,
}

impl<H: ConstructInit> Deferred<H> {
    /// `model` with the hook its resource set calls for.
    ///
    /// # Errors
    /// As [`ConstructInit::for_resource_set`]: a resource of a kind the kit
    /// cannot read.
    pub fn new(model: PretrainedRef) -> BunsenResult<Self> {
        let config = H::for_resource_set(&model.to_resource_set())?;
        Ok(Self { model, config })
    }

    /// A resource set from elsewhere, a checkpoint on disk say, as a
    /// deferred model.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn from_resource_set(map: ResourceSet) -> BunsenResult<Self> {
        Self::new(PretrainedRef::from(map))
    }

    /// The qualified id, or the given resource set's name.
    pub fn id(&self) -> String {
        self.model.id()
    }

    /// The resource set as it stands.
    pub fn to_resource_set(&self) -> ResourceSet {
        self.model.to_resource_set()
    }

    /// The caller's resources over the model's, by key: a `--vocab` over a
    /// row. The hook stays; it was chosen for the model's own resources.
    ///
    /// # Errors
    /// As [`PretrainedRef::with_overlay`].
    pub fn with_overlay(
        self,
        map: ResourceSet,
    ) -> BunsenResult<Self> {
        Ok(Self {
            model: self.model.with_overlay(map)?,
            config: self.config,
        })
    }

    /// Where every resource stands, by key, without touching bytes.
    pub fn status(
        &self,
        cache: &ResourceCache,
    ) -> BTreeMap<String, CacheStatus> {
        cache.map_status(&self.to_resource_set())
    }

    /// The hook's plan: the resource set completed, before anything but what
    /// the plan itself looks at is fetched.
    ///
    /// # Errors
    /// As [`ConstructInit::plan`], under a frame naming the kit and the model.
    /// The model is the caller's input: an
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) error it causes
    /// is re-marked [`Policy`](crate::errors::BunsenErrorKind::Policy).
    pub fn plan(
        &self,
        cache: &ResourceCache,
    ) -> BunsenResult<ResourceSet> {
        self.config
            .plan(&self.model, cache)
            .with_context(|| format!("planning {} {:?}", H::KIT, self.id()))
            .as_policy()
    }

    /// Loads: [`ConstructInit::plan`], [`ResourceCache::load`],
    /// [`ConstructInit::try_init`] under `policy`. The whole pathway once a
    /// model is in hand, whether it came from a factory, a path, or a
    /// manifest.
    ///
    /// # Errors
    /// As [`ConstructInit::plan`], [`ResourceCache::load`] and
    /// [`ConstructInit::try_init`], under a frame naming the kit and the
    /// model. The model and its files are the caller's input: an
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) error they cause
    /// is re-marked [`Policy`](crate::errors::BunsenErrorKind::Policy).
    pub fn load(
        &self,
        cache: &ResourceCache,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Loaded<H::Construct>> {
        let load = || -> BunsenResult<Loaded<H::Construct>> {
            let planned = self.config.plan(&self.model, cache)?;
            let resources = cache.load(&planned)?;
            let handle = self
                .config
                .try_init(&self.model, &resources, device, policy)?;
            Ok(Loaded {
                name: resources.map.name.clone(),
                handle,
                resources,
            })
        };
        load()
            .with_context(|| format!("loading {} {:?}", H::KIT, self.id()))
            .as_policy()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{
        data::{
            builds::{
                Pretrained,
                testing::CheckpointPath,
            },
            cache::BunsenDiskCacheOptions,
            resources::{
                Provenance,
                Resource,
                ResourceCacheOptions,
                Source,
            },
        },
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
    };

    fn offline_cache(dir: &std::path::Path) -> ResourceCache {
        ResourceCache::new(
            ResourceCacheOptions::default()
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
            pretrained: Box::new(Pretrained {
                name: "a/small".to_string(),
                aliases: Vec::new(),
                description: "small weights".to_string(),
                license: None,
                origin: None,
                prefab: Some("small".to_string()),
                resources: ResourceSet::new("small").with_resource(Resource {
                    size: None,
                    dtype: None,
                    slot: "checkpoint".to_string(),
                    file: "small.pt".to_string(),
                    sha256: None,
                    format: Some("pytorch".to_string()),
                    name: "a".to_string(),
                    sources: vec![Source::Url("https://a.example/small.pt".to_string())],
                }),
                presets: Default::default(),
                args: Default::default(),
                tags: Default::default(),
            }),
            selection: Default::default(),
        }
    }

    /// A given path is local already and loads in place through the hook
    /// its resource set called for; a name goes through the cache, which is
    /// offline here and has nothing local.
    #[test]
    fn test_status_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());
        let file = dir.path().join("ckpt.pt");
        fs::write(&file, b"x").unwrap();
        let spec = file.to_str().unwrap();

        let given = Deferred::<CheckpointPath>::from_resource_set(ResourceSet::given(
            spec,
            "checkpoint",
            &file,
        ))
        .unwrap();
        assert_eq!(given.id(), spec);
        assert_eq!(given.status(&cache)["checkpoint"], CacheStatus::LocalDir);
        assert_eq!(given.plan(&cache).unwrap(), given.to_resource_set());
        let loaded = given
            .load(
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::as_stored(),
            )
            .unwrap();
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
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .frame_contains("loading kit \"well-known:a/small\"")
            .message_contains("the cache is offline")
            .assert_err(&named.load(
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::as_stored(),
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
            .with_overlay(ResourceSet::given("mine", "checkpoint", &other))
            .unwrap();
        assert_eq!(model.id(), "well-known:a/small");
        assert_eq!(
            model.to_resource_set().get("checkpoint").unwrap().file,
            "other.pt"
        );
        let loaded = model
            .load(
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::as_stored(),
            )
            .unwrap();
        assert_eq!(*loaded.handle, other);
        assert_eq!(loaded.name, "well-known:a/small");
    }
}
