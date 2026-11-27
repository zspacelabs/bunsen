//! Deferred builds: a resolved ref and its hook, not yet loaded.

use alloc::{
    collections::BTreeMap,
    format,
    sync::Arc,
};
use std::path::PathBuf;

use burn::tensor::{
    DType,
    Device,
    FloatDType,
};

use crate::{
    data::{
        builds::{
            BuildRef,
            ConstructInit,
            Loaded,
        },
        resources::{
            CacheStatus,
            LoadedResourceSet,
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
/// What a name resolved to. [`BuildFactory::resolve`] hands one back
/// from a spec; a resource set from elsewhere, a checkpoint on disk say,
/// becomes one through [`from_resource_set`](Self::from_resource_set),
/// and any [`BuildRef`] through [`new`](Self::new). Either way the hook is
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
/// [`BuildFactory::resolve`]: super::BuildFactory::resolve
#[derive(Clone, Debug)]
pub struct DeferredBuild<H: ConstructInit> {
    /// What resolved: a row of a provider, or a given resource set.
    pub model: BuildRef,

    /// The kit's hook for it, chosen by the resource set. Public, so that a
    /// caller doing surgery of its own, a config the model is built from say,
    /// can adjust what it was given; it never has to build one.
    pub config: H,
}

impl<H: ConstructInit> DeferredBuild<H> {
    /// `model` with the hook its resource set calls for.
    ///
    /// # Errors
    /// As [`ConstructInit::for_resource_set`]: a resource of a kind the kit
    /// cannot read.
    pub fn new(model: BuildRef) -> BunsenResult<Self> {
        let config = H::for_resource_set(&model.to_resource_set())?;
        Ok(Self { model, config })
    }

    /// A resource set from elsewhere, a checkpoint on disk say, as a
    /// deferred model.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn from_resource_set(map: ResourceSet) -> BunsenResult<Self> {
        Self::new(BuildRef::from(map))
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
    /// As [`BuildRef::with_overlay`].
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
            let converted = self.converted(cache, &planned, device, policy);
            let (resources, handle) = match &converted {
                Some((weights, path)) if path.is_file() => {
                    let rest = without(&planned, weights);
                    let resources = cache.load(&rest)?;
                    match self.config.try_init_converted(
                        &self.model,
                        &resources,
                        path,
                        device,
                        policy,
                    ) {
                        Ok(handle) => {
                            crate::data::resources::mark_converted_used(path);
                            log::info!("{}: weights from {}", H::KIT, path.display());
                            (resources, handle)
                        }
                        Err(e) => {
                            log::warn!(
                                "{}: the converted weights at {} did not load ({e}); loading the \
                                 weights' files",
                                H::KIT,
                                path.display()
                            );
                            self.load_and_convert(cache, &planned, device, policy, Some(path))?
                        }
                    }
                }
                _ => self.load_and_convert(
                    cache,
                    &planned,
                    device,
                    policy,
                    converted.as_ref().map(|(_, path)| path),
                )?,
            };
            let summary = self
                .model
                .summary()
                .with_applied(H::applied_presets(&handle));
            Ok(Loaded {
                name: resources.map.name.clone(),
                handle,
                resources,
                summary,
            })
        };
        load()
            .with_context(|| format!("loading {} {:?}", H::KIT, self.id()))
            .as_policy()
    }
}

impl<H: ConstructInit> DeferredBuild<H> {
    /// The weights a converted copy would stand in for, and where the copy
    /// is or would be, when the cache keeps converted weights, the kit keeps
    /// a copy, and the weights are pinned; `None` otherwise.
    fn converted(
        &self,
        cache: &ResourceCache,
        planned: &ResourceSet,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> Option<(ResourceSet, PathBuf)> {
        if !cache.converted_weights() {
            return None;
        }
        let weights = self.config.converted_weights(planned)?;
        let floats = [
            FloatDType::F16,
            FloatDType::BF16,
            FloatDType::F32,
            FloatDType::F64,
        ]
        .map(|d| (d, device.supports_dtype(DType::from(d))));
        let facets = format!("policy {policy:?}\ndevice floats {floats:?}");
        let key = cache.converted_key(H::KIT, &weights, &facets)?;
        let path = cache.converted_path(H::KIT, &weights, &key)?;
        Some((weights, path))
    }

    /// `planned` loaded and built the whole way, and, given a `converted`
    /// path, the weights as built written there for the next load: to a
    /// `.partial` beside it first, then renamed, so a copy is never half
    /// there. A copy that cannot be written is logged, not an error.
    fn load_and_convert(
        &self,
        cache: &ResourceCache,
        planned: &ResourceSet,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
        converted: Option<&PathBuf>,
    ) -> BunsenResult<(LoadedResourceSet, Arc<H::Construct>)> {
        let resources = cache.load(planned)?;
        let handle = self
            .config
            .try_init(&self.model, &resources, device, policy)?;
        if let Some(path) = converted {
            let partial = path.with_extension("bpk.partial");
            let written = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .map_err(crate::errors::sys_at("create", path))
                .and_then(|()| self.config.save_converted(&handle, &partial))
                .and_then(|()| {
                    std::fs::rename(&partial, path)
                        .map_err(crate::errors::sys_at("rename", &partial))
                });
            match written {
                Ok(()) => log::info!("{}: converted weights saved to {}", H::KIT, path.display()),
                Err(e) => {
                    let _ = std::fs::remove_file(&partial);
                    log::warn!("{}: converted weights not saved ({e})", H::KIT);
                }
            }
        }
        Ok((resources, handle))
    }
}

/// `planned` without `weights`' resources.
fn without(
    planned: &ResourceSet,
    weights: &ResourceSet,
) -> ResourceSet {
    let mut rest = planned.clone();
    rest.resources
        .retain(|key, _| !weights.resources.contains_key(key));
    rest
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{
        data::{
            builds::testing::CheckpointPath,
            cache::BunsenDiskCacheOptions,
            catalog::{
                CatalogName,
                ResolvedBuild,
                document::{
                    ResourceEntry,
                    ResourceSource,
                },
            },
            resources::{
                Provenance,
                ResourceCacheOptions,
            },
        },
        errors::testing::ErrorMatcher,
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

    /// `catalog:a/small`, a build whose one resource is only remote.
    fn named() -> BuildRef {
        let n = |name: &str| CatalogName::parse(name).unwrap();
        let build = ResolvedBuild::new(n("a/small"), "small weights").with_resource(
            "checkpoint",
            n("a/small/checkpoint"),
            ResourceEntry::new(
                "small.pt",
                "pytorch",
                vec![ResourceSource::Url(
                    "https://a.example/small.pt".to_string(),
                )],
            ),
        );
        BuildRef::named("catalog", build, Default::default())
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

        let given = DeferredBuild::<CheckpointPath>::from_resource_set(ResourceSet::given(
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
                crate::data::formats::LoadPolicy::load_as_stored(),
            )
            .unwrap();
        assert_eq!(*loaded.handle, file);
        assert_eq!(loaded.name, spec);
        assert_eq!(
            loaded.resources.get("checkpoint").unwrap().provenance,
            Provenance::LocalDir
        );
        assert!(format!("{given:?}").contains("CheckpointPath"));

        let named = DeferredBuild::<CheckpointPath>::new(named()).unwrap();
        assert_eq!(named.id(), "catalog:a/small");
        assert_eq!(named.status(&cache)["checkpoint"], CacheStatus::Remote);
        ErrorMatcher::policy()
            .frame_contains("loading kit \"catalog:a/small\"")
            .message_contains("the cache is offline")
            .assert_err(&named.load(
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::load_as_stored(),
            ));
    }

    /// A hook that keeps converted weights: its construct is the
    /// checkpoint's text, marked by where it was read from; its copy, the
    /// text as a file.
    #[derive(Clone, Debug, Default)]
    struct Converting;

    impl ConstructInit for Converting {
        type Construct = String;

        const KIT: &'static str = "toy";

        fn for_resource_set(_map: &ResourceSet) -> BunsenResult<Self> {
            Ok(Self)
        }

        fn try_init(
            &self,
            _model: &BuildRef,
            loaded: &LoadedResourceSet,
            _device: &Device,
            _policy: crate::data::formats::LoadPolicy,
        ) -> BunsenResult<Arc<String>> {
            let text = fs::read_to_string(loaded.expect("checkpoint")?).unwrap();
            Ok(Arc::new(format!("files: {text}")))
        }

        fn converted_weights(
            &self,
            map: &ResourceSet,
        ) -> Option<ResourceSet> {
            Some(map.family("checkpoint"))
        }

        fn try_init_converted(
            &self,
            _model: &BuildRef,
            loaded: &LoadedResourceSet,
            path: &std::path::Path,
            _device: &Device,
            _policy: crate::data::formats::LoadPolicy,
        ) -> BunsenResult<Arc<String>> {
            assert!(
                loaded.get("checkpoint").is_none(),
                "the weights' files are not loaded"
            );
            let text = fs::read_to_string(path).unwrap();
            Ok(Arc::new(format!("converted: {text}")))
        }

        fn save_converted(
            &self,
            construct: &String,
            path: &std::path::Path,
        ) -> BunsenResult<()> {
            let text = construct.strip_prefix("files: ").unwrap();
            fs::write(path, text).unwrap();
            Ok(())
        }
    }

    /// `catalog:a/pinned`: one pinned, remote-only resource holding `text`,
    /// with its file in `cache` already.
    fn pinned_in(
        cache: &ResourceCache,
        text: &str,
    ) -> BuildRef {
        use sha2::Digest;
        let pin: String = sha2::Sha256::digest(text.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let n = |name: &str| CatalogName::parse(name).unwrap();
        let mut entry = ResourceEntry::new(
            "w.bin",
            "toy",
            vec![ResourceSource::Url("https://a.example/w.bin".to_string())],
        );
        entry.sha256 = Some(pin);
        let build = ResolvedBuild::new(n("a/pinned"), "pinned weights").with_resource(
            "checkpoint",
            n("a/pinned/checkpoint"),
            entry,
        );
        let model = BuildRef::named("catalog", build, Default::default());
        let resource = model.to_resource_set().get("checkpoint").unwrap().clone();
        let path = cache.resource_path(&resource).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, text).unwrap();
        model
    }

    /// A first load reads the weights' files and saves the kit's copy; the
    /// next reads the copy; with the files cleared, an offline load still
    /// builds, from the copy. Off, the cache keeps no copy.
    #[test]
    fn test_converted_weights() {
        let dir = tempfile::tempdir().unwrap();
        let policy = crate::data::formats::LoadPolicy::load_as_stored();
        let device = cpu_device();
        let off = offline_cache(dir.path());
        let on = ResourceCache::new(
            ResourceCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true)
                .with_converted_weights(true),
        )
        .unwrap();
        let model = DeferredBuild::<Converting>::new(pinned_in(&on, "w1")).unwrap();

        let load = |cache: &ResourceCache| {
            model
                .load(cache, &device, policy)
                .map(|l| (*l.handle).clone())
        };
        assert_eq!(load(&off).unwrap(), "files: w1");
        let listed = |cache: &ResourceCache| {
            cache
                .cached_files(&"a/pinned".parse().unwrap())
                .unwrap()
                .into_iter()
                .map(|f| f.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(listed(&on), ["a/pinned/checkpoint"], "off keeps no copy");

        assert_eq!(load(&on).unwrap(), "files: w1");
        assert_eq!(listed(&on), ["a/pinned/checkpoint", "a/pinned/converted"]);
        assert_eq!(load(&on).unwrap(), "converted: w1");

        let sources = on
            .cached_files(&"a/pinned/checkpoint".parse().unwrap())
            .unwrap();
        on.remove_cached(&sources).unwrap();
        assert_eq!(listed(&on), ["a/pinned/converted"]);
        assert_eq!(
            load(&on).unwrap(),
            "converted: w1",
            "no fetch: the cache is offline"
        );
        ErrorMatcher::policy()
            .message_contains("the cache is offline")
            .assert_err(&load(&off));
    }

    /// An overlay replaces by key and keeps the hook the model was given.
    #[test]
    fn test_with_overlay_keeps_the_hook() {
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());
        let other = dir.path().join("other.pt");
        fs::write(&other, b"y").unwrap();

        let model = DeferredBuild::<CheckpointPath>::new(named())
            .unwrap()
            .with_overlay(ResourceSet::given("mine", "checkpoint", &other))
            .unwrap();
        assert_eq!(model.id(), "catalog:a/small");
        assert_eq!(
            model.to_resource_set().get("checkpoint").unwrap().file,
            "other.pt"
        );
        let loaded = model
            .load(
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::load_as_stored(),
            )
            .unwrap();
        assert_eq!(*loaded.handle, other);
        assert_eq!(loaded.name, "catalog:a/small");
    }
}
