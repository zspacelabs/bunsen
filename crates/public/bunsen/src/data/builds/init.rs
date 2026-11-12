use alloc::sync::Arc;

use burn::tensor::Device;

use crate::{
    data::{
        builds::BuildRef,
        resources::{
            LoadedResourceSet,
            ResourceCache,
            ResourceSet,
        },
    },
    errors::BunsenResult,
};

/// What a kit builds from a resolved ref, and how: the kit's construct
/// config, its hook into the pretrained pathway.
///
/// Each kit's `{Kit}ConstructConfig` implements it, and builds the kit's
/// construct. It says what the kit is called ([`KIT`](Self::KIT)), what it
/// builds for a backend ([`Construct`](Self::Construct)), which hook a resource
/// set calls for ([`for_resource_set`](Self::for_resource_set)), how a
/// ref's resource set is completed before it is loaded
/// ([`plan`](Self::plan)), and how the loaded parts become the built thing,
/// behind an `Arc` ([`try_init`](Self::try_init)). A
/// [`DeferredBuild`](super::DeferredBuild) model pairs a ref with its construct
/// config and runs the whole way: plan, [`ResourceCache::load`], `try_init`.
///
/// Not object-safe: [`try_init`](Self::try_init) is generic over the
/// backend so that [`Construct`](Self::Construct) is a real type. A
/// construct config is a value, chosen for a resource set by
/// [`for_resource_set`](Self::for_resource_set) and carried by the
/// deferred model it builds: different providers' rows may be read by different
/// mechanisms, and the resource set's resources, their `format`s, say which. A
/// caller never builds one; it may adjust the one it was given.
///
/// Both steps see the ref, not just its resource set: what a row promises
/// (its prefab) is checked in [`plan`](Self::plan) before bytes are fetched,
/// and a kit whose checkpoint does not describe itself takes its config
/// from the build in [`try_init`](Self::try_init).
pub trait ConstructInit: Sized {
    /// The kit's name: what its factory, its errors and its listings call it.
    const KIT: &'static str;

    /// Whether [`try_init`](Self::try_init) resolves the ref's preset
    /// selection ([`BuildRef::selection`]). A factory refuses a lookup
    /// that selects presets or patches the config of a kit whose hook does
    /// not, rather than let the request be ignored.
    const RESOLVES_PRESETS: bool = false;

    /// What construction yields, for a backend.
    type Construct;

    /// The construct config for `map`: the reader its resources' `format`s call
    /// for.
    ///
    /// # Errors
    /// The kit's: a resource of a kind it has no reader for, refused by
    /// name rather than misread.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self>;

    /// Completes the ref's resource set before it is loaded: a kit rule that
    /// names a resource from another, or checks a declared one against the
    /// rule, or checks the checkpoint against what the row promised. The
    /// default is the resource set as it stands. It may resolve a resource
    /// through the cache to look at it; a scan that reads shapes only is the
    /// intended use.
    ///
    /// # Errors
    /// The kit's: a resource set the rule cannot complete, or one that
    /// contradicts it.
    fn plan(
        &self,
        model: &BuildRef,
        _cache: &ResourceCache,
    ) -> BunsenResult<ResourceSet> {
        Ok(model.to_resource_set())
    }

    /// Builds from every part local. Never fetches. `model` is the ref
    /// [`plan`](Self::plan) saw; `policy` is the call's dtype policy, by
    /// which the hook places its float parameters
    /// ([`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve)).
    ///
    /// # Errors
    /// The kit's: a part the resource set lacks
    /// ([`LoadedResourceSet::expect`]), or one that does not read as what its
    /// key says.
    fn try_init(
        &self,
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<Self::Construct>>;
}

/// The handle a pretrained hands back: bound by what was built.
///
/// The end of the pathway: what
/// [`DeferredBuild::load`](super::DeferredBuild::load)
/// and [`BuildFactory::load`](super::BuildFactory::load) return.
/// Cloning shares the handle; the built thing is constructed once.
#[derive(Debug)]
pub struct Loaded<T> {
    /// The resource set's name: the pretrained's id, or the caller's for a
    /// given resource set.
    pub name: String,

    /// What was built, shared.
    pub handle: Arc<T>,

    /// Where every part came from.
    pub resources: LoadedResourceSet,
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
            builds::DeferredBuild,
            cache::BunsenDiskCacheOptions,
            resources::{
                Fuse,
                Provenance,
                ResourceCacheOptions,
            },
        },
        errors::{
            BunsenErrorKind,
            LookupError,
            LookupProblem,
            testing::{
                ErrorMatcher,
                predicate,
            },
        },
        support::testing::cpu_device,
    };

    /// A hook that builds the list of its parts' paths, in key order, and
    /// completes a resource set with a vocabulary when one is missing.
    struct Paths {
        vocabulary: Option<PathBuf>,
    }

    impl ConstructInit for Paths {
        type Construct = Vec<PathBuf>;

        const KIT: &'static str = "paths";

        fn for_resource_set(_map: &ResourceSet) -> BunsenResult<Self> {
            Ok(Paths { vocabulary: None })
        }

        fn plan(
            &self,
            model: &BuildRef,
            _cache: &ResourceCache,
        ) -> BunsenResult<ResourceSet> {
            let map = model.to_resource_set();
            match &self.vocabulary {
                Some(path) if map.get("vocabulary").is_none() => map.fuse(
                    ResourceSet::given("derived", "vocabulary", path),
                    Fuse::Strict,
                ),
                _ => Ok(map),
            }
        }

        fn try_init(
            &self,
            _model: &BuildRef,
            loaded: &LoadedResourceSet,
            _device: &Device,
            _policy: crate::data::formats::LoadPolicy,
        ) -> BunsenResult<Arc<Vec<PathBuf>>> {
            loaded.expect("checkpoint")?;
            Ok(Arc::new(
                loaded.iter().map(|(_, r)| r.path.clone()).collect(),
            ))
        }
    }

    /// A hook that asks for a key no resource set of these tests has, and
    /// takes the default plan.
    struct NeedsConfig;

    impl ConstructInit for NeedsConfig {
        type Construct = ();

        const KIT: &'static str = "paths";

        fn for_resource_set(_map: &ResourceSet) -> BunsenResult<Self> {
            Ok(NeedsConfig)
        }

        fn try_init(
            &self,
            _model: &BuildRef,
            loaded: &LoadedResourceSet,
            _device: &Device,
            _policy: crate::data::formats::LoadPolicy,
        ) -> BunsenResult<Arc<()>> {
            loaded.expect("config")?;
            Ok(Arc::new(()))
        }
    }

    fn cache_in(dir: &std::path::Path) -> ResourceCache {
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

    /// Plan completes the resource set, load brings every part local,
    /// construct builds; the handle is shared by clones and the resources say
    /// where each part came from.
    #[test]
    fn test_load_plans_loads_and_constructs() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.pt");
        let vocabulary = dir.path().join("gpt2.tiktoken");
        fs::write(&checkpoint, b"x").unwrap();
        fs::write(&vocabulary, b"y").unwrap();
        let cache = cache_in(dir.path());
        let config = Paths {
            vocabulary: Some(vocabulary.clone()),
        };

        let loaded = DeferredBuild {
            model: BuildRef::from(ResourceSet::given("mine", "checkpoint", &checkpoint)),
            config,
        }
        .load(
            &cache,
            &cpu_device(),
            crate::data::formats::LoadPolicy::load_as_stored(),
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

    /// With nothing to derive, the default plan is the ref's resource set,
    /// and a resource set that already has the key is left alone.
    #[test]
    fn test_plan_leaves_a_complete_map_alone() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.pt");
        let vocabulary = dir.path().join("mine.tiktoken");
        fs::write(&checkpoint, b"x").unwrap();
        fs::write(&vocabulary, b"y").unwrap();
        let cache = cache_in(dir.path());
        let model = BuildRef::from(
            ResourceSet::given("mine", "checkpoint", &checkpoint)
                .fuse(
                    ResourceSet::given("flags", "vocabulary", &vocabulary),
                    Fuse::Strict,
                )
                .unwrap(),
        );

        let config = Paths {
            vocabulary: Some(dir.path().join("never-read.tiktoken")),
        };
        let loaded = DeferredBuild {
            model: model.clone(),
            config,
        }
        .load(
            &cache,
            &cpu_device(),
            crate::data::formats::LoadPolicy::load_as_stored(),
        )
        .unwrap();
        assert_eq!(*loaded.handle, vec![checkpoint.clone(), vocabulary.clone()]);

        // The hook a resource set calls for, with no rule of its own.
        let no_rule = DeferredBuild::<Paths>::new(model.clone()).unwrap();
        assert!(no_rule.config.vocabulary.is_none());
        let loaded = no_rule
            .load(
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::load_as_stored(),
            )
            .unwrap();
        assert_eq!(*loaded.handle, vec![checkpoint, vocabulary]);

        assert_eq!(
            NeedsConfig.plan(&model, &cache).unwrap(),
            model.to_resource_set()
        );
    }

    /// A hook that asks for a part the resource set lacks gets a lookup
    /// error naming the keys there are; a resource set with a part that is
    /// not on disk fails at the load, before the hook runs, naming the path.
    #[test]
    fn test_a_missing_part_is_named() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.pt");
        fs::write(&checkpoint, b"x").unwrap();
        let cache = cache_in(dir.path());

        let err = DeferredBuild::<NeedsConfig>::from_resource_set(ResourceSet::given(
            "mine",
            "checkpoint",
            &checkpoint,
        ))
        .unwrap()
        .load(
            &cache,
            &cpu_device(),
            crate::data::formats::LoadPolicy::load_as_stored(),
        )
        .unwrap_err();
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .frame_contains("loading paths \"mine\"")
            .cause(predicate("a missing \"config\"", |c: &LookupError| {
                c.key == "config" && c.candidates == ["checkpoint"]
            }))
            .assert(&err);

        let absent = dir.path().join("absent.pt");
        let err = DeferredBuild::<Paths>::from_resource_set(ResourceSet::given(
            "mine",
            "checkpoint",
            &absent,
        ))
        .unwrap()
        .load(
            &cache,
            &cpu_device(),
            crate::data::formats::LoadPolicy::load_as_stored(),
        )
        .unwrap_err();
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .cause(predicate("the absent path", move |c: &LookupError| {
                c.problem == LookupProblem::Missing && c.key == absent.display().to_string()
            }))
            .assert(&err);
    }
}
