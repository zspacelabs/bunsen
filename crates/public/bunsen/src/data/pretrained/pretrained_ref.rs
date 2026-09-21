//! # Pretrained references
//!
//! What a spec resolved to: a row in a provider, or a map from somewhere
//! else, such as a path. The name-to-model pathway's first step, shared by
//! every kit; the kit supplies the providers, the key a bare path fills,
//! the hook that builds, and any prefab check.

use std::{
    collections::BTreeMap,
    fmt::Debug,
    path::Path,
};

use burn::{
    config::Config,
    prelude::Backend,
};

use super::{
    CacheStatus,
    Construct,
    Loaded,
    PreFabConfig,
    PretrainedCache,
    ResourceMap,
    StaticPreFabMap,
    StaticPretrained,
    StaticPretrainedProvider,
    available_ids,
    load_map,
    lookup_pretrained,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// What a spec resolved to.
#[derive(Debug, Clone)]
pub enum PretrainedRef {
    /// A row in a provider.
    Named {
        /// Its provider.
        provider: &'static StaticPretrainedProvider<'static>,
        /// The row.
        pretrained: &'static StaticPretrained<'static>,
    },

    /// A map from somewhere else: a path, a manifest. No prefab is
    /// promised; what it is, the kit finds out by looking.
    Given(ResourceMap),
}

impl PretrainedRef {
    /// Resolves a spec against `providers`.
    ///
    /// `provider/name` looks up that provider; a bare name looks across all
    /// of them; an alias is honoured. Failing those, a path to an existing
    /// file is taken as a one-resource map under `path_key`, the key the
    /// kit reads a bare checkpoint by. The index wins over the file system.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`], naming what is available.
    pub fn resolve(
        providers: &[&'static StaticPretrainedProvider<'static>],
        spec: &str,
        path_key: &str,
    ) -> BunsenResult<Self> {
        let (provider, name) = match spec.rsplit_once('/') {
            Some((provider, name)) => (Some(provider), name),
            None => (None, spec),
        };
        if let Some((provider, pretrained)) = lookup_pretrained(providers, provider, name) {
            return Ok(Self::Named {
                provider,
                pretrained,
            });
        }

        let path = Path::new(spec);
        if path.is_file() {
            return Ok(Self::Given(ResourceMap::given(spec, path_key, path)));
        }

        Err(BunsenError::ResourceNotFound(format!(
            "no model {spec:?}: not a pretrained name and not a file; there are: {}",
            available_ids(providers).join(", ")
        )))
    }

    /// The qualified id, or the given map's name.
    pub fn id(&self) -> String {
        match self {
            Self::Named {
                provider,
                pretrained,
            } => provider.id(pretrained),
            Self::Given(map) => map.name.clone(),
        }
    }

    /// The provider and row, if the spec was a name.
    pub fn named(
        &self
    ) -> Option<(
        &'static StaticPretrainedProvider<'static>,
        &'static StaticPretrained<'static>,
    )> {
        match self {
            Self::Named {
                provider,
                pretrained,
            } => Some((provider, pretrained)),
            Self::Given(_) => None,
        }
    }

    /// The prefab the row promised, from `prefabs`, if the spec was a name
    /// and the row names one.
    ///
    /// # Panics
    /// If the row names a prefab `prefabs` does not have; a kit's tests
    /// pin that every row's does.
    pub fn prefab<C>(
        &self,
        prefabs: &StaticPreFabMap<C>,
    ) -> Option<PreFabConfig<C>>
    where
        C: 'static + Config + Debug + Clone,
    {
        self.named()
            .and_then(|(_, pretrained)| pretrained.prefab)
            .map(|prefab| prefabs.expect_lookup_prefab(prefab))
    }

    /// The resource map: a row's maps fused, named by the qualified id, or
    /// the given map.
    pub fn to_map(&self) -> ResourceMap {
        match self {
            Self::Named { pretrained, .. } => {
                let mut map = pretrained.to_map();
                map.name = self.id();
                map
            }
            Self::Given(map) => map.clone(),
        }
    }

    /// Where every resource stands, by key, without touching bytes.
    pub fn status(
        &self,
        kit: &str,
        cache: &PretrainedCache,
    ) -> BTreeMap<String, CacheStatus> {
        cache.map_status(kit, &self.to_map())
    }

    /// Loads through `hook`: plan, load, construct.
    ///
    /// # Errors
    /// As [`load_map`].
    pub fn load<B: Backend, H: Construct>(
        &self,
        cache: &PretrainedCache,
        hook: &H,
        device: &B::Device,
    ) -> BunsenResult<Loaded<H::Built<B>>> {
        load_map::<B, H>(self.to_map(), cache, hook, device)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::Arc,
    };

    use super::*;
    use crate::{
        data::{
            cache::BunsenDiskCacheOptions,
            pretrained::{
                LoadedResources,
                PretrainedCacheOptions,
                Provenance,
                StaticBase,
                StaticPreFabConfig,
                StaticResource,
                StaticResourceMap,
            },
        },
        support::testing::{
            CpuBackend,
            default_device,
        },
    };

    #[derive(Config, Debug)]
    struct Shape {
        width: usize,
    }

    static SMALL_MAP: StaticResourceMap<'static> = StaticResourceMap {
        name: "a/small.pt",
        description: "small weights",
        license: None,
        origin: None,
        namespace: "a",
        bases: &[StaticBase::Url("https://a.example")],
        resources: &[StaticResource {
            key: "checkpoint",
            file: "small.pt",
            sha256: None,
            kind: Some("pytorch"),
            sources: &[],
        }],
    };
    static SMALL: StaticPretrained<'static> = StaticPretrained {
        name: "small",
        aliases: &["s"],
        description: "small weights",
        license: None,
        origin: None,
        prefab: Some("small"),
        maps: &[&SMALL_MAP],
    };
    static A: StaticPretrainedProvider<'static> = StaticPretrainedProvider {
        name: "a",
        description: "provider a",
        license: None,
        origin: None,
        items: &[&SMALL],
    };
    static PROVIDERS: &[&StaticPretrainedProvider<'static>] = &[&A];
    static PREFABS: StaticPreFabMap<Shape> = StaticPreFabMap {
        name: "shapes",
        description: "test shapes",
        items: &[&StaticPreFabConfig {
            name: "small",
            description: "a small shape",
            builder: || Shape { width: 4 },
            weights: None,
        }],
    };

    /// A hook that builds the checkpoint's path.
    struct CheckpointPath;

    impl Construct for CheckpointPath {
        type Built<B: Backend> = PathBuf;

        const KIT: &'static str = "kit";

        fn construct<B: Backend>(
            &self,
            loaded: &LoadedResources,
            _device: &B::Device,
        ) -> BunsenResult<Arc<PathBuf>> {
            Ok(Arc::new(loaded.expect("checkpoint")?.to_path_buf()))
        }
    }

    fn offline_cache(dir: &Path) -> PretrainedCache {
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

    #[test]
    fn test_resolve_names_aliases_and_paths() {
        match PretrainedRef::resolve(PROVIDERS, "a/small", "checkpoint").unwrap() {
            PretrainedRef::Named {
                provider,
                pretrained,
            } => {
                assert_eq!((provider.name, pretrained.name), ("a", "small"));
            }
            other => panic!("{other:?}"),
        }
        let by_alias = PretrainedRef::resolve(PROVIDERS, "s", "checkpoint").unwrap();
        assert_eq!(by_alias.id(), "a/small");
        assert_eq!(
            by_alias.named().map(|(p, d)| (p.name, d.name)),
            Some(("a", "small"))
        );
        assert_eq!(
            by_alias.prefab(&PREFABS).map(|p| p.to_config().width),
            Some(4)
        );
        let map = by_alias.to_map();
        assert_eq!(map.name, "a/small", "a row's map is named by its id");
        assert_eq!(map.keys(), ["checkpoint"]);

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("ckpt.pt");
        fs::write(&file, b"x").unwrap();
        let spec = file.to_str().unwrap();
        let by_path = PretrainedRef::resolve(PROVIDERS, spec, "checkpoint").unwrap();
        assert_eq!(by_path.id(), spec);
        assert!(by_path.named().is_none());
        assert!(by_path.prefab(&PREFABS).is_none());
        let map = by_path.to_map();
        assert_eq!(map.keys(), ["checkpoint"]);
        assert_eq!(map.get("checkpoint").unwrap().file, "ckpt.pt");

        match PretrainedRef::resolve(PROVIDERS, "a/gigantic", "checkpoint") {
            Err(BunsenError::ResourceNotFound(m)) => assert!(m.contains("a/small"), "{m}"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            PretrainedRef::resolve(PROVIDERS, "/no/such/file.pt", "checkpoint"),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }

    /// A given path is local already and loads in place; a name goes
    /// through the cache, which is offline here and has nothing local.
    #[test]
    fn test_status_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());
        let file = dir.path().join("ckpt.pt");
        fs::write(&file, b"x").unwrap();

        let given =
            PretrainedRef::resolve(PROVIDERS, file.to_str().unwrap(), "checkpoint").unwrap();
        assert_eq!(
            given.status("kit", &cache)["checkpoint"],
            CacheStatus::LocalDir
        );
        let loaded = given
            .load::<CpuBackend, _>(&cache, &CheckpointPath, &default_device())
            .unwrap();
        assert_eq!(*loaded.handle, file);
        assert_eq!(loaded.name, file.to_str().unwrap());
        assert_eq!(
            loaded.resources.get("checkpoint").unwrap().provenance,
            Provenance::LocalDir
        );

        let named = PretrainedRef::resolve(PROVIDERS, "a/small", "checkpoint").unwrap();
        assert_eq!(
            named.status("kit", &cache)["checkpoint"],
            CacheStatus::Remote
        );
        assert!(matches!(
            named.load::<CpuBackend, _>(&cache, &CheckpointPath, &default_device()),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }
}
