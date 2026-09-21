//! # Pretrained references
//!
//! What a spec resolved to: a row copied out of a provider, or a map from
//! somewhere else, such as a path. The name-to-model pathway's first step,
//! shared by every kit; a [`PretrainedFactory`](super::PretrainedFactory)
//! makes one from a spec, the kit's hook plans and builds from it.
//!
//! A ref is plain data: both halves are serde types, and the provider is
//! held by name, so a ref outlives the factory that made it. The caller's
//! resources ride on it through [`with_overlay`](PretrainedRef::with_overlay),
//! so a hook sees one thing.

use core::fmt::Debug;

use burn::config::Config;

use super::{
    Fuse,
    PreFabConfig,
    Pretrained,
    ResourceMap,
    StaticPreFabMap,
};
use crate::errors::BunsenResult;

/// What a spec resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PretrainedRef {
    /// A row in a provider, copied out of it.
    Named {
        /// The provider's name: the `provider` of `provider:ref`.
        provider: String,
        /// The row; its `name` is the ref.
        pretrained: Pretrained,
    },

    /// A map from somewhere else: a path, a manifest. No prefab is
    /// promised; what it is, the kit finds out by looking.
    Given(ResourceMap),
}

impl From<ResourceMap> for PretrainedRef {
    fn from(map: ResourceMap) -> Self {
        Self::Given(map)
    }
}

impl PretrainedRef {
    /// The qualified id, `provider:ref`, or the given map's name.
    pub fn id(&self) -> String {
        match self {
            Self::Named {
                provider,
                pretrained,
            } => alloc::format!("{provider}:{}", pretrained.name),
            Self::Given(map) => map.name.clone(),
        }
    }

    /// The provider's name and the row, if the spec was a name.
    pub fn named(&self) -> Option<(&str, &Pretrained)> {
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
            .and_then(|(_, pretrained)| pretrained.prefab.as_deref())
            .map(|prefab| prefabs.expect_lookup_prefab(prefab))
    }

    /// The resource map: the row's, named by the qualified id, or the
    /// given map.
    pub fn to_map(&self) -> ResourceMap {
        match self {
            Self::Named { pretrained, .. } => {
                let mut map = pretrained.resources.clone();
                map.name = self.id();
                map
            }
            Self::Given(map) => map.clone(),
        }
    }

    /// The caller's resources over the model's, by key
    /// ([`Fuse::Overlay`]): a `--vocab` over a row. The id is unchanged;
    /// only the copy of the row is.
    ///
    /// # Errors
    /// As [`ResourceMap::fuse`], which does not fail under an overlay.
    pub fn with_overlay(
        self,
        map: ResourceMap,
    ) -> BunsenResult<Self> {
        Ok(match self {
            Self::Named {
                provider,
                mut pretrained,
            } => {
                pretrained.resources = pretrained.resources.fuse(map, Fuse::Overlay)?;
                Self::Named {
                    provider,
                    pretrained,
                }
            }
            Self::Given(mine) => Self::Given(mine.fuse(map, Fuse::Overlay)?),
        })
    }
}

#[cfg(feature = "cache")]
mod with_cache {
    use std::collections::BTreeMap;

    use burn::prelude::Backend;

    use super::PretrainedRef;
    use crate::{
        data::pretrained::{
            CacheStatus,
            Construct,
            Loaded,
            PretrainedCache,
            load_map,
        },
        errors::BunsenResult,
    };

    impl PretrainedRef {
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
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::data::pretrained::{
        GIVEN_NAMESPACE,
        Resource,
        Source,
        StaticPreFabConfig,
    };

    #[derive(Config, Debug)]
    struct Shape {
        width: usize,
    }

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

    fn resource(
        key: &str,
        file: &str,
    ) -> Resource {
        Resource {
            key: key.to_string(),
            file: file.to_string(),
            sha256: None,
            kind: Some("pytorch".to_string()),
            namespace: "a".to_string(),
            sources: vec![Source::Url(alloc::format!("https://a.example/{file}"))],
        }
    }

    /// `well-known:a/small`, as a factory copies it out of a table.
    fn named() -> PretrainedRef {
        PretrainedRef::Named {
            provider: "well-known".to_string(),
            pretrained: Pretrained {
                name: "a/small".to_string(),
                aliases: vec!["s".to_string()],
                description: "small weights".to_string(),
                license: None,
                origin: None,
                prefab: Some("small".to_string()),
                resources: ResourceMap::new("small")
                    .with_resource(resource("checkpoint", "small.pt"))
                    .with_resource(resource("vocabulary", "vocab.txt")),
            },
        }
    }

    /// A named ref is its provider's row: the id is `provider:ref`, the
    /// map is the row's under that id, and the prefab is the row's.
    #[test]
    fn test_a_named_ref_is_its_row() {
        let model = named();
        assert_eq!(model.id(), "well-known:a/small");
        let (provider, row) = model.named().unwrap();
        assert_eq!(provider, "well-known");
        assert_eq!(row.name, "a/small");
        assert_eq!(model.prefab(&PREFABS).map(|p| p.to_config().width), Some(4));

        let map = model.to_map();
        assert_eq!(
            map.name, "well-known:a/small",
            "a row's map is named by its id"
        );
        assert_eq!(map.keys(), ["checkpoint", "vocabulary"]);
        assert_eq!(map.get("checkpoint").unwrap().file, "small.pt");
        assert_eq!(model.clone(), model);
    }

    /// A given ref is the map it was given: no provider, no prefab.
    #[test]
    fn test_a_given_ref_is_its_map() {
        let map = ResourceMap::given("/models/ckpt.pt", "checkpoint", "/models/ckpt.pt");
        let model = PretrainedRef::from(map.clone());
        assert_eq!(model, PretrainedRef::Given(map.clone()));
        assert_eq!(model.id(), "/models/ckpt.pt");
        assert!(model.named().is_none());
        assert!(model.prefab(&PREFABS).is_none());
        assert_eq!(model.to_map(), map);
        assert_eq!(model.to_map().get("checkpoint").unwrap().file, "ckpt.pt");
    }

    /// An overlay replaces by key on either variant and leaves the id
    /// alone; the overlaid resource keeps its own namespace, `given`.
    #[test]
    fn test_with_overlay_replaces_by_key_on_both_variants() {
        let vocab = ResourceMap::given("--vocab", "vocabulary", "/mine/gpt2.tiktoken");

        let model = named().with_overlay(vocab.clone()).unwrap();
        assert_eq!(model.id(), "well-known:a/small");
        let map = model.to_map();
        assert_eq!(map.name, "well-known:a/small");
        assert_eq!(map.keys(), ["checkpoint", "vocabulary"]);
        let overlaid = map.get("vocabulary").unwrap();
        assert_eq!(overlaid.file, "gpt2.tiktoken");
        assert_eq!(overlaid.namespace, GIVEN_NAMESPACE);
        assert_eq!(map.get("checkpoint").unwrap().file, "small.pt", "untouched");
        assert_eq!(
            model
                .named()
                .unwrap()
                .1
                .resources
                .get("vocabulary")
                .unwrap()
                .file,
            "gpt2.tiktoken",
            "the row copy is what changed"
        );

        let given = PretrainedRef::from(ResourceMap::given("mine", "checkpoint", "/m/ckpt.pt"))
            .with_overlay(vocab)
            .unwrap();
        assert_eq!(given.id(), "mine");
        assert_eq!(given.to_map().keys(), ["checkpoint", "vocabulary"]);
    }

    /// A given path is local already and loads in place; a name goes
    /// through the cache, which is offline here and has nothing local.
    #[cfg(feature = "cache")]
    #[test]
    fn test_status_and_load() {
        use std::{
            fs,
            path::PathBuf,
            sync::Arc,
        };

        use burn::prelude::Backend;

        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::{
                    CacheStatus,
                    Construct,
                    LoadedResources,
                    PretrainedCache,
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

        let dir = tempfile::tempdir().unwrap();
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let file = dir.path().join("ckpt.pt");
        fs::write(&file, b"x").unwrap();
        let spec = file.to_str().unwrap();

        let given = PretrainedRef::from(ResourceMap::given(spec, "checkpoint", &file));
        assert_eq!(
            given.status("kit", &cache)["checkpoint"],
            CacheStatus::LocalDir
        );
        let loaded = given
            .load::<CpuBackend, _>(&cache, &CheckpointPath, &default_device())
            .unwrap();
        assert_eq!(*loaded.handle, file);
        assert_eq!(loaded.name, spec);
        assert_eq!(
            loaded.resources.get("checkpoint").unwrap().provenance,
            Provenance::LocalDir
        );

        let named = named();
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
