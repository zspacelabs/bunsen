//! # Model references
//!
//! What a model name resolved to: an entry in a provider table, or a
//! checkpoint on disk. The name-to-model pathway's first step, shared by
//! every kit; the kit supplies the providers, the prefab map, and the
//! check-and-load that follow.

use std::{
    fmt::Debug,
    path::{
        Path,
        PathBuf,
    },
};

use burn::config::Config;

use super::{
    PreFabConfig,
    Provenance,
    ResolvedWeights,
    StaticPreFabMap,
    StaticPretrainedProvider,
    StaticPretrainedWeightsDescriptor,
    WeightsCache,
    available_ids,
    lookup_pretrained,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// What a model name resolved to.
#[derive(Debug, Clone)]
pub enum ModelRef {
    /// An entry in a provider table.
    Pretrained {
        /// Its provider.
        provider: &'static StaticPretrainedProvider<'static>,
        /// The entry.
        pretrained: &'static StaticPretrainedWeightsDescriptor<'static>,
    },

    /// A checkpoint on disk. No prefab is promised; a scan says what it is.
    Path(PathBuf),
}

impl ModelRef {
    /// Resolves a model spec against `providers`.
    ///
    /// `provider/name` looks up that provider; a bare name looks across all
    /// of them; an alias is honoured. Failing those, a path to an existing
    /// file is taken as a checkpoint. The index wins over the file system.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`], naming what is available.
    pub fn resolve(
        providers: &[&'static StaticPretrainedProvider<'static>],
        spec: &str,
    ) -> BunsenResult<Self> {
        let (provider, name) = match spec.rsplit_once('/') {
            Some((provider, name)) => (Some(provider), name),
            None => (None, spec),
        };
        if let Some((provider, pretrained)) = lookup_pretrained(providers, provider, name) {
            return Ok(Self::Pretrained {
                provider,
                pretrained,
            });
        }

        let path = Path::new(spec);
        if path.is_file() {
            return Ok(Self::Path(path.to_path_buf()));
        }

        Err(BunsenError::ResourceNotFound(format!(
            "no model {spec:?}: not a pretrained name and not a file; the names are {}",
            available_ids(providers).join(", ")
        )))
    }

    /// The qualified id, or the path.
    pub fn id(&self) -> String {
        match self {
            Self::Pretrained {
                provider,
                pretrained,
            } => provider.id(pretrained),
            Self::Path(path) => path.display().to_string(),
        }
    }

    /// The provider and entry, if the name was a name.
    pub fn pretrained(
        &self
    ) -> Option<(
        &'static StaticPretrainedProvider<'static>,
        &'static StaticPretrainedWeightsDescriptor<'static>,
    )> {
        match self {
            Self::Pretrained {
                provider,
                pretrained,
            } => Some((provider, pretrained)),
            Self::Path(_) => None,
        }
    }

    /// The prefab the name promised, from `prefabs`, if it was a name.
    ///
    /// # Panics
    /// If the entry names a prefab `prefabs` does not have; a kit's tests
    /// pin that every entry's does.
    pub fn prefab<C>(
        &self,
        prefabs: &StaticPreFabMap<C>,
    ) -> Option<PreFabConfig<C>>
    where
        C: 'static + Config + Debug + Clone,
    {
        self.pretrained()
            .map(|(_, pretrained)| prefabs.expect_lookup_prefab(pretrained.prefab))
    }

    /// Brings the weights local, under `kit` in the cache.
    ///
    /// # Errors
    /// As [`WeightsCache::resolve`]. A path is local already.
    pub fn locate(
        &self,
        kit: &str,
        cache: &WeightsCache,
    ) -> BunsenResult<ResolvedWeights> {
        match self {
            Self::Pretrained {
                provider,
                pretrained,
            } => cache.resolve(kit, provider.name, &pretrained.to_descriptor()),
            Self::Path(path) => Ok(ResolvedWeights {
                path: path.clone(),
                provenance: Provenance::Given,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::data::{
        cache::BunsenDiskCacheOptions,
        pretrained::{
            StaticPreFabConfig,
            StaticWeightsSource,
            WeightsCacheOptions,
            WeightsFormat,
        },
    };

    #[derive(Config, Debug)]
    struct Shape {
        width: usize,
    }

    static SMALL: StaticPretrainedWeightsDescriptor<'static> = StaticPretrainedWeightsDescriptor {
        name: "small",
        description: "small weights",
        license: None,
        origin: None,
        prefab: "small",
        aliases: &["s"],
        file: "small.pt",
        sha256: None,
        format: WeightsFormat::PYTORCH_F32,
        sources: &[StaticWeightsSource::Url("https://a.example/small.pt")],
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

    #[test]
    fn test_resolve_names_aliases_and_paths() {
        match ModelRef::resolve(PROVIDERS, "a/small").unwrap() {
            ModelRef::Pretrained {
                provider,
                pretrained,
            } => {
                assert_eq!((provider.name, pretrained.name), ("a", "small"));
            }
            other => panic!("{other:?}"),
        }
        let by_alias = ModelRef::resolve(PROVIDERS, "s").unwrap();
        assert_eq!(by_alias.id(), "a/small");
        assert_eq!(
            by_alias.pretrained().map(|(p, d)| (p.name, d.name)),
            Some(("a", "small"))
        );
        assert_eq!(
            by_alias.prefab(&PREFABS).map(|p| p.to_config().width),
            Some(4)
        );

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("ckpt.pt");
        fs::write(&file, b"x").unwrap();
        let by_path = ModelRef::resolve(PROVIDERS, file.to_str().unwrap()).unwrap();
        assert!(matches!(&by_path, ModelRef::Path(p) if p == &file));
        assert_eq!(by_path.id(), file.display().to_string());
        assert!(by_path.pretrained().is_none());
        assert!(by_path.prefab(&PREFABS).is_none());

        match ModelRef::resolve(PROVIDERS, "a/gigantic") {
            Err(BunsenError::ResourceNotFound(m)) => assert!(m.contains("a/small"), "{m}"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            ModelRef::resolve(PROVIDERS, "/no/such/file.pt"),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }

    /// A path is local already; a name goes through the cache, which is
    /// offline here and has nothing local.
    #[test]
    fn test_locate() {
        let dir = tempfile::tempdir().unwrap();
        let cache = WeightsCache::new(
            WeightsCacheOptions::default()
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
        let given = ModelRef::Path(file.clone()).locate("kit", &cache).unwrap();
        assert_eq!(
            given,
            ResolvedWeights {
                path: file,
                provenance: Provenance::Given,
            }
        );

        let named = ModelRef::resolve(PROVIDERS, "a/small").unwrap();
        assert!(matches!(
            named.locate("kit", &cache),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }
}
