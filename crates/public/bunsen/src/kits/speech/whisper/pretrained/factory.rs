//! # The Whisper pretrained factory
//!
//! [`default_whisper_factory`] is the index a caller holds: Whisper's
//! providers, and the [`WhisperConstruct`] hook that reads what their rows
//! point at. `factory.load_bundle::<B>(spec, &cache, device)` is the whole
//! pathway from `--model` to a
//! [`WhisperBundle`](crate::kits::speech::whisper::driver::WhisperBundle);
//! the caller never builds a hook. What a caller may want to change on the
//! hook, the scanner for a checkpoint that is not laid out as upstream's,
//! the factory exposes as [`with_scanner`](PretrainedFactory::with_scanner).
//!
//! A caller with a provider of its own, a hub say, builds on the default:
//!
//! ```rust,ignore
//! let factory = default_whisper_factory()?
//!     .with_provider(Arc::new(my_hub))?; // answers `hub:org/repo`, lists nothing
//! let bundle = factory.load_bundle::<B>("openai/base", &cache, &device)?;
//! ```

use std::{
    path::Path,
    sync::Arc,
};

use burn::prelude::Backend;

use crate::{
    data::pretrained::{
        PretrainedCache,
        PretrainedFactory,
        PretrainedRef,
    },
    errors::BunsenResult,
    kits::speech::whisper::{
        WhisperApiConfig,
        driver::WhisperBundle,
        pretrained::{
            PytorchWhisperScanner,
            WhisperConstruct,
            default_whisper_providers,
        },
    },
};

/// Whisper's factory: [`default_whisper_providers`] behind
/// [`WhisperConstruct`], upstream's scanner.
///
/// # Errors
/// [`BunsenError::Invalid`](crate::errors::BunsenError::Invalid) if two of
/// the defaults share a name, which the tests pin they do not.
pub fn default_whisper_factory() -> BunsenResult<PretrainedFactory<WhisperConstruct>> {
    PretrainedFactory::new(WhisperConstruct::new()).with_providers(default_whisper_providers())
}

impl PretrainedFactory<WhisperConstruct> {
    /// Reads checkpoints through `scanner`: for one whose tensors are not
    /// under `model_state_dict`, or a front end or token layout that is
    /// not upstream's.
    pub fn with_scanner(
        self,
        scanner: PytorchWhisperScanner,
    ) -> Self {
        let hook = self.hook().clone().with_scanner(scanner);
        self.with_hook(hook)
    }

    /// Scans a checkpoint for its config without loading its weights, and
    /// checks it against the geometry `model` promises: the read-only
    /// half, for a listing.
    ///
    /// # Errors
    /// As [`WhisperConstruct::scan`].
    pub fn scan(
        &self,
        model: &PretrainedRef,
        path: &Path,
    ) -> BunsenResult<WhisperApiConfig> {
        self.hook().scan(model, path)
    }

    /// `--model` to a bundle: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_bundle<B: Backend>(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &B::Device,
    ) -> BunsenResult<Arc<WhisperBundle<B>>> {
        Ok(self.load::<B>(spec, cache, device)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::{
            WELL_KNOWN,
            testing::ListsNothing,
        },
        errors::BunsenError,
        kits::speech::whisper::pretrained::{
            CHECKPOINT,
            OPENAI,
            WHISPER_KIT,
            openai_download_root,
        },
    };

    /// The default factory is the well-known table, serving the kit, with
    /// every id qualified and upstream's aliases honoured; the bundled
    /// table joins it with the feature, after it.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_whisper_factory().unwrap();
        assert_eq!(factory.kit(), WHISPER_KIT);
        assert_eq!(factory.providers()[0].name(), "well-known");
        assert_eq!(factory.provider(WELL_KNOWN).unwrap().name(), "well-known");
        assert_eq!(factory.provider(WELL_KNOWN).unwrap().ids().len(), 12);
        if cfg!(feature = "whisper-weights") {
            assert_eq!(factory.providers().len(), 2);
            assert_eq!(factory.providers()[1].name(), "bundled");
            assert_eq!(factory.ids().len(), 13);
        } else {
            assert_eq!(factory.providers().len(), 1);
            assert_eq!(factory.ids().len(), 12);
        }

        let (provider, large) = factory.lookup("large").unwrap();
        assert_eq!(provider, "well-known");
        assert_eq!(large.name, "openai/large-v3");
        let (_, turbo) = factory.lookup("openai/turbo").unwrap();
        assert_eq!(turbo.name, "openai/large-v3-turbo");
        assert_eq!(
            factory.lookup("well-known:openai/tiny").unwrap().1.name,
            "openai/tiny"
        );
        assert!(matches!(
            factory.lookup("nobody:base"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            factory.lookup("gigantic"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(openai_download_root().is_some_and(|d| d.ends_with("whisper")));
        assert!(format!("{:?}", factory.hook()).contains("WhisperConstruct"));
    }

    /// A spec resolves to a row, by ref, bare name or alias, or to a path
    /// under the checkpoint key; the scanner rides on the factory.
    #[test]
    fn test_resolve_names_aliases_and_paths() {
        let factory = default_whisper_factory()
            .unwrap()
            .with_scanner(PytorchWhisperScanner::new().with_d_head(32));
        assert_eq!(factory.hook().scanner.d_head, 32);

        match factory.resolve("openai/tiny.en").unwrap() {
            PretrainedRef::Named {
                provider,
                pretrained,
            } => {
                assert_eq!(provider, "well-known");
                assert_eq!(pretrained.name, "openai/tiny.en");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            factory.resolve("well-known:openai/tiny.en").unwrap().id(),
            "well-known:openai/tiny.en"
        );
        assert_eq!(
            factory.resolve("turbo").unwrap().id(),
            "well-known:openai/large-v3-turbo"
        );

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("ckpt.pt");
        std::fs::write(&file, b"x").unwrap();
        match factory.resolve(file.to_str().unwrap()).unwrap() {
            PretrainedRef::Given(map) => {
                assert_eq!(map.keys(), [CHECKPOINT]);
                assert_eq!(map.get(CHECKPOINT).unwrap().file, "ckpt.pt");
            }
            other => panic!("{other:?}"),
        }

        assert!(matches!(
            factory.resolve("openai/gigantic"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            factory.resolve("/no/such/file.pt"),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }

    /// One prefab, many rows: derived from the listing, per group and
    /// through the factory.
    #[test]
    fn test_one_prefab_many_pretrained() {
        let large: Vec<&str> = OPENAI.for_prefab("large").iter().map(|p| p.name).collect();
        assert_eq!(large, ["large-v1", "large-v2"]);
        let derived: Vec<String> = default_whisper_factory()
            .unwrap()
            .for_prefab("large")
            .iter()
            .map(|(p, row)| format!("{p}:{}", row.name))
            .collect();
        assert_eq!(
            derived,
            ["well-known:openai/large-v1", "well-known:openai/large-v2"]
        );
    }

    /// A provider that lists nothing joins the defaults: its refs answer
    /// qualified, the bare names still only reach the tables, and the
    /// listing is unchanged.
    #[test]
    fn test_a_plugin_joins_the_defaults() {
        let hub = Arc::new(ListsNothing::default());
        let defaults = default_whisper_providers().len();
        let factory = default_whisper_factory()
            .unwrap()
            .with_provider(hub.clone())
            .unwrap();
        assert_eq!(factory.providers().len(), defaults + 1);
        assert_eq!(
            factory.ids().len(),
            factory.providers()[..defaults]
                .iter()
                .map(|p| p.ids().len())
                .sum::<usize>()
        );

        let (provider, row) = factory.lookup("hub:openai/whisper-base").unwrap();
        assert_eq!(provider, "hub");
        assert_eq!(row.name, "openai/whisper-base");
        assert_eq!(factory.lookup("base").unwrap().0, "well-known");
        assert_eq!(hub.lookups(), 1, "a bare name never reached the hub");
    }

    /// Registering the defaults twice is the error a duplicate name is.
    #[test]
    fn test_the_defaults_have_no_duplicate() {
        let err = default_whisper_factory()
            .unwrap()
            .with_providers(default_whisper_providers())
            .unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err}");
    }

    /// `bundled:openai/base` is the bundle's files, used in place from a
    /// cache that has nothing: both parts local dir, nothing written; and
    /// a bare `openai/base` still means the well-known row.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_the_bundled_provider_serves_openai_base_in_place() {
        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::{
                    BUNDLED,
                    CacheStatus,
                    PretrainedCacheOptions,
                    Provenance,
                },
            },
            kits::speech::whisper::pretrained::{
                BASE_CHECKPOINT,
                VOCABULARY,
            },
        };
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
        let factory = default_whisper_factory().unwrap();
        let bundled = factory.provider(BUNDLED).unwrap();
        assert_eq!(bundled.ids(), ["bundled:openai/base"]);
        assert_eq!(bundled.license(), None);

        let model = factory.resolve("bundled:openai/base").unwrap();
        assert_eq!(model.id(), "bundled:openai/base");
        let status = model.status(WHISPER_KIT, &cache);
        assert_eq!(status[CHECKPOINT], CacheStatus::LocalDir);
        assert_eq!(status[VOCABULARY], CacheStatus::LocalDir);
        let map = model.to_map();
        assert_eq!(
            map.get(CHECKPOINT).unwrap().sha256,
            BASE_CHECKPOINT.to_map().get(CHECKPOINT).unwrap().sha256,
            "the same file, pinned the same"
        );

        let loaded = cache.load(WHISPER_KIT, &map).unwrap();
        for (key, part) in loaded.iter() {
            assert_eq!(part.provenance, Provenance::LocalDir, "{key}");
        }
        assert_eq!(
            loaded.expect(CHECKPOINT).unwrap(),
            bunsen_bundled_whisper::base_pt()
        );
        assert_eq!(
            loaded.expect(VOCABULARY).unwrap(),
            bunsen_bundled_whisper::multilingual_tiktoken()
        );
        assert!(!dir.path().join("cache").exists(), "nothing was written");

        assert_eq!(
            factory.resolve("openai/base").unwrap().id(),
            "well-known:openai/base"
        );
    }

    /// A cache rooted at the bundle's directory, offline, has `openai/base`
    /// whole: both resources cached, nothing fetched, nothing written.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_the_bundle_serves_openai_base() {
        use crate::{
            data::pretrained::{
                CacheStatus,
                Provenance,
            },
            kits::speech::whisper::pretrained::{
                VOCABULARY,
                testing::offline_cache,
            },
        };
        let cache = offline_cache();
        let factory = default_whisper_factory().unwrap();
        let model = factory.resolve("openai/base").unwrap();
        assert_eq!(model.id(), "well-known:openai/base");
        let status = model.status(WHISPER_KIT, &cache);
        assert_eq!(status[CHECKPOINT], CacheStatus::Cached);
        assert_eq!(status[VOCABULARY], CacheStatus::Cached);

        let loaded = cache.load(WHISPER_KIT, &model.to_map()).unwrap();
        assert_eq!(loaded.map.name, "well-known:openai/base");
        for (key, part) in loaded.iter() {
            assert_eq!(part.provenance, Provenance::Cached, "{key}");
            assert!(part.path.starts_with(bunsen_bundled_whisper::cache_dir()));
        }
        assert_eq!(
            loaded.expect(CHECKPOINT).unwrap(),
            bunsen_bundled_whisper::base_pt()
        );
    }
}
