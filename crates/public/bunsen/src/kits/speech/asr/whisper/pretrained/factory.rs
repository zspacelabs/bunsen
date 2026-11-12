//! # The Whisper pretrained factory

use alloc::sync::Arc;

use burn::tensor::Device;

use crate::{
    data::{
        builds::{
            BuildFactory,
            DeferredBuild,
        },
        resources::{
            LoadedResourceSet,
            ResourceCache,
        },
    },
    errors::BunsenResult,
    kits::speech::asr::whisper::{
        WhisperApiConfig,
        driver::WhisperConstruct,
        pretrained::{
            WhisperConstructConfig,
            default_whisper_providers,
        },
    },
};

/// Whisper's factory over [`default_whisper_providers`], building through
/// [`WhisperConstructConfig`].
///
/// The index a caller holds: Whisper's providers, resolving names to
/// [`DeferredBuild`] models that carry the kit's [`WhisperConstructConfig`]
/// hook, chosen by what each row's map says about its checkpoint.
/// [`load_construct`](BuildFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`WhisperConstruct`]; the caller never builds or passes a hook. A
/// checkpoint on disk is not the factory's: it is a given map through
/// [`DeferredBuild::from_resource_set`], which gets its hook the same way. A
/// caller with a provider of its own, a mirror say, adds it with
/// [`with_provider`](BuildFactory::with_provider).
///
/// # Errors
/// As [`BuildFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_whisper_factory() -> BunsenResult<BuildFactory<WhisperConstructConfig>> {
    BuildFactory::new().with_providers(default_whisper_providers())
}

impl DeferredBuild<WhisperConstructConfig> {
    /// Scans the checkpoint in `loaded` for its config without loading its
    /// weights, and checks it against the geometry this model promises:
    /// the read-only half, for a listing.
    ///
    /// # Errors
    /// As [`WhisperConstructConfig::scan`].
    pub fn scan(
        &self,
        loaded: &LoadedResourceSet,
    ) -> BunsenResult<WhisperApiConfig> {
        self.config.scan(&self.model, loaded)
    }

    /// This model to a construct: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_construct(
        &self,
        cache: &ResourceCache,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<WhisperConstruct>> {
        Ok(self.load(cache, device, policy)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{
            builds::{
                BuildRef,
                CATALOG,
                testing::ListsNothing,
            },
            cache::BunsenDiskCacheOptions,
            resources::{
                ResourceCacheOptions,
                ResourceSet,
            },
        },
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::speech::asr::whisper::pretrained::{
            CHECKPOINT,
            WHISPER_KIT,
            WhisperReader,
        },
    };

    /// An offline cache under `dir`: nothing fetched, nothing reported.
    fn offline_cache_in(dir: &std::path::Path) -> ResourceCache {
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

    /// The default factory is the catalog provider, serving the kit, with
    /// every id qualified and upstream's aliases honoured, then Hugging
    /// Face, which lists nothing.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_whisper_factory().unwrap();
        assert_eq!(factory.kit(), WHISPER_KIT);
        assert_eq!(factory.providers()[0].name(), "catalog");
        assert_eq!(factory.provider(CATALOG).unwrap().name(), "catalog");
        assert_eq!(factory.provider(CATALOG).unwrap().ids().len(), 12);
        assert_eq!(factory.providers().last().unwrap().name(), "hf");
        assert!(factory.provider("hf").unwrap().ids().is_empty());
        assert_eq!(factory.providers().len(), 2);
        assert_eq!(factory.ids().len(), 12);

        let (provider, large) = factory.lookup("large").unwrap();
        assert_eq!(provider, "catalog");
        assert_eq!(large.name, "openai/whisper/large/large-v3");
        let (_, turbo) = factory.lookup("openai/turbo").unwrap();
        assert_eq!(turbo.name, "openai/whisper/large/large-v3-turbo");
        assert_eq!(
            factory.lookup("catalog:openai/tiny").unwrap().1.name,
            "openai/whisper/tiny"
        );
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&factory.lookup("nobody:base"));
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&factory.lookup("gigantic"));
    }

    /// A spec resolves to a deferred row, by ref, bare name or alias, with
    /// the hook its map calls for; a path on disk is not a name the
    /// factory knows, and becomes a deferred model on its own.
    #[test]
    fn test_resolve_names_and_aliases() {
        let factory = default_whisper_factory().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache_in(dir.path());

        let model = factory.resolve("openai/tiny.en", &cache).unwrap();
        match &model.model {
            BuildRef::Named {
                provider, build, ..
            } => {
                assert_eq!(provider, "catalog");
                assert_eq!(build.name, "openai/whisper/tiny.en");
                assert_eq!(build.resource_set().keys(), [CHECKPOINT, "vocabulary"]);
            }
            other => panic!("{other:?}"),
        }
        assert!(format!("{:?}", model.config).contains("WhisperConstructConfig"));
        assert_eq!(
            factory
                .resolve("catalog:openai/tiny.en", &cache)
                .unwrap()
                .id(),
            "catalog:openai/whisper/tiny.en"
        );
        assert_eq!(
            factory.resolve("turbo", &cache).unwrap().id(),
            "catalog:openai/whisper/large/large-v3-turbo"
        );

        let file = dir.path().join("ckpt.pt");
        std::fs::write(&file, b"x").unwrap();
        let spec = file.to_str().unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&factory.resolve(spec, &cache));
        let given = DeferredBuild::<WhisperConstructConfig>::from_resource_set(ResourceSet::given(
            spec, CHECKPOINT, &file,
        ))
        .unwrap();
        assert_eq!(given.id(), spec);
        assert!(given.model.build().is_none());

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .assert_err(&factory.resolve("openai/gigantic", &cache));

        // A Hugging Face ref is resolved through the cache: offline, with
        // no listing cached, it is refused naming the ref; the index alone
        // does not answer it; a bare `org/repo` never reaches the hub.
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .display_contains("hf:openai/whisper-tiny")
            .assert_err(&factory.resolve("hf:openai/whisper-tiny", &cache));
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .assert_err(&factory.lookup("hf:openai/whisper-tiny"));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .assert_err(&factory.lookup("openai/whisper-tiny"));
        ErrorMatcher::kind(BunsenErrorKind::Illegal).assert_err(&factory.lookup("hf:whisper-tiny"));
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
        assert_eq!(row.name, "hub/openai/whisper-base");
        assert_eq!(factory.lookup("base").unwrap().0, "catalog");
        assert_eq!(hub.lookups(), 1, "a bare name never reached the hub");

        // The hub's rows are safetensors: resolving one chooses the
        // safetensors reader, with its feature, before anything is read;
        // without the feature it is refused naming the feature.
        let dir = tempfile::tempdir().unwrap();
        let resolved = factory.resolve("hub:openai/whisper-base", &offline_cache_in(dir.path()));
        assert!(matches!(
            resolved.unwrap().config.reader,
            WhisperReader::Safetensors(_)
        ));
    }

    /// Registering the defaults twice is the error a duplicate name is.
    #[test]
    fn test_the_defaults_have_no_duplicate() {
        ErrorMatcher::kind(BunsenErrorKind::Illegal).assert_err(
            &default_whisper_factory()
                .unwrap()
                .with_providers(default_whisper_providers()),
        );
    }
}

/// Against the hub: the smallest repo, fetched once into the default
/// cache, read whole, and checked against `OpenAI`'s own `tiny.pt`, which
/// it was converted from.
#[cfg(all(test, feature = "fetch"))]
mod hub_tests {
    use burn::tensor::{
        Tensor,
        Tolerance,
    };

    use crate::{
        data::{
            pretrained::Prefabs,
            resources::{
                Provenance,
                ResourceCache,
                ResourceCacheOptions,
            },
        },
        kits::speech::asr::whisper::{
            WhisperApiConfig,
            WhisperMeta,
            pretrained::{
                CHECKPOINT,
                default_whisper_factory,
            },
        },
        support::testing::{
            assert_tensors_close,
            cpu_device,
        },
    };

    /// `hf:openai/whisper-tiny` resolves through the hub's listing to a
    /// pinned checkpoint, loads to the `tiny` geometry with the
    /// multilingual layout and vocabulary, and its weights are
    /// `openai/tiny`'s: the same numbers through two files, two layouts and
    /// two readers.
    #[test]
    fn test_hf_whisper_tiny_is_openai_tiny() {
        let device = cpu_device();
        let cache = ResourceCache::new(ResourceCacheOptions::default()).unwrap();
        let factory = default_whisper_factory().unwrap();

        let model = factory.resolve("hf:openai/whisper-tiny", &cache).unwrap();
        let checkpoint = model.to_resource_set();
        let checkpoint = checkpoint.get(CHECKPOINT).unwrap();
        assert_eq!(checkpoint.file, "model.safetensors");
        assert!(checkpoint.is_pinned(), "pinned by the hub's listing");

        let loaded = model
            .load(
                &cache,
                &device,
                crate::data::formats::LoadPolicy::as_stored(),
            )
            .unwrap();
        assert!(matches!(
            loaded.resources.get(CHECKPOINT).unwrap().provenance,
            Provenance::Cached | Provenance::Downloaded
        ));
        let hf = loaded.handle;
        let tiny = WhisperApiConfig::expect_prefab("tiny").geometry();
        assert_eq!(hf.model.n_mels(), tiny.n_mels);
        assert_eq!(hf.model.vocab_size(), tiny.vocab_size);
        assert_eq!(hf.model.d_model(), tiny.d_model);
        assert_eq!(hf.model.max_audio_ctx(), tiny.max_audio_ctx);
        assert_eq!(hf.model.max_text_ctx(), tiny.max_text_ctx);
        assert_eq!(hf.model.encoder.blocks.len(), tiny.n_encoder_layers);
        assert_eq!(hf.model.decoder.blocks.len(), tiny.n_decoder_layers);
        assert!(hf.layout.ids().is_multilingual());
        assert_eq!(hf.ranks.as_ref().map(|r| r.len()), Some(50257));

        let openai = factory
            .load_construct(
                "openai/tiny",
                &cache,
                &device,
                crate::data::formats::LoadPolicy::as_stored(),
            )
            .unwrap();
        let close = |a: Tensor<2>, b: Tensor<2>| {
            assert_tensors_close(&a, &b, Tolerance::<f32>::default());
        };
        close(
            hf.model.encoder.blocks[0].attn.branch.query.weight.val(),
            openai.model.encoder.blocks[0]
                .attn
                .branch
                .query
                .weight
                .val(),
        );
        close(
            hf.model.decoder.blocks[3]
                .cross_attn
                .branch
                .output
                .weight
                .val(),
            openai.model.decoder.blocks[3]
                .cross_attn
                .branch
                .output
                .weight
                .val(),
        );
        close(
            hf.model.decoder.blocks[1].mlp.branch.linear2.weight.val(),
            openai.model.decoder.blocks[1]
                .mlp
                .branch
                .linear2
                .weight
                .val(),
        );
        close(
            hf.model.encoder.positional_embedding.val(),
            openai.model.encoder.positional_embedding.val(),
        );
        close(
            hf.model.decoder.token_embedding.weight.val(),
            openai.model.decoder.token_embedding.weight.val(),
        );
        close(
            hf.model.decoder.ln.gamma.val().unsqueeze(),
            openai.model.decoder.ln.gamma.val().unsqueeze(),
        );
    }
}
