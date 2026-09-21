//! # Constructing a Whisper pretrained
//!
//! The [`Construct`] hook for the Whisper kit: from a resolved model to a
//! [`WhisperBundle`]. [`plan`](Construct::plan) scans the checkpoint,
//! checks it against the geometry the model promises, and applies the
//! vocabulary rule, all before anything but the checkpoint is brought
//! local; [`construct`](Construct::construct) reads the checkpoint through
//! the scanner and the vocabulary through the rank parser, and checks that
//! they agree. [`scan`](WhisperConstruct::scan) is the read-only half, for
//! a listing that wants the geometry without the weights.

use std::{
    path::Path,
    sync::Arc,
};

use burn::prelude::Backend;

use crate::{
    data::pretrained::{
        Construct,
        Fuse,
        GIVEN_NAMESPACE,
        LoadedResources,
        PretrainedCache,
        PretrainedRef,
        ResourceMap,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::whisper::{
            WhisperApiConfig,
            WhisperGeometry,
            driver::WhisperBundle,
            pretrained::{
                CHECKPOINT,
                PytorchWhisperScanner,
                VOCABULARY,
                WHISPER_KIT,
                WHISPER_PREFABS,
                WhisperVocabulary,
            },
        },
        tokens::TiktokenRanks,
    },
};

/// How a Whisper pretrained is built: the scanner that reads the
/// checkpoint, and the geometry it must have.
#[derive(Clone, Debug)]
pub struct WhisperConstruct {
    /// How the checkpoint is read: the key its tensors sit under, the head
    /// size, and the front end and token layout to declare.
    pub scanner: PytorchWhisperScanner,

    /// The geometry the checkpoint must scan to, overriding what a row's
    /// prefab promises. `None`, the default, expects the prefab's geometry
    /// when the model was named, and accepts whatever is found when it was
    /// given.
    pub expected: Option<WhisperGeometry>,
}

impl Default for WhisperConstruct {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperConstruct {
    /// Upstream's scanner, expecting what the model promises.
    pub fn new() -> Self {
        Self {
            scanner: PytorchWhisperScanner::new(),
            expected: None,
        }
    }

    /// Sets the scanner.
    pub fn with_scanner(
        mut self,
        scanner: PytorchWhisperScanner,
    ) -> Self {
        self.scanner = scanner;
        self
    }

    /// Sets the geometry the checkpoint must scan to, which wins over the
    /// one a row's prefab promises.
    pub fn with_expected(
        mut self,
        expected: Option<WhisperGeometry>,
    ) -> Self {
        self.expected = expected;
        self
    }

    /// The geometry `model` must scan to: the explicit one, else its
    /// prefab's when it was named and names one, else none.
    pub fn expected_for(
        &self,
        model: &PretrainedRef,
    ) -> Option<WhisperGeometry> {
        self.expected.or_else(|| {
            model
                .prefab(&WHISPER_PREFABS)
                .map(|prefab| prefab.to_config().geometry())
        })
    }

    /// Scans a checkpoint for its config without loading its weights, and
    /// checks it against the geometry `model` promises.
    ///
    /// # Errors
    /// As [`PytorchWhisperScanner::scan_cfg`];
    /// [`BunsenError::Invalid`] naming both geometries when the file at
    /// that name is not the model it claims to be.
    pub fn scan(
        &self,
        model: &PretrainedRef,
        path: &Path,
    ) -> BunsenResult<WhisperApiConfig> {
        let (_, cfg) = self.scanner.scan_cfg(path)?;
        self.check(model, &cfg.geometry())?;
        Ok(cfg)
    }

    /// Checks a scanned geometry against the one `model` promises.
    fn check(
        &self,
        model: &PretrainedRef,
        found: &WhisperGeometry,
    ) -> BunsenResult<()> {
        match self.expected_for(model) {
            Some(expected) if expected != *found => Err(BunsenError::Invalid(format!(
                "{}: the checkpoint's geometry is {found:?}, not the promised {expected:?}",
                model.id()
            ))),
            _ => Ok(()),
        }
    }
}

impl Construct for WhisperConstruct {
    type Built<B: Backend> = WhisperBundle<B>;

    const GIVEN_KEY: Option<&'static str> = Some(CHECKPOINT);
    const KIT: &'static str = WHISPER_KIT;

    /// Scans the checkpoint, which comes local for it, checks its geometry
    /// against the promised one, and settles the vocabulary: a map without
    /// one gets the one the checkpoint's layout selects; a map that
    /// declares one must declare that, unless the caller gave it, which is
    /// trusted.
    fn plan(
        &self,
        model: &PretrainedRef,
        cache: &PretrainedCache,
    ) -> BunsenResult<ResourceMap> {
        let map = model.to_map();
        let checkpoint = cache.resolve(WHISPER_KIT, map.try_get(CHECKPOINT)?)?;
        let cfg = self.scan(model, &checkpoint.path)?;

        let ids = *cfg.token_layout.policy_for_vocab(cfg.vocab_size)?.ids();
        let rule = WhisperVocabulary::for_layout(&ids).map().to_map();
        match map.get(VOCABULARY).cloned() {
            None => map.fuse(rule, Fuse::Strict),
            Some(given) if given.namespace == GIVEN_NAMESPACE => Ok(map),
            Some(declared) => {
                let selected = rule.try_get(VOCABULARY)?;
                if &declared != selected {
                    return Err(BunsenError::Invalid(format!(
                        "{}: declares the vocabulary {} but the checkpoint's layout selects {}",
                        map.name, declared.file, selected.file
                    )));
                }
                Ok(map)
            }
        }
    }

    /// Reads the checkpoint into a model at the precision it ships in, and
    /// the vocabulary, when there is one, into ranks the layout agrees
    /// with.
    fn construct<B: Backend>(
        &self,
        _model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &B::Device,
    ) -> BunsenResult<Arc<WhisperBundle<B>>> {
        let (model, cfg) = self
            .scanner
            .load::<B, _>(loaded.expect(CHECKPOINT)?, device)?;
        let layout = cfg.token_layout.policy_for_vocab(cfg.vocab_size)?;
        let mut bundle = WhisperBundle::new(model, layout);
        if let Some(part) = loaded.get(VOCABULARY) {
            bundle = bundle.with_ranks(TiktokenRanks::load(&part.path)?);
        }
        bundle.validate()?;
        Ok(Arc::new(bundle))
    }
}

/// Against the bundle's directory: the whole pathway on real files.
#[cfg(all(test, feature = "whisper-weights"))]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        data::pretrained::Provenance,
        kits::speech::whisper::{
            WhisperMeta,
            pretrained::{
                BASE_CHECKPOINT,
                GPT2_VOCABULARY,
                default_whisper_factory,
                testing::offline_cache,
            },
        },
        support::testing::{
            PerformanceBackend,
            default_device,
        },
    };

    fn resolve_model(spec: &str) -> BunsenResult<PretrainedRef> {
        default_whisper_factory()?.resolve_for::<WhisperConstruct>(spec)
    }

    fn geometry_of(prefab: &str) -> WhisperGeometry {
        WHISPER_PREFABS
            .expect_lookup_prefab(prefab)
            .to_config()
            .geometry()
    }

    /// `openai/base` loads whole from the bundle's directory: the model, a
    /// multilingual layout, the multilingual vocabulary, both parts cached.
    #[test]
    #[serial]
    fn test_the_factory_builds_the_bundle() {
        let cache = offline_cache();
        let loaded = default_whisper_factory()
            .unwrap()
            .load::<PerformanceBackend, _>(
                "openai/base",
                &cache,
                &WhisperConstruct::new(),
                &default_device(),
            )
            .unwrap();

        assert_eq!(loaded.name, "well-known:openai/base");
        assert_eq!(loaded.resources.keys(), [CHECKPOINT, VOCABULARY]);
        for (key, part) in loaded.resources.iter() {
            assert_eq!(part.provenance, Provenance::Cached, "{key}");
        }
        let bundle = &loaded.handle;
        assert_eq!(bundle.model.vocab_size(), 51865);
        assert_eq!(bundle.model.n_mels(), 80);
        assert!(bundle.layout.ids().is_multilingual());
        assert_eq!(bundle.ranks.as_ref().map(|r| r.len()), Some(50257));
        assert!(!bundle.default_filters().is_empty());
        assert_eq!(Arc::strong_count(&loaded.handle), 1);
    }

    /// A path has only the rule: the plan adds the vocabulary the scanned
    /// layout selects.
    #[test]
    fn test_plan_derives_a_paths_vocabulary() {
        let cache = offline_cache();
        let given = PretrainedRef::from(ResourceMap::given(
            "base",
            CHECKPOINT,
            bunsen_bundled_whisper::base_pt(),
        ));
        let planned = WhisperConstruct::new().plan(&given, &cache).unwrap();
        assert_eq!(planned.name, "base");
        assert_eq!(planned.keys(), [CHECKPOINT, VOCABULARY]);
        assert_eq!(
            planned.get(VOCABULARY).unwrap().file,
            "multilingual.tiktoken"
        );
    }

    /// A row that declares the wrong vocabulary is refused; a vocabulary
    /// the caller gave is trusted.
    #[test]
    fn test_plan_checks_a_declared_vocabulary() {
        let cache = offline_cache();
        let wrong = BASE_CHECKPOINT
            .to_map()
            .fuse(GPT2_VOCABULARY.to_map(), Fuse::Strict)
            .unwrap();
        let err = WhisperConstruct::new()
            .plan(&PretrainedRef::from(wrong), &cache)
            .unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("gpt2.tiktoken") && m.contains("multilingual.tiktoken")),
            "{err}"
        );

        let overridden = PretrainedRef::from(BASE_CHECKPOINT.to_map())
            .with_overlay(ResourceMap::given(
                "--vocab",
                VOCABULARY,
                bunsen_bundled_whisper::gpt2_tiktoken(),
            ))
            .unwrap();
        let planned = WhisperConstruct::new().plan(&overridden, &cache).unwrap();
        assert_eq!(planned.get(VOCABULARY).unwrap().file, "gpt2.tiktoken");
    }

    /// The geometry a name promises is checked against the scan before
    /// anything is loaded: a named model plans against its prefab, a
    /// given one promises nothing, and an explicit expectation wins over
    /// both.
    #[test]
    fn test_plan_checks_the_prefab_a_name_promises() {
        let cache = offline_cache();
        let base_pt = bunsen_bundled_whisper::base_pt();
        let given = PretrainedRef::from(BASE_CHECKPOINT.to_map());
        let named = resolve_model("openai/base").unwrap();
        let hook = WhisperConstruct::new();

        assert_eq!(hook.expected_for(&given), None);
        assert_eq!(hook.expected_for(&named), Some(geometry_of("base")));
        hook.plan(&named, &cache).unwrap();
        hook.plan(&given, &cache).unwrap();

        let tiny = hook.clone().with_expected(Some(geometry_of("tiny")));
        let err = tiny.plan(&given, &cache).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("promised")),
            "{err}"
        );

        // An explicit expectation overrides a wrong promise: `base.pt`
        // under `openai/tiny`, told to expect base, scans.
        let wrong_name = resolve_model("openai/tiny").unwrap();
        assert!(hook.scan(&wrong_name, base_pt).is_err());
        let explicit = hook.clone().with_expected(Some(geometry_of("base")));
        assert_eq!(
            explicit.expected_for(&wrong_name),
            Some(geometry_of("base"))
        );
        explicit.scan(&wrong_name, base_pt).unwrap();
    }

    /// The bundled checkpoint is `openai/base`; scanning it must agree with
    /// the `base` prefab, which is what pins the prefab table to a real
    /// file.
    #[test]
    fn test_the_bundled_base_scans_as_the_base_prefab() {
        let model = resolve_model("openai/base").unwrap();
        let cfg = WhisperConstruct::new()
            .scan(&model, bunsen_bundled_whisper::base_pt())
            .unwrap();

        let geometry = cfg.geometry();
        assert_eq!(geometry.prefab().map(|p| p.name), Some("base"));
        assert_eq!(geometry.n_heads(), 8);
    }

    /// The scanner is honored: one that declares another head size scans
    /// the bundled file to another geometry, which a named model rejects
    /// and a given one reports.
    #[test]
    fn test_scan_honors_the_scanner() {
        let base = bunsen_bundled_whisper::base_pt();
        let hook =
            WhisperConstruct::new().with_scanner(PytorchWhisperScanner::new().with_d_head(32));

        let named = resolve_model("openai/base").unwrap();
        let err = hook.scan(&named, base).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err}");

        let given = PretrainedRef::from(ResourceMap::given("base", CHECKPOINT, base));
        let cfg = hook.scan(&given, base).unwrap();
        assert_eq!(cfg.geometry().d_head, 32);
        assert_eq!(cfg.geometry().n_heads(), 16);
    }

    /// `base.pt` scanned as if it were `openai/tiny`: same file, wrong
    /// promise.
    #[test]
    fn test_a_checkpoint_under_the_wrong_name_is_rejected() {
        let model = resolve_model("openai/tiny").unwrap();
        let err = WhisperConstruct::new()
            .scan(&model, bunsen_bundled_whisper::base_pt())
            .unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err}");
        assert!(err.to_string().contains("openai/tiny"), "{err}");
    }
}
