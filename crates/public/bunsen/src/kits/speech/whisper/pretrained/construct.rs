//! # Constructing a Whisper pretrained
//!
//! The [`Construct`] hook for the Whisper kit: from a loaded resource map
//! to a [`WhisperBundle`]. [`plan`](Construct::plan) applies the vocabulary
//! rule and the geometry check before anything but the checkpoint is
//! brought local; [`construct`](Construct::construct) reads the checkpoint
//! through the scanner and the vocabulary through the rank parser, and
//! checks that they agree.

use std::sync::Arc;

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
            WhisperGeometry,
            driver::WhisperBundle,
            pretrained::{
                CHECKPOINT,
                PytorchWhisperScanner,
                VOCABULARY,
                WHISPER_KIT,
                WHISPER_PREFABS,
                vocabulary_map,
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

    /// The geometry the checkpoint must scan to: what a row's prefab
    /// promised. `None` accepts whatever is found.
    pub expected: Option<WhisperGeometry>,
}

impl Default for WhisperConstruct {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperConstruct {
    /// Upstream's scanner, expecting nothing in particular.
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

    /// Sets the geometry the checkpoint must scan to.
    pub fn with_expected(
        mut self,
        expected: Option<WhisperGeometry>,
    ) -> Self {
        self.expected = expected;
        self
    }

    /// Expects the geometry `model`'s prefab promises, if it was named and
    /// names one; a given model promises nothing.
    pub fn expecting(
        self,
        model: &PretrainedRef,
    ) -> Self {
        let expected = model
            .prefab(&WHISPER_PREFABS)
            .map(|prefab| prefab.to_config().geometry());
        self.with_expected(expected)
    }
}

impl Construct for WhisperConstruct {
    type Built<B: Backend> = WhisperBundle<B>;

    const KIT: &'static str = WHISPER_KIT;

    /// Scans the checkpoint, which comes local for it, checks its geometry
    /// against the expected one, and settles the vocabulary: a map without
    /// one gets the one the checkpoint's layout selects; a map that
    /// declares one must declare that, unless the caller gave it, which is
    /// trusted.
    fn plan(
        &self,
        map: ResourceMap,
        cache: &PretrainedCache,
    ) -> BunsenResult<ResourceMap> {
        let checkpoint = cache.resolve(WHISPER_KIT, map.try_get(CHECKPOINT)?)?;
        let (_, cfg) = self.scanner.scan_cfg(&checkpoint.path)?;
        if let Some(expected) = &self.expected {
            let found = cfg.geometry();
            if found != *expected {
                return Err(BunsenError::Invalid(format!(
                    "{}: the checkpoint's geometry is {found:?}, not the promised {expected:?}",
                    map.name
                )));
            }
        }

        let ids = *cfg.token_layout.policy_for_vocab(cfg.vocab_size)?.ids();
        let rule = vocabulary_map(&ids).to_map();
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
                load_named,
                testing::offline_cache,
            },
        },
        support::testing::{
            PerformanceBackend,
            default_device,
        },
    };

    /// `openai/base` loads whole from the bundle's directory: the model, a
    /// multilingual layout, the multilingual vocabulary, both parts cached.
    #[test]
    #[serial]
    fn test_load_named_builds_the_bundle() {
        let cache = offline_cache();
        let loaded = load_named::<PerformanceBackend>(
            "openai/base",
            &cache,
            &WhisperConstruct::new(),
            &default_device(),
        )
        .unwrap();

        assert_eq!(loaded.name, "openai/base");
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
        let given = ResourceMap::given("base", CHECKPOINT, bunsen_bundled_whisper::base_pt());
        let planned = WhisperConstruct::new().plan(given, &cache).unwrap();
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
        let err = WhisperConstruct::new().plan(wrong, &cache).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("gpt2.tiktoken") && m.contains("multilingual.tiktoken")),
            "{err}"
        );

        let overridden = BASE_CHECKPOINT
            .to_map()
            .fuse(
                ResourceMap::given(
                    "--vocab",
                    VOCABULARY,
                    bunsen_bundled_whisper::gpt2_tiktoken(),
                ),
                Fuse::Overlay,
            )
            .unwrap();
        let planned = WhisperConstruct::new().plan(overridden, &cache).unwrap();
        assert_eq!(planned.get(VOCABULARY).unwrap().file, "gpt2.tiktoken");
    }

    /// The geometry a name promises is checked against the scan before
    /// anything is loaded.
    #[test]
    fn test_plan_checks_the_expected_geometry() {
        let cache = offline_cache();
        let tiny = WHISPER_PREFABS
            .expect_lookup_prefab("tiny")
            .to_config()
            .geometry();
        let hook = WhisperConstruct::new().with_expected(Some(tiny));
        let err = hook.plan(BASE_CHECKPOINT.to_map(), &cache).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("promised")),
            "{err}"
        );

        let base = PretrainedRef::resolve(
            crate::kits::speech::whisper::pretrained::WHISPER_PROVIDERS,
            "openai/base",
            CHECKPOINT,
        )
        .unwrap();
        let hook = WhisperConstruct::new().expecting(&base);
        assert!(hook.expected.is_some());
        hook.plan(base.to_map(), &cache).unwrap();
    }
}
