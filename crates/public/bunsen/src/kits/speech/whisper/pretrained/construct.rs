//! # Constructing a Whisper pretrained

use std::{
    error::Error,
    fmt,
    sync::Arc,
};

use burn::prelude::Backend;

#[cfg(feature = "store_safetensors")]
use crate::data::pretrained::SafetensorsCheckpoint;
#[cfg(not(feature = "store_safetensors"))]
use crate::kits::speech::whisper::Whisper;
#[cfg(feature = "store_safetensors")]
use crate::kits::speech::whisper::{
    Whisper,
    pretrained::SafetensorsWhisperScanner,
};
use crate::{
    data::pretrained::{
        Construct,
        Fuse,
        GIVEN_NAMESPACE,
        LoadedResources,
        PretrainedCache,
        PretrainedRef,
        ResourceMap,
        SAFETENSORS,
    },
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        ResultContext,
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

/// A checkpoint scanned to a geometry other than the one its model
/// promises: the files at a name are not the model the name claims.
///
/// The cause of the [`InvalidResource`](BunsenErrorKind::InvalidResource)
/// error [`WhisperConstruct::scan`] (and so
/// [`plan`](Construct::plan)) returns for it. A caller that wants to tell
/// this apart from a checkpoint it could not read at all finds it with
/// [`BunsenError::find`]:
///
/// ```
/// # #[cfg(all(feature = "store_pytorch", feature = "cache"))] {
/// use bunsen::{
///     errors::{
///         BunsenError,
///         BunsenErrorKind,
///     },
///     kits::speech::whisper::{
///         WhisperGeometry,
///         pretrained::GeometryMismatch,
///     },
/// };
///
/// let e = BunsenError::from(GeometryMismatch::new(
///     "well-known:openai/tiny",
///     WhisperGeometry::openai(80, 51865, 384, 4, 4),
///     WhisperGeometry::openai(80, 51865, 512, 6, 6),
/// ));
/// assert_eq!(e.kind(), BunsenErrorKind::InvalidResource);
/// let mismatch = e.find::<GeometryMismatch>().unwrap();
/// assert_eq!(mismatch.model, "well-known:openai/tiny");
/// # }
/// ```
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeometryMismatch {
    /// The model's id, as [`PretrainedRef::id`] gives it.
    pub model: String,
    /// The geometry the model promises.
    pub expected: WhisperGeometry,
    /// The geometry the checkpoint scanned to.
    pub found: WhisperGeometry,
}

impl GeometryMismatch {
    /// `model`'s checkpoint scanned to `found`, not the promised `expected`.
    pub fn new(
        model: impl Into<String>,
        expected: WhisperGeometry,
        found: WhisperGeometry,
    ) -> Self {
        Self {
            model: model.into(),
            expected,
            found,
        }
    }
}

impl fmt::Display for GeometryMismatch {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "{}: the checkpoint's geometry is {:?}, not the promised {:?}",
            self.model, self.found, self.expected
        )
    }
}

impl Error for GeometryMismatch {}

impl From<GeometryMismatch> for BunsenError {
    #[track_caller]
    fn from(error: GeometryMismatch) -> Self {
        BunsenError::from_cause(BunsenErrorKind::InvalidResource, error)
    }
}

/// How a Whisper checkpoint is read: by the layout its resource's `kind`
/// names.
///
/// Chosen by the checkpoint resource's `kind` when the hook is made for a
/// map: `OpenAI`'s `.pt` through the `PyTorch` scanner, `transformers`'
/// `model.safetensors`, one file or shards, through the safetensors one.
/// Both scan to the same [`WhisperApiConfig`] and load the same
/// [`Whisper`] model; only the files' names and layout differ.
#[derive(Clone, Debug)]
pub enum WhisperReader {
    /// `OpenAI`'s `.pt`: a `PyTorch` state dict under `model_state_dict`,
    /// with upstream's names. The reader for a checkpoint of no declared
    /// kind, and of any kind starting `pytorch`.
    Pytorch(PytorchWhisperScanner),

    /// `transformers`' `model.safetensors`, one file or shards with an
    /// index, with its names: what a Hugging Face repo serves. The reader
    /// for any kind starting `safetensors`.
    #[cfg(feature = "store_safetensors")]
    Safetensors(SafetensorsWhisperScanner),
}

impl Default for WhisperReader {
    /// Upstream's `.pt` scanner.
    fn default() -> Self {
        Self::Pytorch(PytorchWhisperScanner::new())
    }
}

impl WhisperReader {
    /// The reader for a checkpoint of `kind`: `None` and `pytorch…` are
    /// upstream's `.pt`; `safetensors…` is `transformers`' file, with the
    /// `store_safetensors` feature; anything else has no reader here.
    ///
    /// # Errors
    /// [`Unsupported`](BunsenErrorKind::Unsupported) naming the kind, and the
    /// feature when it is the one missing.
    pub fn for_kind(kind: Option<&str>) -> BunsenResult<Self> {
        match kind {
            None => Ok(Self::default()),
            Some(kind) if kind.starts_with("pytorch") => Ok(Self::default()),
            #[cfg(feature = "store_safetensors")]
            Some(kind) if kind.starts_with(SAFETENSORS) => {
                Ok(Self::Safetensors(SafetensorsWhisperScanner::new()))
            }
            #[cfg(not(feature = "store_safetensors"))]
            Some(kind) if kind.starts_with(SAFETENSORS) => Err(BunsenError::unsupported(format!(
                "no reader for a {kind:?} checkpoint without the `store_safetensors` feature"
            ))),
            Some(kind) => Err(BunsenError::unsupported(format!(
                "no reader for a {kind:?} checkpoint; the Whisper kit reads PyTorch and safetensors checkpoints"
            ))),
        }
    }

    /// Scans the checkpoint in `loaded`, under [`CHECKPOINT`], for its
    /// config without loading its weights.
    ///
    /// # Errors
    /// [`Lookup`](BunsenErrorKind::Lookup) when `loaded` has no checkpoint;
    /// the scanner's for files that are not a checkpoint of its layout
    /// ([`InvalidResource`](BunsenErrorKind::InvalidResource), for the most
    /// part).
    pub fn scan_cfg(
        &self,
        loaded: &LoadedResources,
    ) -> BunsenResult<WhisperApiConfig> {
        match self {
            Self::Pytorch(scanner) => Ok(scanner.scan_cfg(loaded.expect(CHECKPOINT)?)?.1),
            #[cfg(feature = "store_safetensors")]
            Self::Safetensors(scanner) => {
                scanner.scan_cfg(&SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?)
            }
        }
    }

    /// Loads the checkpoint in `loaded` into a model at the precision it
    /// ships in.
    ///
    /// # Errors
    /// As [`scan_cfg`](Self::scan_cfg); the scanner's.
    pub fn load<B: Backend>(
        &self,
        loaded: &LoadedResources,
        device: &B::Device,
    ) -> BunsenResult<(Whisper<B>, WhisperApiConfig)> {
        match self {
            Self::Pytorch(scanner) => scanner.load::<B, _>(loaded.expect(CHECKPOINT)?, device),
            #[cfg(feature = "store_safetensors")]
            Self::Safetensors(scanner) => scanner.load::<B>(
                &SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?,
                device,
            ),
        }
    }
}

/// How a Whisper pretrained is built: the reader for its checkpoint, and
/// the geometry the checkpoint must have.
///
/// The [`Construct`] hook for the Whisper kit: from a resolved model to a
/// [`WhisperBundle`]. [`plan`](Construct::plan) scans the checkpoint,
/// checks it against the geometry the model promises, and applies the
/// vocabulary rule, all before anything but the checkpoint is brought
/// local; [`construct`](Construct::construct) reads the checkpoint through
/// the reader and the vocabulary through the rank parser, and checks that
/// they agree. [`scan`](Self::scan) is the read-only half, for a listing
/// that wants the geometry without the weights. The checkpoint is the
/// map's [family](ResourceMap::family) under [`CHECKPOINT`]: one resource,
/// or an index and its shards.
///
/// [`for_map`](Construct::for_map) chooses the hook by the checkpoint
/// resource's `kind`, through [`WhisperReader::for_kind`]: a `PyTorch`
/// row gets the `.pt` scanner, a Hugging Face row the safetensors one, and
/// a row of a kind neither reads is refused by name rather than misread. A
/// caller gets this with the deferred model the factory resolves and never
/// builds it.
#[derive(Clone, Debug)]
pub struct WhisperConstruct {
    /// How the checkpoint is read: which layout, and what the reader
    /// declares that the file does not record (the head size, the front
    /// end and the token layout).
    pub reader: WhisperReader,

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
    /// Upstream's `.pt` scanner, expecting what the model promises.
    pub fn new() -> Self {
        Self {
            reader: WhisperReader::default(),
            expected: None,
        }
    }

    /// Sets the reader.
    pub fn with_reader(
        mut self,
        reader: WhisperReader,
    ) -> Self {
        self.reader = reader;
        self
    }

    /// Reads through `scanner`: a `.pt` with another head size, say.
    pub fn with_scanner(
        self,
        scanner: PytorchWhisperScanner,
    ) -> Self {
        self.with_reader(WhisperReader::Pytorch(scanner))
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

    /// Scans the checkpoint in `loaded` for its config without loading its
    /// weights, and checks it against the geometry `model` promises.
    ///
    /// # Errors
    /// As [`WhisperReader::scan_cfg`];
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource), with a
    /// [`GeometryMismatch`] cause naming both geometries, when the files at
    /// that name are not the model it claims to be.
    pub fn scan(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
    ) -> BunsenResult<WhisperApiConfig> {
        let cfg = self.reader.scan_cfg(loaded)?;
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
            Some(expected) if expected != *found => {
                Err(GeometryMismatch::new(model.id(), expected, *found).into())
            }
            _ => Ok(()),
        }
    }
}

impl Construct for WhisperConstruct {
    type Built<B: Backend> = WhisperBundle<B>;

    const KIT: &'static str = WHISPER_KIT;

    /// The reader the checkpoint's `kind` names, through
    /// [`WhisperReader::for_kind`], expecting what the model promises. The
    /// checkpoint is the map's family under [`CHECKPOINT`]: one resource,
    /// or an index and shards, whose first resource's kind speaks for
    /// them. A checkpoint of a kind with no reader is refused by name.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
        let family = map.family(CHECKPOINT);
        let Some(first) = family.resources.values().next() else {
            map.try_get(CHECKPOINT)?;
            unreachable!("an empty family is a missing key");
        };
        let reader = WhisperReader::for_kind(first.kind.as_deref()).context(&first.key)?;
        Ok(Self::new().with_reader(reader))
    }

    /// Scans the checkpoint, which comes local for it (every shard, when
    /// it is sharded), checks its geometry against the promised one, and
    /// settles the vocabulary: a map without one gets the one the
    /// checkpoint's layout selects; a map that declares one must declare
    /// that file, by name and digest, from wherever it serves it, unless
    /// the caller gave it, which is trusted.
    ///
    /// A vocabulary size that is not a Whisper layout is the checkpoint's
    /// fault, not the code's: it comes back as
    /// [`Policy`](BunsenErrorKind::Policy), under a frame naming the model.
    /// A declared vocabulary the checkpoint does not select is
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource).
    fn plan(
        &self,
        model: &PretrainedRef,
        cache: &PretrainedCache,
    ) -> BunsenResult<ResourceMap> {
        let map = model.to_map();
        let checkpoint = map.family(CHECKPOINT);
        if checkpoint.is_empty() {
            map.try_get(CHECKPOINT)?;
        }
        let loaded = cache.load(WHISPER_KIT, &checkpoint)?;
        let cfg = self.scan(model, &loaded)?;

        let ids = *cfg
            .token_layout
            .policy_for_vocab(cfg.vocab_size)
            .as_policy()
            .with_context(|| model.id())?
            .ids();
        let rule = WhisperVocabulary::for_layout(&ids).map().to_map();
        match map.get(VOCABULARY).cloned() {
            None => map.fuse(rule, Fuse::Strict),
            Some(given) if given.namespace == GIVEN_NAMESPACE => Ok(map),
            Some(declared) => {
                // The same file, by name and digest: where a row's copy is
                // served from (a bundle, a mirror) is its own business.
                let selected = rule.try_get(VOCABULARY)?;
                if declared.file != selected.file || declared.sha256 != selected.sha256 {
                    return Err(BunsenError::invalid_resource(format!(
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
    ///
    /// A layout the checkpoint and the vocabulary disagree on is their
    /// fault, not the code's: it comes back as
    /// [`Policy`](BunsenErrorKind::Policy), under a frame naming the model.
    fn construct<B: Backend>(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &B::Device,
    ) -> BunsenResult<Arc<WhisperBundle<B>>> {
        let (whisper, cfg) = self.reader.load::<B>(loaded, device)?;
        let layout = cfg
            .token_layout
            .policy_for_vocab(cfg.vocab_size)
            .as_policy()
            .with_context(|| model.id())?;
        let mut bundle = WhisperBundle::new(whisper, layout);
        if let Some(part) = loaded.get(VOCABULARY) {
            bundle = bundle.with_ranks(TiktokenRanks::load(&part.path)?);
        }
        bundle.validate().as_policy().with_context(|| model.id())?;
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod reader_tests {
    use super::*;
    use crate::{
        data::pretrained::Deferred,
        errors::testing::ErrorMatcher,
    };

    /// The checkpoint's `kind` chooses the reader when the hook is chosen,
    /// before its file is opened: an unlabeled or `PyTorch` one gets the
    /// `.pt` scanner, a safetensors one the safetensors scanner (with its
    /// feature; refused naming the feature without), and a kind neither
    /// reads is refused by name.
    #[test]
    fn test_the_kind_chooses_the_reader() {
        let mut map = ResourceMap::given("mine", CHECKPOINT, "/no/such/model.safetensors");
        let hook = WhisperConstruct::for_map(&map).unwrap();
        assert!(matches!(hook.reader, WhisperReader::Pytorch(_)));

        map.resources.get_mut(CHECKPOINT).unwrap().kind = Some("pytorch fp16".to_string());
        let hook = WhisperConstruct::for_map(&map).unwrap();
        assert!(matches!(hook.reader, WhisperReader::Pytorch(_)));

        map.resources.get_mut(CHECKPOINT).unwrap().kind = Some(SAFETENSORS.to_string());
        #[cfg(feature = "store_safetensors")]
        {
            let hook = WhisperConstruct::for_map(&map).unwrap();
            assert!(matches!(hook.reader, WhisperReader::Safetensors(_)));
            assert!(Deferred::<WhisperConstruct>::from_map(map.clone()).is_ok());
        }
        #[cfg(not(feature = "store_safetensors"))]
        {
            ErrorMatcher::kind(BunsenErrorKind::Unsupported)
                .message_contains("store_safetensors")
                .assert_err(&WhisperConstruct::for_map(&map));
        }

        // A sharded family under the key: its first resource speaks.
        let mut sharded = ResourceMap::new("sharded");
        for key in ["checkpoint.00001", "checkpoint.00002", "checkpoint.index"] {
            let mut r = crate::data::pretrained::Resource::given(key, format!("/no/such/{key}"));
            r.kind = Some(if key.ends_with("index") {
                crate::data::pretrained::SAFETENSORS_INDEX.to_string()
            } else {
                SAFETENSORS.to_string()
            });
            sharded.insert(r);
        }
        #[cfg(feature = "store_safetensors")]
        assert!(matches!(
            WhisperConstruct::for_map(&sharded).unwrap().reader,
            WhisperReader::Safetensors(_)
        ));
        #[cfg(not(feature = "store_safetensors"))]
        assert!(WhisperConstruct::for_map(&sharded).is_err());

        map.resources.get_mut(CHECKPOINT).unwrap().kind = Some("burnpack".to_string());
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .frame_contains("checkpoint")
            .display_contains("checkpoint: no reader for a \"burnpack\"")
            .assert_err(&WhisperConstruct::for_map(&map));
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .assert_err(&Deferred::<WhisperConstruct>::from_map(map));

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .assert_err(&WhisperConstruct::for_map(&ResourceMap::new("empty")));
    }
}

/// Against the bundle's directory: the whole pathway on real files.
#[cfg(all(test, feature = "whisper-weights"))]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        data::pretrained::Provenance,
        errors::testing::{
            ErrorMatcher,
            predicate,
        },
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
            performance_device,
        },
    };

    fn resolve_model(spec: &str) -> BunsenResult<PretrainedRef> {
        Ok(default_whisper_factory()?
            .resolve(spec, &offline_cache())?
            .model)
    }

    /// A checkpoint on disk, as loaded resources: used in place.
    fn loaded(path: &std::path::Path) -> LoadedResources {
        offline_cache()
            .load(WHISPER_KIT, &ResourceMap::given("given", CHECKPOINT, path))
            .unwrap()
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
            .load::<PerformanceBackend>("openai/base", &cache, &performance_device())
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
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("gpt2.tiktoken")
            .message_contains("multilingual.tiktoken")
            .assert_err(&WhisperConstruct::new().plan(&PretrainedRef::from(wrong), &cache));

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

    /// The bundled row declares the same vocabulary file as the rule, served
    /// from the bundle rather than upstream: the plan accepts it as it
    /// stands, and nothing is written to a cache that has nothing.
    #[test]
    fn test_plan_accepts_a_bundled_rows_vocabulary() {
        use crate::data::{
            cache::BunsenDiskCacheOptions,
            pretrained::{
                PretrainedCacheOptions,
                Source,
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
        let model = default_whisper_factory()
            .unwrap()
            .resolve("bundled:openai/base", &cache)
            .unwrap()
            .model;
        let planned = WhisperConstruct::new().plan(&model, &cache).unwrap();
        assert_eq!(planned.keys(), [CHECKPOINT, VOCABULARY]);
        let vocabulary = planned.get(VOCABULARY).unwrap();
        assert_eq!(vocabulary.file, "multilingual.tiktoken");
        assert!(
            matches!(vocabulary.sources.as_slice(), [Source::LocalDir { name, .. }] if name == "bundled"),
            "the row's own sources stand: {:?}",
            vocabulary.sources
        );
        assert!(!dir.path().join("cache").exists(), "nothing was written");
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
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("promised")
            .cause(predicate(
                "tiny expected, base found",
                |m: &GeometryMismatch| {
                    m.expected == geometry_of("tiny") && m.found == geometry_of("base")
                },
            ))
            .assert_err(&tiny.plan(&given, &cache));

        // An explicit expectation overrides a wrong promise: `base.pt`
        // under `openai/tiny`, told to expect base, scans.
        let wrong_name = resolve_model("openai/tiny").unwrap();
        let base = loaded(base_pt);
        assert!(hook.scan(&wrong_name, &base).is_err());
        let explicit = hook.clone().with_expected(Some(geometry_of("base")));
        assert_eq!(
            explicit.expected_for(&wrong_name),
            Some(geometry_of("base"))
        );
        explicit.scan(&wrong_name, &base).unwrap();
    }

    /// The bundled checkpoint is `openai/base`; scanning it must agree with
    /// the `base` prefab, which is what pins the prefab table to a real
    /// file.
    #[test]
    fn test_the_bundled_base_scans_as_the_base_prefab() {
        let model = resolve_model("openai/base").unwrap();
        let cfg = WhisperConstruct::new()
            .scan(&model, &loaded(bunsen_bundled_whisper::base_pt()))
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
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .has_cause::<GeometryMismatch>()
            .assert_err(&hook.scan(&named, &loaded(base)));

        let given = PretrainedRef::from(ResourceMap::given("base", CHECKPOINT, base));
        let cfg = hook.scan(&given, &loaded(base)).unwrap();
        assert_eq!(cfg.geometry().d_head, 32);
        assert_eq!(cfg.geometry().n_heads(), 16);
    }

    /// `base.pt` scanned as if it were `openai/tiny`: same file, wrong
    /// promise.
    #[test]
    fn test_a_checkpoint_under_the_wrong_name_is_rejected() {
        let model = resolve_model("openai/tiny").unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .cause(predicate(
                "a mismatch under openai/tiny",
                |m: &GeometryMismatch| m.model.contains("openai/tiny"),
            ))
            .display_contains("openai/tiny")
            .assert_err(
                &WhisperConstruct::new().scan(&model, &loaded(bunsen_bundled_whisper::base_pt())),
            );
    }
}
