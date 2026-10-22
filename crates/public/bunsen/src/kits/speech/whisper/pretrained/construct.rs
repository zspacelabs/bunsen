//! # Constructing a Whisper pretrained

use std::{
    error::Error,
    fmt,
    sync::Arc,
};

use burn::tensor::Device;

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
    pub fn load(
        &self,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<(Whisper, WhisperApiConfig)> {
        match self {
            Self::Pytorch(scanner) => scanner.load::<_>(loaded.expect(CHECKPOINT)?, device),
            #[cfg(feature = "store_safetensors")]
            Self::Safetensors(scanner) => scanner.load(
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
    type Built = WhisperBundle;

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
    fn construct(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<WhisperBundle>> {
        let (whisper, cfg) = self.reader.load(loaded, device)?;
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
