//! The kit's hook on a real checkpoint: plan, scan, construct.

use std::sync::Arc;

use bunsen::{
    data::pretrained::{
        Construct,
        Fuse,
        LoadedResources,
        PretrainedRef,
        Provenance,
        ResourceMap,
        Source,
    },
    errors::{
        BunsenErrorKind,
        BunsenResult,
        testing::{
            ErrorMatcher,
            predicate,
        },
    },
    kits::speech::whisper::{
        WhisperGeometry,
        WhisperMeta,
        pretrained::{
            BASE_CHECKPOINT,
            CHECKPOINT,
            GPT2_VOCABULARY,
            GeometryMismatch,
            PytorchWhisperScanner,
            VOCABULARY,
            WHISPER_KIT,
            WHISPER_PREFABS,
            WhisperConstruct,
            default_whisper_factory,
        },
    },
    support::testing::performance_device,
};
use serial_test::serial;

use super::{
    cache_in,
    factory_with_bundle,
    offline_cache,
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
        .load("openai/base", &cache, &performance_device())
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
        bind_whisper_onnx::base_pt(),
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
            bind_whisper_onnx::gpt2_tiktoken(),
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
    let dir = tempfile::tempdir().unwrap();
    let cache = cache_in(dir.path().join("cache"));
    let model = factory_with_bundle()
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
    let base_pt = bind_whisper_onnx::base_pt();
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
        .scan(&model, &loaded(bind_whisper_onnx::base_pt()))
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
    let base = bind_whisper_onnx::base_pt();
    let hook = WhisperConstruct::new().with_scanner(PytorchWhisperScanner::new().with_d_head(32));

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
        .assert_err(&WhisperConstruct::new().scan(&model, &loaded(bind_whisper_onnx::base_pt())));
}
