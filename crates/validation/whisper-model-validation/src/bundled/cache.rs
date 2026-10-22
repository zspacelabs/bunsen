//! The bundle's directory as a pretrained cache, and its files as a table.

use bunsen::{
    data::pretrained::{
        BUNDLED,
        CacheStatus,
        Fuse,
        Provenance,
    },
    kits::speech::whisper::pretrained::{
        BASE_CHECKPOINT,
        CHECKPOINT,
        GPT2_VOCABULARY,
        MULTILINGUAL_VOCABULARY,
        PYTORCH_FP16,
        TIKTOKEN,
        VOCABULARY,
        WHISPER_KIT,
        default_whisper_factory,
    },
};

use super::{
    cache_in,
    factory_with_bundle,
    offline_cache,
};

/// The bundle's directory is a cache of exactly these maps: pointed at
/// it, offline, the cache finds `openai/base` and both vocabularies
/// cached, at paths under it, and writes nothing.
#[test]
fn test_the_bundle_is_a_cache_of_these_maps() {
    let root = bunsen_bundled_whisper::cache_dir();
    let cache = offline_cache();

    for map in [&BASE_CHECKPOINT, &MULTILINGUAL_VOCABULARY, &GPT2_VOCABULARY] {
        let owned = map.to_map();
        for (key, r) in &owned.resources {
            assert_eq!(cache.status(WHISPER_KIT, r), CacheStatus::Cached, "{key}");
            let resolved = cache.resolve(WHISPER_KIT, r).unwrap();
            assert_eq!(resolved.provenance, Provenance::Cached, "{key}");
            assert!(
                resolved.path.starts_with(root),
                "{}",
                resolved.path.display()
            );
            assert!(resolved.path.is_file(), "{}", resolved.path.display());
        }
    }

    let base = BASE_CHECKPOINT
        .to_map()
        .fuse(MULTILINGUAL_VOCABULARY.to_map(), Fuse::Strict)
        .unwrap();
    let loaded = cache.load(WHISPER_KIT, &base).unwrap();
    assert_eq!(loaded.keys(), [CHECKPOINT, VOCABULARY]);
    for (_, part) in loaded.iter() {
        assert_eq!(part.provenance, Provenance::Cached);
    }
    assert_eq!(loaded.kind(CHECKPOINT), Some(PYTORCH_FP16));
    assert_eq!(loaded.kind(VOCABULARY), Some(TIKTOKEN));
    assert!(
        !root
            .join("pretrained")
            .join(WHISPER_KIT)
            .join("given")
            .exists(),
        "nothing was written"
    );
}

/// `bundled:openai/base` (`bundled_table`) is the bundle's files, used in place
/// from a cache that has nothing: both parts local dir, nothing written; and
/// a bare `openai/base` still means the well-known row.
#[test]
fn test_the_bundled_provider_serves_openai_base_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let cache = cache_in(dir.path().join("cache"));
    let factory = factory_with_bundle();
    let bundled = factory.provider(BUNDLED).unwrap();
    assert_eq!(bundled.ids(), ["bundled:openai/base"]);
    assert_eq!(bundled.license(), None);

    let model = factory.resolve("bundled:openai/base", &cache).unwrap();
    assert_eq!(model.id(), "bundled:openai/base");
    let status = model.status(&cache);
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
        factory.resolve("openai/base", &cache).unwrap().id(),
        "well-known:openai/base"
    );
}

/// A cache rooted at the bundle's directory, offline, has `openai/base`
/// whole: both resources cached, nothing fetched, nothing written.
#[test]
fn test_the_bundle_serves_openai_base() {
    let cache = offline_cache();
    let factory = default_whisper_factory().unwrap();
    let model = factory.resolve("openai/base", &cache).unwrap();
    assert_eq!(model.id(), "well-known:openai/base");
    let status = model.status(&cache);
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
