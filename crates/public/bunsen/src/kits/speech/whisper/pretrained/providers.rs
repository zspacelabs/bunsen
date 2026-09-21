//! # Whisper pretrained providers
//!
//! Which models exist, under which names, made of which resource maps.
//! [`default_whisper_factory`] is the index a caller holds; its one
//! compiled-in provider is [`WELL_KNOWN_TABLE`], whose refs are
//! `{group}/{name}`. A group is a label, the `openai` in `openai/tiny.en`,
//! over rows that each name their prefab in
//! [`WHISPER_PREFABS`](super::WHISPER_PREFABS) and fuse a checkpoint map
//! with the vocabulary map its token layout selects:
//! [`OPENAI_CHECKPOINTS`](super::OPENAI_CHECKPOINTS) and
//! [`WhisperVocabulary`](super::WhisperVocabulary).
//!
//! A caller with a provider of its own, a hub say, builds its own factory:
//!
//! ```rust,ignore
//! let factory = Arc::new(
//!     PretrainedFactory::new(WHISPER_KIT)
//!         .with_providers(default_whisper_providers())?
//!         .with_provider(Arc::new(my_hub))?, // answers `hub:org/repo`, lists nothing
//! );
//! ```
//!
//! The `openai` table is `whisper/__init__.py`'s `_MODELS`, with upstream's
//! two aliases (`large`, `turbo`) folded onto the rows they name rather
//! than repeated. Every row declares the vocabulary the rule would select
//! for its prefab's layout; the tests pin that the declaration and the
//! rule agree, so a path model, which has only the rule, gets the same
//! file a name does.

use std::{
    path::PathBuf,
    sync::Arc,
};

use crate::{
    data::pretrained::{
        PretrainedFactory,
        PretrainedProvider,
        StaticPretrained,
        StaticPretrainedGroup,
        StaticPretrainedTable,
        StaticResourceMap,
        WELL_KNOWN,
    },
    errors::BunsenResult,
    kits::speech::whisper::pretrained::{
        BASE_CHECKPOINT,
        BASE_EN_CHECKPOINT,
        GPT2_VOCABULARY,
        LARGE_V1_CHECKPOINT,
        LARGE_V2_CHECKPOINT,
        LARGE_V3_CHECKPOINT,
        LARGE_V3_TURBO_CHECKPOINT,
        MEDIUM_CHECKPOINT,
        MEDIUM_EN_CHECKPOINT,
        MULTILINGUAL_VOCABULARY,
        SMALL_CHECKPOINT,
        SMALL_EN_CHECKPOINT,
        TINY_CHECKPOINT,
        TINY_EN_CHECKPOINT,
    },
};

/// The kit segment of a Whisper resource's path in the cache:
/// `<cache>/pretrained/whisper/<namespace>/<sha256>/<file>`.
pub const WHISPER_KIT: &str = "whisper";

/// The name of the local-dir base that is `openai-whisper`'s download
/// root; a [`PretrainedCacheOptions`](crate::data::pretrained::PretrainedCacheOptions)
/// override by this name points it elsewhere.
pub const OPENAI_LOCAL_DIR: &str = "openai-whisper";

/// `openai-whisper`'s default `download_root`: `$XDG_CACHE_HOME/whisper`, or
/// `~/.cache/whisper`. Upstream uses `~/.cache` on every platform, so this
/// does not ask the platform for its cache directory.
pub fn openai_download_root() -> Option<PathBuf> {
    let cache_home = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::home_dir().map(|h| h.join(".cache")))?;
    Some(cache_home.join("whisper"))
}

/// One `openai` row: a checkpoint map and the vocabulary map its layout
/// selects, naming its prefab.
const fn openai(
    name: &'static str,
    aliases: &'static [&'static str],
    prefab: &'static str,
    description: &'static str,
    maps: &'static [&'static StaticResourceMap<'static>],
) -> StaticPretrained<'static> {
    StaticPretrained {
        name,
        aliases,
        description,
        license: Some("MIT"),
        origin: Some("https://github.com/openai/whisper/blob/main/whisper/__init__.py"),
        prefab: Some(prefab),
        maps,
    }
}

static TINY_EN: StaticPretrained<'static> = openai(
    "tiny.en",
    &[],
    "tiny.en",
    "39 M parameters, English-only",
    &[&TINY_EN_CHECKPOINT, &GPT2_VOCABULARY],
);

static TINY: StaticPretrained<'static> = openai(
    "tiny",
    &[],
    "tiny",
    "39 M parameters, multilingual",
    &[&TINY_CHECKPOINT, &MULTILINGUAL_VOCABULARY],
);

static BASE_EN: StaticPretrained<'static> = openai(
    "base.en",
    &[],
    "base.en",
    "74 M parameters, English-only",
    &[&BASE_EN_CHECKPOINT, &GPT2_VOCABULARY],
);

static BASE: StaticPretrained<'static> = openai(
    "base",
    &[],
    "base",
    "74 M parameters, multilingual",
    &[&BASE_CHECKPOINT, &MULTILINGUAL_VOCABULARY],
);

static SMALL_EN: StaticPretrained<'static> = openai(
    "small.en",
    &[],
    "small.en",
    "244 M parameters, English-only",
    &[&SMALL_EN_CHECKPOINT, &GPT2_VOCABULARY],
);

static SMALL: StaticPretrained<'static> = openai(
    "small",
    &[],
    "small",
    "244 M parameters, multilingual",
    &[&SMALL_CHECKPOINT, &MULTILINGUAL_VOCABULARY],
);

static MEDIUM_EN: StaticPretrained<'static> = openai(
    "medium.en",
    &[],
    "medium.en",
    "769 M parameters, English-only",
    &[&MEDIUM_EN_CHECKPOINT, &GPT2_VOCABULARY],
);

static MEDIUM: StaticPretrained<'static> = openai(
    "medium",
    &[],
    "medium",
    "769 M parameters, multilingual",
    &[&MEDIUM_CHECKPOINT, &MULTILINGUAL_VOCABULARY],
);

static LARGE_V1: StaticPretrained<'static> = openai(
    "large-v1",
    &[],
    "large",
    "1550 M parameters, multilingual",
    &[&LARGE_V1_CHECKPOINT, &MULTILINGUAL_VOCABULARY],
);

static LARGE_V2: StaticPretrained<'static> = openai(
    "large-v2",
    &[],
    "large",
    "1550 M parameters, multilingual",
    &[&LARGE_V2_CHECKPOINT, &MULTILINGUAL_VOCABULARY],
);

static LARGE_V3: StaticPretrained<'static> = openai(
    "large-v3",
    &["large"],
    "large-v3",
    "1550 M parameters, multilingual, 128 mels",
    &[&LARGE_V3_CHECKPOINT, &MULTILINGUAL_VOCABULARY],
);

static LARGE_V3_TURBO: StaticPretrained<'static> = openai(
    "large-v3-turbo",
    &["turbo"],
    "large-v3-turbo",
    "809 M parameters, multilingual, 128 mels, four-layer decoder",
    &[&LARGE_V3_TURBO_CHECKPOINT, &MULTILINGUAL_VOCABULARY],
);

/// `OpenAI`'s checkpoints, as `openai-whisper` names and pins them, each
/// with the vocabulary it decodes through.
pub static OPENAI: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "openai",
    description: "OpenAI's Whisper checkpoints, as `openai-whisper` names and pins them",
    license: Some("MIT"),
    origin: Some("https://github.com/openai/whisper/blob/main/whisper/__init__.py"),
    items: &[
        &TINY_EN,
        &TINY,
        &BASE_EN,
        &BASE,
        &SMALL_EN,
        &SMALL,
        &MEDIUM_EN,
        &MEDIUM,
        &LARGE_V1,
        &LARGE_V2,
        &LARGE_V3,
        &LARGE_V3_TURBO,
    ],
};

/// The checkpoints bunsen knows by name, behind the [`WELL_KNOWN`]
/// provider: `well-known:openai/tiny`, or `openai/tiny`, or `tiny`.
pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
    name: WELL_KNOWN,
    description: "the Whisper checkpoints bunsen knows by name",
    groups: &[&OPENAI],
};

/// The rows `bunsen-bundled-whisper` ships, behind the
/// [`BUNDLED`](crate::data::pretrained::BUNDLED) provider:
/// `bundled:openai/base`, the same files as `well-known:openai/base`,
/// served in place from the bundle's build directory rather than fetched.
///
/// # Panics
/// If the `base` row no longer has the resources the bundle ships; the
/// tests pin that it does.
#[cfg(feature = "whisper-weights")]
pub fn bundled_whisper_table() -> crate::data::pretrained::PretrainedTable {
    use crate::{
        data::pretrained::{
            BUNDLED,
            PretrainedGroup,
            PretrainedTable,
        },
        kits::speech::whisper::pretrained::{
            CHECKPOINT,
            VOCABULARY,
        },
    };

    let mut base = BASE.to_pretrained();
    base.resources = base
        .resources
        .with_local_files(
            BUNDLED,
            &[
                (CHECKPOINT, bunsen_bundled_whisper::base_pt()),
                (VOCABULARY, bunsen_bundled_whisper::multilingual_tiktoken()),
            ],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    PretrainedTable {
        name: BUNDLED.to_string(),
        description: "the Whisper assets bunsen-bundled-whisper ships, used in place".to_string(),
        groups: vec![PretrainedGroup {
            name: OPENAI.name.to_string(),
            description: OPENAI.description.to_string(),
            license: OPENAI.license.map(str::to_string),
            origin: OPENAI.origin.map(str::to_string),
            items: vec![base],
        }],
    }
}

/// Whisper's compiled-in providers, in search order: the well-known table,
/// then, with the `whisper-weights` feature, the bundled one.
pub fn default_whisper_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    let well_known: Arc<dyn PretrainedProvider> = Arc::new(WELL_KNOWN_TABLE.to_table());
    #[cfg(feature = "whisper-weights")]
    {
        vec![well_known, Arc::new(bundled_whisper_table())]
    }
    #[cfg(not(feature = "whisper-weights"))]
    {
        vec![well_known]
    }
}

/// Whisper's factory over [`default_whisper_providers`]: the index
/// `--model` is resolved against.
///
/// # Errors
/// [`BunsenError::Invalid`](crate::errors::BunsenError::Invalid) if two of
/// the defaults share a name, which the tests pin they do not.
pub fn default_whisper_factory() -> BunsenResult<Arc<PretrainedFactory>> {
    Ok(Arc::new(
        PretrainedFactory::new(WHISPER_KIT).with_providers(default_whisper_providers())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::testing::ListsNothing,
        errors::BunsenError,
        kits::speech::whisper::{
            driver::WhisperSpecialIds,
            pretrained::{
                CHECKPOINT,
                VOCABULARY,
                WHISPER_PREFABS,
                WhisperVocabulary,
            },
        },
    };

    fn layout_of(prefab: &str) -> WhisperSpecialIds {
        let cfg = WHISPER_PREFABS.expect_lookup_prefab(prefab).to_config();
        *cfg.token_layout
            .policy_for_vocab(cfg.vocab_size)
            .unwrap()
            .ids()
    }

    /// Every row names a prefab the map has and fuses into a checkpoint
    /// and a vocabulary, the checkpoint file named after the row and
    /// pinned.
    #[test]
    fn test_every_row_names_a_prefab_and_fuses() {
        for group in WELL_KNOWN_TABLE.groups {
            for row in group.items {
                let prefab = row.prefab.expect("every whisper row names a prefab");
                assert!(
                    WHISPER_PREFABS.lookup_prefab(prefab).is_some(),
                    "{}: prefab {prefab:?} is not in WHISPER_PREFABS",
                    group.id(row),
                );
                let map = row.try_to_map().unwrap();
                map.validate().unwrap();
                assert_eq!(map.keys(), [CHECKPOINT, VOCABULARY], "{}", group.id(row));
                let checkpoint = map.get(CHECKPOINT).unwrap();
                assert_eq!(checkpoint.file, format!("{}.pt", row.name));
                assert!(checkpoint.is_pinned());
                assert!(map.get(VOCABULARY).unwrap().is_pinned());
            }
        }
        assert_eq!(OPENAI.items.len(), 12);
    }

    /// The vocabulary a row declares is the one the rule selects for its
    /// prefab's layout, so a name and a path agree on it.
    #[test]
    fn test_the_declared_vocabulary_is_the_rules() {
        for row in OPENAI.items {
            let declared = row.to_map();
            let declared = declared.get(VOCABULARY).unwrap();
            let rule = WhisperVocabulary::for_layout(&layout_of(row.prefab.unwrap()))
                .map()
                .to_map();
            let rule = rule.get(VOCABULARY).unwrap();
            assert_eq!(declared, rule, "{}", OPENAI.id(row));
        }
    }

    /// The well-known table lists the twelve `openai` rows by ref and
    /// answers the refs, the bare names and upstream's aliases.
    #[test]
    fn test_the_well_known_table_lists_the_openai_rows() {
        let table = WELL_KNOWN_TABLE.to_table();
        assert_eq!(table.name(), "well-known");
        let ids = table.ids();
        assert_eq!(ids.len(), 12);
        assert_eq!(ids[0], "well-known:openai/tiny.en");
        assert_eq!(ids[11], "well-known:openai/large-v3-turbo");
        let name = |spec: &str| table.lookup(spec).unwrap().map(|p| p.name);
        assert_eq!(name("openai/tiny.en").as_deref(), Some("openai/tiny.en"));
        assert_eq!(
            name("openai/turbo").as_deref(),
            Some("openai/large-v3-turbo")
        );
        assert_eq!(name("turbo").as_deref(), Some("openai/large-v3-turbo"));
        assert_eq!(name("large").as_deref(), Some("openai/large-v3"));
        assert_eq!(
            name("large-v3-turbo").as_deref(),
            Some("openai/large-v3-turbo")
        );
        assert_eq!(name("openai/gigantic"), None);
        assert_eq!(name("gigantic"), None);
    }

    #[test]
    fn test_names_and_aliases_are_unique_within_a_group() {
        for group in WELL_KNOWN_TABLE.groups {
            let mut names: Vec<&str> = group
                .items
                .iter()
                .flat_map(|p| std::iter::once(p.name).chain(p.aliases.iter().copied()))
                .collect();
            let n = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), n, "{}: a name is repeated", group.name);
        }
    }

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
    }

    /// A spec resolves to a row, by ref, bare name or alias, or to a path
    /// under the checkpoint key.
    #[test]
    fn test_resolve_names_aliases_and_paths() {
        use crate::data::pretrained::PretrainedRef;
        let factory = default_whisper_factory().unwrap();
        let resolve = |spec: &str| factory.resolve(spec, Some(CHECKPOINT));

        match resolve("openai/tiny.en").unwrap() {
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
            resolve("well-known:openai/tiny.en").unwrap().id(),
            "well-known:openai/tiny.en"
        );
        assert_eq!(
            resolve("turbo").unwrap().id(),
            "well-known:openai/large-v3-turbo"
        );

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("ckpt.pt");
        std::fs::write(&file, b"x").unwrap();
        match resolve(file.to_str().unwrap()).unwrap() {
            PretrainedRef::Given(map) => {
                assert_eq!(map.keys(), [CHECKPOINT]);
                assert_eq!(map.get(CHECKPOINT).unwrap().file, "ckpt.pt");
            }
            other => panic!("{other:?}"),
        }

        assert!(matches!(
            resolve("openai/gigantic"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            resolve("/no/such/file.pt"),
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
    /// qualified, the bare names still only reach the table, and the
    /// listing is unchanged.
    #[test]
    fn test_a_plugin_joins_the_defaults() {
        let hub = Arc::new(ListsNothing::default());
        let factory = PretrainedFactory::new(WHISPER_KIT)
            .with_providers(default_whisper_providers())
            .unwrap()
            .with_provider(hub.clone())
            .unwrap();
        let defaults = default_whisper_providers().len();
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
        let err = PretrainedFactory::new(WHISPER_KIT)
            .with_providers(default_whisper_providers())
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
        use crate::data::{
            cache::BunsenDiskCacheOptions,
            pretrained::{
                BUNDLED,
                CacheStatus,
                PretrainedCache,
                PretrainedCacheOptions,
                Provenance,
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

        let model = factory
            .resolve("bundled:openai/base", Some(CHECKPOINT))
            .unwrap();
        assert_eq!(model.id(), "bundled:openai/base");
        let status = model.status(WHISPER_KIT, &cache);
        assert_eq!(status[CHECKPOINT], CacheStatus::LocalDir);
        assert_eq!(status[VOCABULARY], CacheStatus::LocalDir);
        let map = model.to_map();
        assert_eq!(
            map.get(CHECKPOINT).unwrap().sha256,
            BASE.to_map().get(CHECKPOINT).unwrap().sha256,
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
            factory
                .resolve("openai/base", Some(CHECKPOINT))
                .unwrap()
                .id(),
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
                WhisperConstruct,
                testing::offline_cache,
            },
        };
        let cache = offline_cache();
        let factory = default_whisper_factory().unwrap();
        let model = factory
            .resolve_for::<WhisperConstruct>("openai/base")
            .unwrap();
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
