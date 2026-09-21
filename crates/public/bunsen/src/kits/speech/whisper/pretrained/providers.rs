//! # Whisper pretrained providers
//!
//! Which models exist, under which names, made of which resource maps. A
//! provider is a namespace, the `openai` in `openai/tiny.en`, over rows
//! that each name their prefab in [`WHISPER_PREFABS`](super::WHISPER_PREFABS)
//! and fuse a checkpoint map with the vocabulary map its token layout
//! selects: [`OPENAI_CHECKPOINTS`](super::OPENAI_CHECKPOINTS) and
//! [`vocabulary_map`](super::vocabulary_map).
//!
//! The `openai` table is `whisper/__init__.py`'s `_MODELS`, with upstream's
//! two aliases (`large`, `turbo`) folded onto the rows they name rather
//! than repeated. Every row declares the vocabulary the rule would select
//! for its prefab's layout; the tests pin that the declaration and the
//! rule agree, so a path model, which has only the rule, gets the same
//! file a name does.

use std::path::PathBuf;

use crate::{
    data::pretrained::{
        StaticPretrained,
        StaticPretrainedProvider,
        StaticResourceMap,
    },
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
pub static OPENAI: StaticPretrainedProvider<'static> = StaticPretrainedProvider {
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

/// Every Whisper provider, in lookup order.
pub static WHISPER_PROVIDERS: &[&StaticPretrainedProvider<'static>] = &[&OPENAI];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::{
            lookup_pretrained,
            pretrained_for_prefab,
        },
        kits::speech::whisper::{
            driver::WhisperSpecialIds,
            pretrained::{
                CHECKPOINT,
                VOCABULARY,
                WHISPER_PREFABS,
                vocabulary_map,
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
        for provider in WHISPER_PROVIDERS {
            for row in provider.items {
                let prefab = row.prefab.expect("every whisper row names a prefab");
                assert!(
                    WHISPER_PREFABS.lookup_prefab(prefab).is_some(),
                    "{}: prefab {prefab:?} is not in WHISPER_PREFABS",
                    provider.id(row),
                );
                let map = row.try_to_map().unwrap();
                map.validate().unwrap();
                assert_eq!(map.keys(), [CHECKPOINT, VOCABULARY], "{}", provider.id(row));
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
            let rule = vocabulary_map(&layout_of(row.prefab.unwrap())).to_map();
            let rule = rule.get(VOCABULARY).unwrap();
            assert_eq!(declared, rule, "{}", OPENAI.id(row));
        }
    }

    #[test]
    fn test_names_and_aliases_are_unique_within_a_provider() {
        for provider in WHISPER_PROVIDERS {
            let mut names: Vec<&str> = provider
                .items
                .iter()
                .flat_map(|p| std::iter::once(p.name).chain(p.aliases.iter().copied()))
                .collect();
            let n = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), n, "{}: a name is repeated", provider.name);
        }
    }

    #[test]
    fn test_upstream_aliases() {
        let (provider, large) = lookup_pretrained(WHISPER_PROVIDERS, None, "large").unwrap();
        assert_eq!(provider.name, "openai");
        assert_eq!(large.name, "large-v3");

        let (_, turbo) = lookup_pretrained(WHISPER_PROVIDERS, Some("openai"), "turbo").unwrap();
        assert_eq!(turbo.name, "large-v3-turbo");

        assert!(lookup_pretrained(WHISPER_PROVIDERS, Some("nobody"), "base").is_none());
        assert!(lookup_pretrained(WHISPER_PROVIDERS, None, "gigantic").is_none());
    }

    #[test]
    fn test_one_prefab_many_pretrained() {
        let large: Vec<&str> = OPENAI.for_prefab("large").iter().map(|p| p.name).collect();
        assert_eq!(large, ["large-v1", "large-v2"]);
        let derived: Vec<String> = pretrained_for_prefab(WHISPER_PROVIDERS, "large")
            .iter()
            .map(|(p, d)| p.id(d))
            .collect();
        assert_eq!(derived, ["openai/large-v1", "openai/large-v2"]);
        assert!(openai_download_root().is_some_and(|d| d.ends_with("whisper")));
    }

    /// A cache rooted at the bundle's directory, offline, has `openai/base`
    /// whole: both resources cached, nothing fetched, nothing written.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_the_bundle_serves_openai_base() {
        use crate::{
            data::pretrained::{
                CacheStatus,
                PretrainedRef,
                Provenance,
            },
            kits::speech::whisper::pretrained::testing::offline_cache,
        };
        let cache = offline_cache();
        let model = PretrainedRef::resolve(WHISPER_PROVIDERS, "openai/base", CHECKPOINT).unwrap();
        assert_eq!(model.id(), "openai/base");
        let status = model.status(WHISPER_KIT, &cache);
        assert_eq!(status[CHECKPOINT], CacheStatus::Cached);
        assert_eq!(status[VOCABULARY], CacheStatus::Cached);

        let loaded = cache.load(WHISPER_KIT, &model.to_map()).unwrap();
        assert_eq!(loaded.map.name, "openai/base");
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
