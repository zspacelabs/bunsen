//! # Whisper pretrained providers
//!
//! Which weights exist, for which prefab, and where. A provider is a
//! namespace, the `openai` in `openai/tiny.en`, over entries that each name
//! their prefab in [`WHISPER_PREFABS`](super::WHISPER_PREFABS), their format,
//! and the sources their bytes can be had from, all pinned to one SHA-256.
//!
//! The `openai` table is `whisper/__init__.py`'s `_MODELS`, with upstream's
//! two aliases (`large`, `turbo`) folded onto the entries they name rather
//! than repeated, and the digest, which upstream embeds in the URL and
//! checks on every load, lifted out where every source can share it.

use std::path::PathBuf;

use crate::data::pretrained::{
    StaticPretrainedProvider,
    StaticPretrainedWeightsDescriptor,
    StaticWeightsSource,
    WeightsFormat,
};

/// The kit segment of a Whisper weights path in the cache:
/// `<cache>/weights/whisper/<provider>/<sha256>/<file>`.
pub const WHISPER_KIT: &str = "whisper";

/// The name of the local-dir source that is `openai-whisper`'s download
/// root; a [`WeightsCacheOptions`](crate::data::pretrained::WeightsCacheOptions)
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

/// Upstream's download cache, read as a local source. Upstream keeps the
/// file under its bare name and re-hashes it on every load; here it is
/// hashed once and linked into the cache.
const UPSTREAM: StaticWeightsSource<'static> = StaticWeightsSource::LocalDir {
    name: OPENAI_LOCAL_DIR,
    default: openai_download_root,
};

/// One `openai` entry: fp16 `PyTorch`, pinned to the digest upstream embeds in
/// the URL, from upstream's cache, then upstream's URL.
const fn openai(
    name: &'static str,
    aliases: &'static [&'static str],
    prefab: &'static str,
    description: &'static str,
    file: &'static str,
    sha256: &'static str,
    sources: &'static [StaticWeightsSource<'static>],
) -> StaticPretrainedWeightsDescriptor<'static> {
    StaticPretrainedWeightsDescriptor {
        name,
        description,
        license: Some("MIT"),
        origin: Some("https://github.com/openai/whisper/blob/main/whisper/__init__.py"),
        prefab,
        aliases,
        file,
        sha256: Some(sha256),
        format: WeightsFormat::PYTORCH_F16,
        sources,
    }
}

static TINY_EN: StaticPretrainedWeightsDescriptor<'static> = openai(
    "tiny.en",
    &[],
    "tiny.en",
    "39 M parameters, English-only",
    "tiny.en.pt",
    "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
        ),
    ],
);

static TINY: StaticPretrainedWeightsDescriptor<'static> = openai(
    "tiny",
    &[],
    "tiny",
    "39 M parameters, multilingual",
    "tiny.pt",
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        ),
    ],
);

static BASE_EN: StaticPretrainedWeightsDescriptor<'static> = openai(
    "base.en",
    &[],
    "base.en",
    "74 M parameters, English-only",
    "base.en.pt",
    "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
        ),
    ],
);

/// `openai/base`: the checkpoint bunsen bundles, when the `whisper-weights`
/// feature is on, which is then the first source.
static BASE: StaticPretrainedWeightsDescriptor<'static> = openai(
    "base",
    &[],
    "base",
    "74 M parameters, multilingual; the checkpoint bunsen bundles",
    "base.pt",
    "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
    &[
        #[cfg(feature = "whisper-weights")]
        StaticWeightsSource::File(bunsen_bundled_whisper::base_pt),
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        ),
    ],
);

static SMALL_EN: StaticPretrainedWeightsDescriptor<'static> = openai(
    "small.en",
    &[],
    "small.en",
    "244 M parameters, English-only",
    "small.en.pt",
    "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
        ),
    ],
);

static SMALL: StaticPretrainedWeightsDescriptor<'static> = openai(
    "small",
    &[],
    "small",
    "244 M parameters, multilingual",
    "small.pt",
    "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        ),
    ],
);

static MEDIUM_EN: StaticPretrainedWeightsDescriptor<'static> = openai(
    "medium.en",
    &[],
    "medium.en",
    "769 M parameters, English-only",
    "medium.en.pt",
    "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
        ),
    ],
);

static MEDIUM: StaticPretrainedWeightsDescriptor<'static> = openai(
    "medium",
    &[],
    "medium",
    "769 M parameters, multilingual",
    "medium.pt",
    "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        ),
    ],
);

static LARGE_V1: StaticPretrainedWeightsDescriptor<'static> = openai(
    "large-v1",
    &[],
    "large",
    "1550 M parameters, multilingual",
    "large-v1.pt",
    "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
        ),
    ],
);

static LARGE_V2: StaticPretrainedWeightsDescriptor<'static> = openai(
    "large-v2",
    &[],
    "large",
    "1550 M parameters, multilingual",
    "large-v2.pt",
    "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
        ),
    ],
);

static LARGE_V3: StaticPretrainedWeightsDescriptor<'static> = openai(
    "large-v3",
    &["large"],
    "large-v3",
    "1550 M parameters, multilingual, 128 mels",
    "large-v3.pt",
    "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
        ),
    ],
);

static LARGE_V3_TURBO: StaticPretrainedWeightsDescriptor<'static> = openai(
    "large-v3-turbo",
    &["turbo"],
    "large-v3-turbo",
    "809 M parameters, multilingual, 128 mels, four-layer decoder",
    "large-v3-turbo.pt",
    "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
    &[
        UPSTREAM,
        StaticWeightsSource::Url(
            "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
        ),
    ],
);

/// `OpenAI`'s checkpoints, as `openai-whisper` names and pins them.
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
        kits::speech::whisper::pretrained::WHISPER_PREFABS,
    };

    #[test]
    fn test_every_pretrained_names_a_prefab_and_validates() {
        for provider in WHISPER_PROVIDERS {
            for p in provider.items {
                assert!(
                    WHISPER_PREFABS.lookup_prefab(p.prefab).is_some(),
                    "{}: prefab {:?} is not in WHISPER_PREFABS",
                    provider.id(p),
                    p.prefab,
                );
                p.to_descriptor().validate().unwrap();
            }
        }
        assert_eq!(OPENAI.items.len(), 12);
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

    /// `OpenAI`'s URLs are digest-addressed; the pin and the path must agree.
    #[test]
    fn test_digests_pin_the_urls() {
        for provider in WHISPER_PROVIDERS {
            for p in provider.items {
                let d = p.to_descriptor();
                let sha256 = d.sha256.as_deref().expect("every openai entry is pinned");
                assert!(!d.urls().is_empty(), "{}: no URL", provider.id(p));
                for url in d.urls() {
                    assert!(
                        url.ends_with(&format!("/{sha256}/{}", d.file)),
                        "{}: {url} does not end in the digest and file",
                        provider.id(p),
                    );
                }
                assert_eq!(d.format, WeightsFormat::PYTORCH_F16);
            }
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
    }

    /// Every entry lists upstream's download root before its URL, so a file
    /// `openai-whisper` already fetched is found before the network is.
    #[test]
    fn test_sources_prefer_upstreams_cache() {
        for p in OPENAI.items {
            let d = p.to_descriptor();
            let local = d
                .sources
                .iter()
                .position(|s| matches!(s, crate::data::pretrained::WeightsSource::LocalDir { name, .. } if name == OPENAI_LOCAL_DIR))
                .expect("a local dir source");
            let url = d
                .sources
                .iter()
                .position(|s| matches!(s, crate::data::pretrained::WeightsSource::Url(_)))
                .expect("a url source");
            assert!(local < url, "{}", OPENAI.id(p));
        }
        assert!(openai_download_root().is_some_and(|d| d.ends_with("whisper")));
    }

    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_the_bundled_checkpoint_is_the_openai_base() {
        let base = OPENAI.lookup("base").unwrap();
        assert!(
            matches!(base.sources[0], StaticWeightsSource::File(_)),
            "openai/base lists the bundled file first",
        );
        let d = base.to_descriptor();
        match &d.sources[0] {
            crate::data::pretrained::WeightsSource::File(path) => {
                assert_eq!(path.as_path(), bunsen_bundled_whisper::base_pt());
            }
            other => panic!("{other:?}"),
        }
    }

    /// The bundled checkpoint resolves offline, in place, through the cache.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_the_bundled_base_resolves_offline() {
        use crate::data::{
            cache::BunsenDiskCacheOptions,
            pretrained::{
                CacheStatus,
                Provenance,
                WeightsCache,
                WeightsCacheOptions,
            },
        };
        let dir = tempfile::tempdir().unwrap();
        let cache = WeightsCache::new(
            WeightsCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true)
                .with_local_dir(OPENAI_LOCAL_DIR, dir.path().join("upstream")),
        )
        .unwrap();
        let base = OPENAI.lookup("base").unwrap().to_descriptor();

        assert_eq!(
            cache.status(WHISPER_KIT, "openai", &base),
            CacheStatus::File
        );
        let resolved = cache.resolve(WHISPER_KIT, "openai", &base).unwrap();
        assert_eq!(resolved.provenance, Provenance::File);
        assert!(resolved.path.is_file());
        assert!(!dir.path().join("cache").join("weights").exists());
    }
}
