//! The pretrained index: which weights exist, for which prefab, and where.
//!
//! A [`WeightsProvider`] is a namespace &mdash; the `openai` in
//! `openai/tiny.en` &mdash; over [`WhisperPretrained`] entries. Each entry
//! names its [prefab](super::prefab), its [`WeightsFormat`], and the
//! [`WeightsSource`]s its bytes can be had from, all pinned to one SHA-256.
//!
//! The `openai` table is `whisper/__init__.py`'s `_MODELS`, with upstream's
//! two aliases (`large`, `turbo`) folded onto the entries they name rather
//! than repeated, and the digest &mdash; which upstream embeds in the URL and
//! checks on every load &mdash; lifted out where every source can share it.

use std::{
    fmt,
    path::Path,
};

use bunsen::{
    data::pretrained::PreFabConfig,
    kits::speech::whisper::{
        WhisperApiConfig,
        pretrained::bundled,
    },
};

use crate::models::prefab::WHISPER_PREFABS;

/// How a pretrained's weights are stored: the quantization axis.
///
/// `OpenAI` ships one format. A `q8` or `safetensors` export of the same
/// prefab would be another [`WhisperPretrained`] with another variant here,
/// and a loader keyed on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsFormat {
    /// A `PyTorch` checkpoint (`dims` + `model_state_dict`) of fp16 tensors.
    PytorchFp16,
}

impl fmt::Display for WeightsFormat {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::PytorchFp16 => write!(f, "pytorch fp16"),
        }
    }
}

/// One place a pretrained's bytes can be had from.
///
/// Sources are tried in the order listed, and every one is checked against
/// the pretrained's digest before it is used, except the bundled file, which
/// the build already verified.
#[derive(Debug, Clone, Copy)]
pub enum WeightsSource {
    /// A URL, fetched into the cache.
    Url(&'static str),

    /// `openai-whisper`'s own download cache: `<upstream cache>/<file>`,
    /// `~/.cache/whisper` by default. Upstream keeps the file under its bare
    /// name and re-hashes it on every load; here it is hashed once and
    /// linked into the cache.
    UpstreamCache,

    /// A file `bunsen-bundled-whisper` fetched and digest-checked at build
    /// time. Used in place: it is already trusted, and `OUT_DIR` is not a
    /// place to link to, since `cargo clean` empties it.
    Bundled(fn() -> &'static Path),
}

impl fmt::Display for WeightsSource {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Url(url) => write!(f, "url {url}"),
            Self::UpstreamCache => f.write_str("upstream cache (openai-whisper's download root)"),
            Self::Bundled(path) => write!(f, "bundled {}", path().display()),
        }
    }
}

/// A trained set of weights for one prefab.
#[derive(Debug)]
pub struct WhisperPretrained {
    /// The name, unique within its provider.
    pub name: &'static str,

    /// Other names upstream answers to for the same bytes.
    pub aliases: &'static [&'static str],

    /// The [`WHISPER_PREFABS`] entry these weights instantiate.
    pub prefab: &'static str,

    /// A line for a listing.
    pub description: &'static str,

    /// The weights' format.
    pub format: WeightsFormat,

    /// The file name, under every source.
    pub file: &'static str,

    /// Lowercase hex SHA-256 of the file, which pins every source.
    pub sha256: &'static str,

    /// Where the file can be had from, in preference order.
    pub sources: &'static [WeightsSource],
}

impl WhisperPretrained {
    /// Does `name` name this entry, by its name or an alias?
    pub fn matches(
        &self,
        name: &str,
    ) -> bool {
        self.name == name || self.aliases.contains(&name)
    }

    /// The prefab these weights instantiate.
    ///
    /// # Panics
    /// If the table names a prefab that does not exist; a test pins that
    /// every entry's does.
    pub fn prefab(&self) -> PreFabConfig<WhisperApiConfig> {
        WHISPER_PREFABS.expect_lookup_prefab(self.prefab)
    }

    /// The remote sources, in order.
    pub fn urls(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.sources.iter().filter_map(|s| match s {
            WeightsSource::Url(url) => Some(*url),
            _ => None,
        })
    }
}

/// A namespace of pretrained weights: the `provider` in `provider/name`.
#[derive(Debug)]
pub struct WeightsProvider {
    /// The namespace.
    pub name: &'static str,

    /// A line for a listing.
    pub description: &'static str,

    /// Where the table came from.
    pub origin: &'static str,

    /// The license the weights are distributed under.
    pub license: &'static str,

    /// The entries.
    pub items: &'static [WhisperPretrained],
}

impl WeightsProvider {
    /// The entry `name` names, by its name or an alias.
    pub fn lookup(
        &'static self,
        name: &str,
    ) -> Option<&'static WhisperPretrained> {
        self.items.iter().find(|p| p.matches(name))
    }

    /// The qualified id of an entry: `provider/name`.
    pub fn id(
        &self,
        pretrained: &WhisperPretrained,
    ) -> String {
        format!("{}/{}", self.name, pretrained.name)
    }
}

/// `OpenAI`'s checkpoints, as `openai-whisper` names and pins them.
pub static OPENAI: WeightsProvider = WeightsProvider {
    name: "openai",
    description: "OpenAI's Whisper checkpoints, as `openai-whisper` names and pins them",
    origin: "https://github.com/openai/whisper/blob/main/whisper/__init__.py",
    license: "MIT",
    items: &[
        WhisperPretrained {
            name: "tiny.en",
            aliases: &[],
            prefab: "tiny.en",
            description: "39 M parameters, English-only",
            format: WeightsFormat::PytorchFp16,
            file: "tiny.en.pt",
            sha256: "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "tiny",
            aliases: &[],
            prefab: "tiny",
            description: "39 M parameters, multilingual",
            format: WeightsFormat::PytorchFp16,
            file: "tiny.pt",
            sha256: "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "base.en",
            aliases: &[],
            prefab: "base.en",
            description: "74 M parameters, English-only",
            format: WeightsFormat::PytorchFp16,
            file: "base.en.pt",
            sha256: "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "base",
            aliases: &[],
            prefab: "base",
            description: "74 M parameters, multilingual; the checkpoint bunsen bundles",
            format: WeightsFormat::PytorchFp16,
            file: "base.pt",
            sha256: "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
            sources: &[
                WeightsSource::Bundled(bundled::base_pt),
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "small.en",
            aliases: &[],
            prefab: "small.en",
            description: "244 M parameters, English-only",
            format: WeightsFormat::PytorchFp16,
            file: "small.en.pt",
            sha256: "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "small",
            aliases: &[],
            prefab: "small",
            description: "244 M parameters, multilingual",
            format: WeightsFormat::PytorchFp16,
            file: "small.pt",
            sha256: "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "medium.en",
            aliases: &[],
            prefab: "medium.en",
            description: "769 M parameters, English-only",
            format: WeightsFormat::PytorchFp16,
            file: "medium.en.pt",
            sha256: "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "medium",
            aliases: &[],
            prefab: "medium",
            description: "769 M parameters, multilingual",
            format: WeightsFormat::PytorchFp16,
            file: "medium.pt",
            sha256: "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "large-v1",
            aliases: &[],
            prefab: "large",
            description: "1550 M parameters, multilingual",
            format: WeightsFormat::PytorchFp16,
            file: "large-v1.pt",
            sha256: "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "large-v2",
            aliases: &[],
            prefab: "large",
            description: "1550 M parameters, multilingual",
            format: WeightsFormat::PytorchFp16,
            file: "large-v2.pt",
            sha256: "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "large-v3",
            aliases: &["large"],
            prefab: "large-v3",
            description: "1550 M parameters, multilingual, 128 mels",
            format: WeightsFormat::PytorchFp16,
            file: "large-v3.pt",
            sha256: "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
                ),
            ],
        },
        WhisperPretrained {
            name: "large-v3-turbo",
            aliases: &["turbo"],
            prefab: "large-v3-turbo",
            description: "809 M parameters, multilingual, 128 mels, four-layer decoder",
            format: WeightsFormat::PytorchFp16,
            file: "large-v3-turbo.pt",
            sha256: "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
            sources: &[
                WeightsSource::UpstreamCache,
                WeightsSource::Url(
                    "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
                ),
            ],
        },
    ],
};

/// Every provider, in lookup order.
pub static PROVIDERS: &[&WeightsProvider] = &[&OPENAI];

/// The provider called `name`.
pub fn provider(name: &str) -> Option<&'static WeightsProvider> {
    PROVIDERS.iter().copied().find(|p| p.name == name)
}

/// The entry `name` names under `provider`, or under any provider when none
/// is given and exactly one has it.
pub fn lookup_pretrained(
    provider: Option<&str>,
    name: &str,
) -> Option<(&'static WeightsProvider, &'static WhisperPretrained)> {
    match provider {
        Some(provider) => {
            let provider = self::provider(provider)?;
            provider.lookup(name).map(|p| (provider, p))
        }
        None => {
            let mut hits = PROVIDERS
                .iter()
                .copied()
                .filter_map(|provider| provider.lookup(name).map(|p| (provider, p)));
            let first = hits.next()?;
            match hits.next() {
                // Ambiguous: the caller has to qualify it.
                Some(_) => None,
                None => Some(first),
            }
        }
    }
}

/// Every qualified id, for a "did you mean" listing.
pub fn available_ids() -> Vec<String> {
    PROVIDERS
        .iter()
        .flat_map(|provider| provider.items.iter().map(|p| provider.id(p)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_pretrained_names_a_prefab() {
        for provider in PROVIDERS {
            for p in provider.items {
                assert!(
                    WHISPER_PREFABS.lookup_prefab(p.prefab).is_some(),
                    "{}: prefab {:?} is not in WHISPER_PREFABS",
                    provider.id(p),
                    p.prefab,
                );
            }
        }
    }

    #[test]
    fn test_names_and_aliases_are_unique_within_a_provider() {
        for provider in PROVIDERS {
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
    fn test_digests_pin_the_urls() {
        for provider in PROVIDERS {
            for p in provider.items {
                assert_eq!(p.sha256.len(), 64, "{}", provider.id(p));
                assert!(
                    p.sha256
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                    "{}: digest is not lowercase hex",
                    provider.id(p),
                );
                assert!(p.urls().next().is_some(), "{}: no URL", provider.id(p));
                for url in p.urls() {
                    // OpenAI's URLs are digest-addressed; the pin and the
                    // path must agree.
                    assert!(
                        url.ends_with(&format!("/{}/{}", p.sha256, p.file)),
                        "{}: {url} does not end in the digest and file",
                        provider.id(p),
                    );
                }
            }
        }
    }

    #[test]
    fn test_upstream_aliases() {
        let (provider, large) = lookup_pretrained(None, "large").unwrap();
        assert_eq!(provider.name, "openai");
        assert_eq!(large.name, "large-v3");

        let (_, turbo) = lookup_pretrained(Some("openai"), "turbo").unwrap();
        assert_eq!(turbo.name, "large-v3-turbo");

        assert!(lookup_pretrained(Some("nobody"), "base").is_none());
        assert!(lookup_pretrained(None, "gigantic").is_none());
    }

    #[test]
    fn test_one_prefab_many_pretrained() {
        let large: Vec<&str> = OPENAI
            .items
            .iter()
            .filter(|p| p.prefab == "large")
            .map(|p| p.name)
            .collect();
        assert_eq!(large, ["large-v1", "large-v2"]);
    }

    #[test]
    fn test_the_bundled_checkpoint_is_the_openai_base() {
        let base = OPENAI.lookup("base").unwrap();
        assert!(
            base.sources
                .iter()
                .any(|s| matches!(s, WeightsSource::Bundled(_))),
            "openai/base lists the bundled file as a source",
        );
    }
}
