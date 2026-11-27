//! [`TextNormalizationConfig`]: the built-in normalizers, as config.

use alloc::{
    format,
    sync::Arc,
    vec,
    vec::Vec,
};

use burn::config::Config;

use super::{
    CollapseLineBreaks,
    TextNormalizer,
    TextNormalizerChain,
    TranscriptTextNormalizer,
    ZyphraNormalizationConfig,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
    LookupError,
    sys_at,
};

/// A text normalizer bunsen ships, as config: what a catalog row, a preset or
/// a file names, and [`try_init`](Self::try_init) builds.
///
/// A normalizer bunsen does not ship is not a variant: it plugs in as an
/// `Arc<dyn TextNormalizer>` wherever a built one goes.
#[derive(Config, Debug, PartialEq)]
pub enum TextNormalizationConfig {
    /// The text as written.
    None,

    /// [`CollapseLineBreaks`].
    CollapseLineBreaks,

    /// [`TranscriptTextNormalizer`].
    Transcript,

    /// Zyphra's normalizer
    /// ([`ZyphraTextNormalizer`](super::ZyphraTextNormalizer)); in a build
    /// without the feature `zyphra-textnorm`, an init error.
    Zyphra(ZyphraNormalizationConfig),

    /// A `tokenizers` normalizer (`TokenizersTextNormalizer`); in a build
    /// without the feature `tokenizer`, an init error.
    Tokenizers(TokenizersSpec),

    /// Each in order ([`TextNormalizerChain`]).
    Chain(Vec<TextNormalizationConfig>),
}

/// Config for a `tokenizers` normalizer, in every build: the normalizer's
/// spec in `tokenizer.json`'s own syntax, or a `tokenizer.json` whose
/// `normalizer` it is.
#[derive(Config, Debug, PartialEq)]
pub enum TokenizersSpec {
    /// A normalizer in `tokenizer.json`'s syntax: `{"type": "NFKC"}`.
    Spec(serde_json::Value),

    /// The `normalizer` of the `tokenizer.json` at this path.
    TokenizerJson(String),
}

impl TokenizersSpec {
    /// The normalizer.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) in a
    /// build without the feature `tokenizer`; else as
    /// `TokenizersTextNormalizer`'s constructors.
    pub fn try_init(&self) -> BunsenResult<Arc<dyn TextNormalizer>> {
        #[cfg(feature = "tokenizer")]
        {
            use super::TokenizersTextNormalizer;
            Ok(Arc::new(match self {
                Self::Spec(spec) => TokenizersTextNormalizer::from_spec(spec)?,
                Self::TokenizerJson(path) => {
                    TokenizersTextNormalizer::from_tokenizer_json(std::path::Path::new(path))?
                }
            }))
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = self;
            Err(crate::errors::BunsenError::unsupported(
                "a tokenizers normalizer needs bunsen's feature `tokenizer`",
            ))
        }
    }
}

/// A normalizer [`TextNormalizationConfig::parse_spec`] names, and whether
/// this build has it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextNormalizerInfo {
    /// Its spec: `zyphra`, `tokenizers:<TYPE|JSON>`.
    pub spec: &'static str,

    /// What it does.
    pub description: &'static str,

    /// The bunsen feature it needs, if any.
    pub feature: Option<&'static str>,

    /// Whether this build has it.
    pub available: bool,
}

impl TextNormalizationConfig {
    /// The normalizers a spec names ([`parse_spec`](Self::parse_spec)), and
    /// which of them this build has.
    pub fn builtins() -> Vec<TextNormalizerInfo> {
        vec![
            TextNormalizerInfo {
                spec: "none",
                description: "the text as written",
                feature: None,
                available: true,
            },
            TextNormalizerInfo {
                spec: "collapse-line-breaks",
                description: "line breaks, tabs and other whitespace runs to one space",
                feature: None,
                available: true,
            },
            TextNormalizerInfo {
                spec: "transcript",
                description: "lowercase, punctuation dropped, one space between words: what word \
                              error rates compare",
                feature: None,
                available: true,
            },
            TextNormalizerInfo {
                spec: "zyphra",
                description: "Zyphra's normalizer: numbers, money and dates spoken, by the \
                              language's grammar, in 43 languages (needs a language)",
                feature: Some("zyphra-textnorm"),
                available: cfg!(feature = "zyphra-textnorm"),
            },
            TextNormalizerInfo {
                spec: "tokenizers:<TYPE|JSON>",
                description: "a `tokenizers` normalizer, in `tokenizer.json`'s syntax: \
                              `tokenizers:NFKC`, or a spec's JSON",
                feature: Some("tokenizer"),
                available: cfg!(feature = "tokenizer"),
            },
            TextNormalizerInfo {
                spec: "tokenizer-json:<PATH>",
                description: "the `normalizer` of a `tokenizer.json`: a model's own",
                feature: Some("tokenizer"),
                available: cfg!(feature = "tokenizer"),
            },
            TextNormalizerInfo {
                spec: "<FILE>.json",
                description: "a `TextNormalizationConfig`, as JSON",
                feature: None,
                available: true,
            },
        ]
    }

    /// The config a short spec names, as a command line takes one:
    ///
    /// - `none`, `collapse-line-breaks`, `transcript`, `zyphra`: those
    ///   normalizers, with their defaults;
    /// - `tokenizers:NFKC`: a `tokenizers` normalizer of that type, or
    ///   `tokenizers:{...}`, one in `tokenizer.json`'s syntax;
    /// - `tokenizer-json:PATH`: a `tokenizer.json`'s `normalizer`;
    /// - `FILE.json`: a config, as JSON.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for an unknown
    /// name, naming the known;
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a spec's JSON
    /// that does not parse; a file's read or parse.
    pub fn parse_spec(spec: &str) -> BunsenResult<Self> {
        if let Some(kind) = spec.strip_prefix("tokenizers:") {
            let value = if kind.trim_start().starts_with('{') {
                serde_json::from_str(kind).map_err(|error| {
                    BunsenError::illegal(format!("{spec:?}: not a JSON normalizer spec"))
                        .with_cause(error)
                })?
            } else {
                serde_json::json!({ "type": kind })
            };
            return Ok(Self::Tokenizers(TokenizersSpec::Spec(value)));
        }
        if let Some(path) = spec.strip_prefix("tokenizer-json:") {
            return Ok(Self::Tokenizers(TokenizersSpec::TokenizerJson(path.into())));
        }
        if spec.ends_with(".json") {
            let text = std::fs::read_to_string(spec).map_err(sys_at("read", spec))?;
            return serde_json::from_str(&text).map_err(|error| {
                BunsenError::invalid_resource(format!("{spec} is not a text normalization config"))
                    .with_cause(error)
            });
        }
        match spec {
            "none" => Ok(Self::None),
            "collapse-line-breaks" => Ok(Self::CollapseLineBreaks),
            "transcript" => Ok(Self::Transcript),
            "zyphra" => Ok(Self::Zyphra(ZyphraNormalizationConfig::new())),
            _ => Err(BunsenError::lookup(
                LookupError::missing("text normalizer", spec)
                    .with_candidates(Self::builtins().iter().map(|info| info.spec)),
            )),
        }
    }

    /// The config `specs` name, in order ([`parse_spec`](Self::parse_spec)):
    /// one spec's own, several as a [`Chain`](Self::Chain), none as
    /// [`None`](Self::None).
    ///
    /// # Errors
    /// The first spec's.
    pub fn parse_specs<S: AsRef<str>>(specs: impl IntoIterator<Item = S>) -> BunsenResult<Self> {
        let mut configs = specs
            .into_iter()
            .map(|spec| Self::parse_spec(spec.as_ref()))
            .collect::<BunsenResult<Vec<_>>>()?;
        Ok(match configs.len() {
            0 => Self::None,
            1 => configs.remove(0),
            _ => Self::Chain(configs),
        })
    }

    /// The normalizer.
    ///
    /// # Errors
    /// A variant's own: one whose feature this build lacks is
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported).
    pub fn try_init(&self) -> BunsenResult<Arc<dyn TextNormalizer>> {
        Ok(match self {
            Self::None => Arc::new(TextNormalizerChain::default()),
            Self::CollapseLineBreaks => Arc::new(CollapseLineBreaks),
            Self::Transcript => Arc::new(TranscriptTextNormalizer),
            Self::Zyphra(config) => Arc::new(config.try_init()?),
            Self::Tokenizers(spec) => spec.try_init()?,
            Self::Chain(stages) => Arc::new(TextNormalizerChain::new(
                stages
                    .iter()
                    .map(Self::try_init)
                    .collect::<BunsenResult<Vec<_>>>()?,
            )),
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_init() {
        let config = TextNormalizationConfig::Chain(vec![
            TextNormalizationConfig::CollapseLineBreaks,
            TextNormalizationConfig::None,
            TextNormalizationConfig::Transcript,
        ]);
        let normalizer = config.try_init().unwrap();
        assert_eq!(
            normalizer.name(),
            "collapse-line-breaks > none > transcript"
        );
        assert_eq!(
            normalizer.normalize("Hi,\nTHERE.", None).unwrap(),
            "hi there"
        );
        assert_eq!(
            TextNormalizationConfig::None.try_init().unwrap().name(),
            "none"
        );
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn test_tokenizers() {
        let config = TextNormalizationConfig::Tokenizers(TokenizersSpec::Spec(
            serde_json::json!({"type": "Lowercase"}),
        ));
        let normalizer = config.try_init().unwrap();
        assert_eq!(normalizer.name(), "tokenizers(Lowercase)");
        assert_eq!(normalizer.normalize("ABC", None).unwrap(), "abc");
    }

    #[cfg(not(feature = "tokenizer"))]
    #[test]
    fn test_tokenizers_needs_the_feature() {
        let config = TextNormalizationConfig::Tokenizers(TokenizersSpec::Spec(
            serde_json::json!({"type": "Lowercase"}),
        ));
        let error = config.init().unwrap_err();
        assert_eq!(error.kind(), crate::errors::BunsenErrorKind::Unsupported);
    }

    #[test]
    fn test_parse_spec() {
        assert_eq!(
            TextNormalizationConfig::parse_spec("zyphra").unwrap(),
            TextNormalizationConfig::Zyphra(ZyphraNormalizationConfig::new())
        );
        assert_eq!(
            TextNormalizationConfig::parse_spec("tokenizers:NFKC").unwrap(),
            TextNormalizationConfig::Tokenizers(TokenizersSpec::Spec(
                serde_json::json!({"type": "NFKC"})
            ))
        );
        assert_eq!(
            TextNormalizationConfig::parse_spec(
                r#"tokenizers:{"type": "Strip", "strip_left": true, "strip_right": true}"#
            )
            .unwrap(),
            TextNormalizationConfig::Tokenizers(TokenizersSpec::Spec(
                serde_json::json!({"type": "Strip", "strip_left": true, "strip_right": true})
            ))
        );
        assert_eq!(
            TextNormalizationConfig::parse_spec("tokenizer-json:/m/tokenizer.json").unwrap(),
            TextNormalizationConfig::Tokenizers(TokenizersSpec::TokenizerJson(
                "/m/tokenizer.json".into()
            ))
        );
        let error = TextNormalizationConfig::parse_spec("zyfra").unwrap_err();
        assert_eq!(error.kind(), crate::errors::BunsenErrorKind::Lookup);
        assert!(error.to_string().contains("zyfra"), "{error}");
        let error = TextNormalizationConfig::parse_spec("tokenizers:{nope").unwrap_err();
        assert_eq!(error.kind(), crate::errors::BunsenErrorKind::Illegal);
    }

    #[test]
    fn test_parse_spec_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("norm.json");
        let config = TextNormalizationConfig::Chain(vec![
            TextNormalizationConfig::CollapseLineBreaks,
            TextNormalizationConfig::Transcript,
        ]);
        std::fs::write(&path, serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(
            TextNormalizationConfig::parse_spec(path.to_str().unwrap()).unwrap(),
            config
        );
        let error =
            TextNormalizationConfig::parse_spec(dir.path().join("missing.json").to_str().unwrap())
                .unwrap_err();
        assert_eq!(error.kind(), crate::errors::BunsenErrorKind::Lookup);
    }

    #[test]
    fn test_parse_specs() {
        assert_eq!(
            TextNormalizationConfig::parse_specs::<&str>([]).unwrap(),
            TextNormalizationConfig::None
        );
        assert_eq!(
            TextNormalizationConfig::parse_specs(["transcript"]).unwrap(),
            TextNormalizationConfig::Transcript
        );
        assert_eq!(
            TextNormalizationConfig::parse_specs(["collapse-line-breaks", "transcript"]).unwrap(),
            TextNormalizationConfig::Chain(vec![
                TextNormalizationConfig::CollapseLineBreaks,
                TextNormalizationConfig::Transcript,
            ])
        );
    }

    #[test]
    fn test_builtins() {
        let builtins = TextNormalizationConfig::builtins();
        let zyphra = builtins.iter().find(|info| info.spec == "zyphra").unwrap();
        assert_eq!(zyphra.available, cfg!(feature = "zyphra-textnorm"));
        // Every plain name parses.
        for info in builtins
            .iter()
            .filter(|info| !info.spec.contains(['<', ':']))
        {
            TextNormalizationConfig::parse_spec(info.spec).unwrap();
        }
    }

    #[test]
    fn test_serde() {
        let config = TextNormalizationConfig::Chain(vec![
            TextNormalizationConfig::CollapseLineBreaks,
            TextNormalizationConfig::Transcript,
        ]);
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(json, r#"{"Chain":["CollapseLineBreaks","Transcript"]}"#);
        assert_eq!(
            serde_json::from_str::<TextNormalizationConfig>(&json).unwrap(),
            config
        );
    }
}
