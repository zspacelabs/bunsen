//! [`TextNormalizationConfig`]: the built-in normalizers, as config.

use alloc::{
    sync::Arc,
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
use crate::errors::BunsenResult;

/// A text normalizer bunsen ships, as config: what a catalog row, a preset or
/// a file names, and [`init`](Self::init) builds.
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
    pub fn init(&self) -> BunsenResult<Arc<dyn TextNormalizer>> {
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

impl TextNormalizationConfig {
    /// The normalizer.
    ///
    /// # Errors
    /// A variant's own: one whose feature this build lacks is
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported).
    pub fn init(&self) -> BunsenResult<Arc<dyn TextNormalizer>> {
        Ok(match self {
            Self::None => Arc::new(TextNormalizerChain::default()),
            Self::CollapseLineBreaks => Arc::new(CollapseLineBreaks),
            Self::Transcript => Arc::new(TranscriptTextNormalizer),
            Self::Zyphra(config) => Arc::new(config.init()?),
            Self::Tokenizers(spec) => spec.init()?,
            Self::Chain(stages) => Arc::new(TextNormalizerChain::new(
                stages
                    .iter()
                    .map(Self::init)
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
        let normalizer = config.init().unwrap();
        assert_eq!(
            normalizer.name(),
            "collapse-line-breaks > none > transcript"
        );
        assert_eq!(
            normalizer.normalize("Hi,\nTHERE.", None).unwrap(),
            "hi there"
        );
        assert_eq!(TextNormalizationConfig::None.init().unwrap().name(), "none");
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn test_tokenizers() {
        let config = TextNormalizationConfig::Tokenizers(TokenizersSpec::Spec(
            serde_json::json!({"type": "Lowercase"}),
        ));
        let normalizer = config.init().unwrap();
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
