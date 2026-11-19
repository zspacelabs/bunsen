//! Zyphra's normalizer, `zyphra-textnorm`, behind [`TextNormalizer`].

use alloc::{
    borrow::Cow,
    format,
    string::String,
    vec::Vec,
};
use core::fmt;

use burn::config::Config;

use super::{
    BunsenLanguageTag,
    TextNormalizer,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Config for [`ZyphraTextNormalizer`].
#[derive(Config, Debug, PartialEq)]
pub struct ZyphraNormalizationConfig {
    /// The markup tags kept as written: `laugh` keeps `<laugh/>`. None by
    /// default, as ZONOS3's release reference normalizes.
    #[config(default = "Vec::new()")]
    pub markup_tags: Vec<String>,

    /// Whether the normalizer's pre-pass runs before its grammars (markdown,
    /// emoji, look-alike characters, and more readings of numbers, dates and
    /// money). Off by default: the grammars' own reading, as ZONOS3's release
    /// reference normalizes.
    #[config(default = false)]
    pub prepass: bool,
}

impl ZyphraNormalizationConfig {
    /// The normalizer.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) in a
    /// build without the feature `zyphra-textnorm`.
    pub fn init(&self) -> BunsenResult<ZyphraTextNormalizer> {
        ZyphraTextNormalizer::new(self)
    }
}

/// Zyphra's normalizer (`zyphra-textnorm`): numbers, money, dates and the like
/// to their spoken form, by the grammar of the text's language, in 43
/// languages.
///
/// The grammar is the language tag's primary language (`en` of `en-US`;
/// `zh` is Mandarin). A text needs its language: without one, or in a
/// language without a grammar, it is an error, never a guess. In a build
/// without the feature `zyphra-textnorm` it cannot be built
/// ([`ZyphraNormalizationConfig::init`]).
pub struct ZyphraTextNormalizer {
    #[cfg(feature = "zyphra-textnorm")]
    normalizer: zyphra_textnorm::Normalizer,
    config: ZyphraNormalizationConfig,
}

impl fmt::Debug for ZyphraTextNormalizer {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("ZyphraTextNormalizer")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ZyphraTextNormalizer {
    /// The normalizer of `config`.
    ///
    /// # Errors
    /// As [`ZyphraNormalizationConfig::init`].
    #[cfg(feature = "zyphra-textnorm")]
    pub fn new(config: &ZyphraNormalizationConfig) -> BunsenResult<Self> {
        Ok(Self {
            normalizer: zyphra_textnorm::Normalizer::with_markup_tags(&config.markup_tags)
                .with_prepass(config.prepass),
            config: config.clone(),
        })
    }

    /// The normalizer of `config`.
    ///
    /// # Errors
    /// As [`ZyphraNormalizationConfig::init`].
    #[cfg(not(feature = "zyphra-textnorm"))]
    pub fn new(_config: &ZyphraNormalizationConfig) -> BunsenResult<Self> {
        Err(BunsenError::unsupported(
            "Zyphra text normalization needs bunsen's feature `zyphra-textnorm`",
        ))
    }

    /// Its config.
    pub fn config(&self) -> &ZyphraNormalizationConfig {
        &self.config
    }

    /// The grammar for `language`: its primary language.
    fn grammar<'l>(
        &self,
        language: Option<&'l BunsenLanguageTag>,
    ) -> BunsenResult<&'l str> {
        let language = language.ok_or_else(|| {
            BunsenError::illegal("Zyphra text normalization needs the text's language")
        })?;
        let grammar = language.language();
        if !Self::has_grammar(grammar) {
            return Err(BunsenError::unsupported(format!(
                "zyphra-textnorm has no grammar for the language {language}"
            )));
        }
        Ok(grammar)
    }

    #[cfg(feature = "zyphra-textnorm")]
    fn has_grammar(grammar: &str) -> bool {
        zyphra_textnorm::supported(grammar)
    }

    #[cfg(not(feature = "zyphra-textnorm"))]
    fn has_grammar(_grammar: &str) -> bool {
        false
    }
}

impl TextNormalizer for ZyphraTextNormalizer {
    /// `zyphra`, with `+prepass` and its markup tags when set.
    fn name(&self) -> String {
        let mut name = String::from("zyphra");
        if self.config.prepass {
            name.push_str("+prepass");
        }
        if !self.config.markup_tags.is_empty() {
            name.push_str(&format!("(tags: {})", self.config.markup_tags.join(", ")));
        }
        name
    }

    fn check_language(
        &self,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<()> {
        self.grammar(language).map(|_| ())
    }

    fn normalize<'a>(
        &self,
        text: &'a str,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<Cow<'a, str>> {
        let grammar = self.grammar(language)?;
        #[cfg(feature = "zyphra-textnorm")]
        {
            Ok(Cow::Owned(self.normalizer.normalize(text, grammar)))
        }
        #[cfg(not(feature = "zyphra-textnorm"))]
        {
            let _ = (text, grammar);
            unreachable!("a ZyphraTextNormalizer is built only with `zyphra-textnorm`")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::BunsenErrorKind;

    #[cfg(feature = "zyphra-textnorm")]
    #[test]
    fn test_zyphra() {
        let zyphra = ZyphraNormalizationConfig::new().init().unwrap();
        let en_us = BunsenLanguageTag::parse("en-US").unwrap();
        assert_eq!(zyphra.name(), "zyphra");
        assert_eq!(
            zyphra.normalize("It costs $5.32.", Some(&en_us)).unwrap(),
            "It costs five dollars thirty two cents."
        );
        zyphra.check_language(Some(&en_us)).unwrap();

        let error = zyphra.normalize("x", None).unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Illegal);

        let klingon = BunsenLanguageTag::parse("tlh").unwrap();
        let error = zyphra.check_language(Some(&klingon)).unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Unsupported);
        assert!(error.to_string().contains("tlh"));

        let tagged = ZyphraNormalizationConfig::new()
            .with_prepass(true)
            .with_markup_tags(alloc::vec!["laugh".into()])
            .init()
            .unwrap();
        assert_eq!(tagged.name(), "zyphra+prepass(tags: laugh)");
    }

    #[cfg(not(feature = "zyphra-textnorm"))]
    #[test]
    fn test_zyphra_needs_the_feature() {
        let error = ZyphraNormalizationConfig::new().init().unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Unsupported);
        assert!(error.to_string().contains("zyphra-textnorm"));
    }
}
