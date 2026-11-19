//! Text normalization: a request's text as typed, to what its prompt reads.

use alloc::{
    collections::BTreeMap,
    string::String,
    vec::Vec,
};
use core::fmt;

use burn::config::Config;

use crate::{
    errors::BunsenResult,
    kits::support::text::collapse_line_breaks,
};

/// How [`Zonos3Driver`](super::Zonos3Driver) normalizes a request's text
/// before its prompt.
#[derive(Config, Debug, PartialEq)]
pub enum Zonos3TextNormalization {
    /// The text as typed, line breaks collapsed
    /// ([`collapse_line_breaks`]): the release's with its
    /// normalization off.
    None,

    /// Zyphra's normalizer, `zyphra-textnorm`, as the release normalizes:
    /// numbers, money, dates and the like to their spoken form, by the
    /// language's grammar (line breaks collapsed first). Needs the feature
    /// `zyphra-textnorm`: in a build without it, a driver configured with
    /// this mode is an init error.
    Zyphra(Zonos3ZyphraNormalization),
}

/// Config for [`Zonos3TextNormalization::Zyphra`].
#[derive(Config, Debug, PartialEq)]
pub struct Zonos3ZyphraNormalization {
    /// The language: a request code of the release's `languages` table
    /// (`en_us`), or one of the normalizer's grammars (`en`).
    #[config(default = "String::from(\"en_us\")")]
    pub language: String,

    /// The markup tags kept as typed: `laugh` keeps `<laugh/>`. None, as
    /// the release's reference normalizes.
    #[config(default = "Vec::new()")]
    pub markup_tags: Vec<String>,

    /// Whether the normalizer's pre-pass runs before its grammars
    /// (markdown, emoji, look-alike characters, and more readings of
    /// numbers, dates and money). Off, the grammars' own reading, as the
    /// release's reference normalizes.
    #[config(default = false)]
    pub prepass: bool,
}

/// A [`Zonos3TextNormalization`], ready to normalize.
pub struct Zonos3Normalizer {
    inner: Inner,
}

enum Inner {
    None,
    #[cfg(feature = "zyphra-textnorm")]
    Zyphra {
        normalizer: zyphra_textnorm::Normalizer,
        grammar: String,
    },
}

impl fmt::Debug for Zonos3Normalizer {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match &self.inner {
            Inner::None => f.write_str("Zonos3Normalizer::None"),
            #[cfg(feature = "zyphra-textnorm")]
            Inner::Zyphra { grammar, .. } => f
                .debug_struct("Zonos3Normalizer::Zyphra")
                .field("grammar", grammar)
                .finish_non_exhaustive(),
        }
    }
}

impl Zonos3Normalizer {
    /// The normalizer of `mode`, a request language resolved through
    /// `languages`, the release's table of request codes to grammars.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for
    /// [`Zyphra`](Zonos3TextNormalization::Zyphra) in a build without the
    /// feature `zyphra-textnorm`, or for a language it has no grammar for.
    pub fn new(
        mode: &Zonos3TextNormalization,
        languages: &BTreeMap<String, String>,
    ) -> BunsenResult<Self> {
        let inner = match mode {
            Zonos3TextNormalization::None => Inner::None,
            Zonos3TextNormalization::Zyphra(config) => Self::zyphra(config, languages)?,
        };
        Ok(Self { inner })
    }

    #[cfg(feature = "zyphra-textnorm")]
    fn zyphra(
        config: &Zonos3ZyphraNormalization,
        languages: &BTreeMap<String, String>,
    ) -> BunsenResult<Inner> {
        let grammar = languages
            .get(&config.language)
            .cloned()
            .unwrap_or_else(|| config.language.clone());
        if !zyphra_textnorm::supported(&grammar) {
            return Err(crate::errors::BunsenError::unsupported(alloc::format!(
                "zyphra-textnorm has no grammar for the language {:?}",
                config.language
            )));
        }
        let normalizer = zyphra_textnorm::Normalizer::with_markup_tags(&config.markup_tags)
            .with_prepass(config.prepass);
        Ok(Inner::Zyphra {
            normalizer,
            grammar,
        })
    }

    #[cfg(not(feature = "zyphra-textnorm"))]
    fn zyphra(
        _config: &Zonos3ZyphraNormalization,
        _languages: &BTreeMap<String, String>,
    ) -> BunsenResult<Inner> {
        Err(crate::errors::BunsenError::unsupported(
            "Zyphra text normalization needs bunsen's feature `zyphra-textnorm`",
        ))
    }

    /// `text`, normalized.
    pub fn normalize(
        &self,
        text: &str,
    ) -> String {
        match &self.inner {
            Inner::None => collapse_line_breaks(text).into_owned(),
            #[cfg(feature = "zyphra-textnorm")]
            Inner::Zyphra {
                normalizer,
                grammar,
            } => normalizer.normalize(text, grammar),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::BunsenErrorKind;

    fn languages() -> BTreeMap<String, String> {
        BTreeMap::from([("en_us".to_string(), "en".to_string())])
    }

    #[test]
    fn test_none() {
        let none = Zonos3Normalizer::new(&Zonos3TextNormalization::None, &languages()).unwrap();
        assert_eq!(none.normalize("It costs $5.\nOk"), "It costs $5. Ok");
    }

    #[cfg(feature = "zyphra-textnorm")]
    #[test]
    fn test_zyphra() {
        let zyphra = Zonos3Normalizer::new(
            &Zonos3TextNormalization::Zyphra(Zonos3ZyphraNormalization::new()),
            &languages(),
        )
        .unwrap();
        assert_eq!(
            zyphra.normalize("It costs $5.32."),
            "It costs five dollars thirty two cents."
        );
        let error = Zonos3Normalizer::new(
            &Zonos3TextNormalization::Zyphra(
                Zonos3ZyphraNormalization::new().with_language("xx_yy".to_string()),
            ),
            &languages(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Unsupported);
    }

    #[cfg(not(feature = "zyphra-textnorm"))]
    #[test]
    fn test_zyphra_needs_the_feature() {
        let error = Zonos3Normalizer::new(
            &Zonos3TextNormalization::Zyphra(Zonos3ZyphraNormalization::new()),
            &languages(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Unsupported);
        assert!(error.to_string().contains("zyphra-textnorm"));
    }
}
