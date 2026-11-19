//! The [`TextNormalizer`] seam, and the chain that composes normalizers.

use alloc::{
    borrow::Cow,
    string::String,
    sync::Arc,
    vec::Vec,
};
use core::fmt::Debug;

use super::BunsenLanguageTag;
use crate::errors::BunsenResult;

/// Text, as written, to the form a model or a comparison reads: numbers
/// spoken, Unicode folded, punctuation dropped, whatever the normalizer is
/// for.
///
/// Plugged in as an `Arc<dyn TextNormalizer>`, so a normalizer bunsen does
/// not ship plugs in beside the ones it does
/// ([`TextNormalizationConfig`](super::TextNormalizationConfig) builds
/// those). A normalizer rewrites text; it is never where user input is
/// dropped.
pub trait TextNormalizer: Debug + Send + Sync {
    /// What it is called in reports: `collapse-line-breaks`,
    /// `zyphra(prepass)`.
    fn name(&self) -> String;

    /// `text`, normalized; borrowed when it is `text`, or a part of it.
    ///
    /// `language` is the text's, when known; a normalizer that reads by
    /// language says what it does without one.
    ///
    /// # Errors
    /// The normalizer's own: a language it cannot read
    /// ([`Unsupported`](crate::errors::BunsenErrorKind::Unsupported)), say.
    fn normalize<'a>(
        &self,
        text: &'a str,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<Cow<'a, str>>;
}

/// Normalizers run in order, each on the one before's output. The empty
/// chain is the identity.
#[derive(Clone, Debug, Default)]
pub struct TextNormalizerChain {
    stages: Vec<Arc<dyn TextNormalizer>>,
}

impl TextNormalizerChain {
    /// The chain of `stages`, in order.
    pub fn new(stages: impl IntoIterator<Item = Arc<dyn TextNormalizer>>) -> Self {
        Self {
            stages: stages.into_iter().collect(),
        }
    }

    /// Its stages, in order.
    pub fn stages(&self) -> &[Arc<dyn TextNormalizer>] {
        &self.stages
    }

    /// Each stage's name and its output, in order: how a chain reads `text`,
    /// step by step.
    ///
    /// # Errors
    /// The first stage's error.
    pub fn explain(
        &self,
        text: &str,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<Vec<(String, String)>> {
        let mut current = String::from(text);
        let mut steps = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            current = stage.normalize(&current, language)?.into_owned();
            steps.push((stage.name(), current.clone()));
        }
        Ok(steps)
    }
}

impl TextNormalizer for TextNormalizerChain {
    /// `none` for the empty chain; else the stages' names, joined by ` > `.
    fn name(&self) -> String {
        if self.stages.is_empty() {
            return String::from("none");
        }
        self.stages
            .iter()
            .map(|stage| stage.name())
            .collect::<Vec<_>>()
            .join(" > ")
    }

    fn normalize<'a>(
        &self,
        text: &'a str,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<Cow<'a, str>> {
        let mut current = Cow::Borrowed(text);
        for stage in &self.stages {
            let changed = match stage.normalize(&current, language)? {
                Cow::Owned(changed) => Some(changed),
                // A stage may borrow a part of its input (a trim).
                Cow::Borrowed(part) if part != current.as_ref() => Some(String::from(part)),
                Cow::Borrowed(_) => None,
            };
            if let Some(changed) = changed {
                current = Cow::Owned(changed);
            }
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        format,
        string::ToString,
        vec,
    };

    use super::*;
    use crate::errors::BunsenError;

    /// Appends its mark, and the language it was given.
    #[derive(Debug)]
    struct Mark(&'static str);

    impl TextNormalizer for Mark {
        fn name(&self) -> String {
            self.0.to_string()
        }

        fn normalize<'a>(
            &self,
            text: &'a str,
            language: Option<&BunsenLanguageTag>,
        ) -> BunsenResult<Cow<'a, str>> {
            let language = language.map_or("-".to_string(), ToString::to_string);
            Ok(Cow::Owned(format!("{text}{}[{language}]", self.0)))
        }
    }

    #[derive(Debug)]
    struct Refuse;

    impl TextNormalizer for Refuse {
        fn name(&self) -> String {
            "refuse".to_string()
        }

        fn normalize<'a>(
            &self,
            _text: &'a str,
            _language: Option<&BunsenLanguageTag>,
        ) -> BunsenResult<Cow<'a, str>> {
            Err(BunsenError::unsupported("no"))
        }
    }

    #[test]
    fn test_empty_chain_is_identity() {
        let chain = TextNormalizerChain::default();
        assert_eq!(chain.name(), "none");
        assert!(matches!(
            chain.normalize("abc", None).unwrap(),
            Cow::Borrowed("abc")
        ));
        assert!(chain.explain("abc", None).unwrap().is_empty());
    }

    #[test]
    fn test_chain_runs_in_order_with_the_language() {
        let chain = TextNormalizerChain::new(vec![
            Arc::new(Mark("a")) as Arc<dyn TextNormalizer>,
            Arc::new(Mark("b")),
        ]);
        let en = BunsenLanguageTag::parse("en").unwrap();
        assert_eq!(chain.name(), "a > b");
        assert_eq!(chain.normalize("x", Some(&en)).unwrap(), "xa[en]b[en]");
        assert_eq!(
            chain.explain("x", None).unwrap(),
            vec![
                ("a".to_string(), "xa[-]".to_string()),
                ("b".to_string(), "xa[-]b[-]".to_string()),
            ]
        );
    }

    #[test]
    fn test_chain_stops_at_an_error() {
        let chain = TextNormalizerChain::new(vec![
            Arc::new(Mark("a")) as Arc<dyn TextNormalizer>,
            Arc::new(Refuse),
            Arc::new(Mark("b")),
        ]);
        assert!(chain.normalize("x", None).is_err());
        assert!(chain.explain("x", None).is_err());
    }
}
