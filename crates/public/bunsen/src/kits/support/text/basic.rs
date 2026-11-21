//! The model-free normalizers.

use alloc::{
    borrow::Cow,
    string::String,
};

use super::{
    BunsenLanguageTag,
    TextNormalizer,
};
use crate::errors::BunsenResult;

/// `text` with every whitespace run that holds a line break, a tab or any
/// whitespace but a plain space collapsed to one space; runs of plain spaces
/// are kept.
pub fn collapse_line_breaks(text: &str) -> Cow<'_, str> {
    if !text.chars().any(|c| c.is_whitespace() && c != ' ') {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut run = String::new();
    let mut breaks = false;
    for c in text.chars() {
        if c.is_whitespace() {
            run.push(c);
            breaks |= c != ' ';
        } else {
            if breaks {
                out.push(' ');
            } else {
                out.push_str(&run);
            }
            run.clear();
            breaks = false;
            out.push(c);
        }
    }
    if breaks {
        out.push(' ');
    } else {
        out.push_str(&run);
    }
    Cow::Owned(out)
}

/// Line breaks, tabs and other whitespace runs to one space
/// ([`collapse_line_breaks`]); any language.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollapseLineBreaks;

impl TextNormalizer for CollapseLineBreaks {
    fn name(&self) -> String {
        String::from("collapse-line-breaks")
    }

    fn normalize<'a>(
        &self,
        text: &'a str,
        _language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<Cow<'a, str>> {
        Ok(collapse_line_breaks(text))
    }
}

/// A transcript for comparison: lowercase, punctuation dropped (apostrophes
/// kept), one space between words; any language.
///
/// What word error rates count over (`kits::speech::support::testing`'s
/// `normalize_transcript` splits its output into words). Deliberately small:
/// it does not reconcile
/// spelled-out numbers with digits, or expand contractions; a normalizer that
/// speaks numbers chained before it does the first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TranscriptTextNormalizer;

impl TranscriptTextNormalizer {
    /// `text`, normalized: [`normalize`](TextNormalizer::normalize), which
    /// never fails.
    pub fn transcript(
        &self,
        text: &str,
    ) -> String {
        let mut out = String::with_capacity(text.len());
        for word in text.split_whitespace() {
            let start = out.len();
            if start > 0 {
                out.push(' ');
            }
            let before = out.len();
            out.extend(
                word.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '\'')
                    .flat_map(char::to_lowercase),
            );
            if out.len() == before {
                out.truncate(start);
            }
        }
        out
    }
}

impl TextNormalizer for TranscriptTextNormalizer {
    fn name(&self) -> String {
        String::from("transcript")
    }

    fn normalize<'a>(
        &self,
        text: &'a str,
        _language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<Cow<'a, str>> {
        Ok(Cow::Owned(self.transcript(text)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_line_breaks() {
        assert_eq!(collapse_line_breaks("a  b\n\nc \t d\n"), "a  b c d ");
        assert!(matches!(
            collapse_line_breaks("plain  text"),
            Cow::Borrowed("plain  text")
        ));
        assert_eq!(
            CollapseLineBreaks
                .normalize("It costs $5.\nOk", None)
                .unwrap(),
            "It costs $5. Ok"
        );
    }

    #[test]
    fn test_transcript() {
        let normalize = |text| {
            TranscriptTextNormalizer
                .normalize(text, None)
                .unwrap()
                .into_owned()
        };
        assert_eq!(normalize("Hello, World!  It's  me."), "hello world it's me");
        assert_eq!(normalize(" -- ...  ok ? "), "ok");
        assert_eq!(normalize("ÉCOLE Straße"), "école straße");
        assert_eq!(normalize(""), "");
    }
}
