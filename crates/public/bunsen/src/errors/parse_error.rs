//! `ParseError`: text or bytes did not parse.

use alloc::{
    borrow::Cow,
    string::{
        String,
        ToString,
    },
    sync::Arc,
};
use core::{
    error::Error,
    fmt,
};

/// Text or bytes did not parse as what was asked for.
///
/// A `ParseError` has no kind of its own: whether the text was outside data
/// ([`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)), a
/// request ([`Policy`](crate::errors::BunsenErrorKind::Policy)) or text the
/// program supplied ([`Illegal`](crate::errors::BunsenErrorKind::Illegal)) is
/// for the site to say, with
/// [`BunsenError::from_cause`](crate::errors::BunsenError::from_cause).
///
/// ```
/// use bunsen::errors::{
///     BunsenError,
///     BunsenErrorKind,
///     ParseError,
/// };
///
/// let e = BunsenError::from_cause(
///     BunsenErrorKind::InvalidResource,
///     ParseError::new("tiktoken vocab line")
///         .at("vocab.tiktoken:3")
///         .because("not two fields"),
/// );
/// assert_eq!(
///     e.to_string(),
///     "vocab.tiktoken:3: cannot parse tiktoken vocab line: not two fields"
/// );
/// ```
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ParseError {
    /// What the text should have been: `"dtype"`, `"safetensors header"`.
    pub what: Cow<'static, str>,
    /// Where the text came from, if known: a path, `path:line`, a row.
    pub at: Option<String>,
    /// The offending text, if short enough to show.
    pub input: Option<String>,
    /// Why it did not parse, in words, when there is no source error.
    pub reason: Option<String>,
    /// The parser's own error, if any.
    pub source: Option<Arc<dyn Error + Send + Sync + 'static>>,
}

impl ParseError {
    /// The text did not parse as `what`.
    pub fn new(what: impl Into<Cow<'static, str>>) -> Self {
        Self {
            what: what.into(),
            at: None,
            input: None,
            reason: None,
            source: None,
        }
    }

    /// Sets where the text came from.
    pub fn at(
        mut self,
        at: impl fmt::Display,
    ) -> Self {
        self.at = Some(at.to_string());
        self
    }

    /// Sets the offending text.
    pub fn input(
        mut self,
        input: impl fmt::Display,
    ) -> Self {
        self.input = Some(input.to_string());
        self
    }

    /// Sets why it did not parse, in words.
    pub fn because(
        mut self,
        reason: impl fmt::Display,
    ) -> Self {
        self.reason = Some(reason.to_string());
        self
    }

    /// Sets the parser's own error. Its text follows the parse error's own in
    /// `Display`.
    pub fn with_source<E>(
        mut self,
        source: E,
    ) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        self.source = Some(Arc::new(source));
        self
    }
}

impl fmt::Display for ParseError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if let Some(at) = &self.at {
            write!(f, "{at}: ")?;
        }
        write!(f, "cannot parse {}", self.what)?;
        if let Some(input) = &self.input {
            write!(f, " {input:?}")?;
        }
        if let Some(reason) = &self.reason {
            write!(f, ": {reason}")?;
        } else if let Some(source) = &self.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}

impl Error for ParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // The source's text is already in this error's `Display`; its own
        // source, if any, comes next.
        self.source.as_deref().and_then(|s| s.source())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(
            ParseError::new("dtype").input("f99").to_string(),
            "cannot parse dtype \"f99\""
        );
        let int_err = "x".parse::<u32>().unwrap_err();
        let e = ParseError::new("grid shape")
            .at("--grid")
            .with_source(int_err);
        assert_eq!(
            e.to_string(),
            "--grid: cannot parse grid shape: invalid digit found in string"
        );
    }
}
