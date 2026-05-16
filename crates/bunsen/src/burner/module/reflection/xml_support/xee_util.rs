use spanned_error_message::{
    Location,
    SpannedErrorMessage,
};
use xee_xpath::error::ErrorValue;

use crate::errors::BunsenError;

/// Construct a long-form error message from an [`ErrorValue`].
pub fn pretty_errorvalue(e: &ErrorValue) -> String {
    format!("XPath Error: {}: {}\n{}", e.code(), e.message(), e.note())
}

fn offset_to_location(
    src: &str,
    offset: usize,
) -> Location {
    let src = &src[..offset];

    let line = src.lines().count();
    let column = src.lines().last().unwrap().len();

    Location { line, column }
}

/// Adapt an [`xee_xpath::error::Error`] into a [`BunsenError`].
pub fn adapt_xee_error(
    e: xee_xpath::error::Error,
    src: Option<&str>,
) -> BunsenError {
    let value_descr = pretty_errorvalue(&e.error);

    let mut lines = vec![value_descr];

    if let Some(src) = src
        && let Some(src_span) = e.span
    {
        let rng = src_span.range();

        let section = spanned_error_message::Section {
            start: offset_to_location(src, rng.start),
            end: offset_to_location(src, rng.end),
            document: spanned_error_message::Document::from_content(src),
            label: "Parse Error".to_string(),
        };

        let clip_message = SpannedErrorMessage::new().create(&section);

        lines.push(clip_message);
    }

    let msg = lines.join("\n");

    match e.error {
        ErrorValue::XPST0003 => BunsenError::ParseError(msg),
        _ => BunsenError::External(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error() {
        let e = ErrorValue::XPST0003;

        println!("{}", pretty_errorvalue(&e));

        println!("debug:{e:?}, display:{e}");
        println!("msg: {}", e.message());
        println!("note: {}", e.note());
    }
}
