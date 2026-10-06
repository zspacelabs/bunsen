use spanned_error_message::{
    Location,
    SpannedErrorMessage,
};
use xee_xpath::error::ErrorValue;

use crate::errors::{
    BunsenError,
    ParseError,
};

/// Constructs a long-form error message from an [`ErrorValue`].
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

/// Names the kind of an `XPath` error, for the label on its source span.
///
/// The standard codes carry their kind: `XPST`/`XQST` static, `XPTY`/`XQTY`
/// type, `XPDY`/`XQDY` dynamic; `FO` codes come from functions and
/// operators. `XPST0003`, the static error for text outside the grammar, is
/// a parse error. Anything else (`xee`'s own codes, application errors) is
/// just an `XPath` error.
fn error_kind_label(e: &ErrorValue) -> &'static str {
    if matches!(e, ErrorValue::XPST0003) {
        return "Parse Error";
    }
    if matches!(e, ErrorValue::Application(_)) {
        return "XPath Error";
    }

    let code = e.code();
    match code.get(..4) {
        Some("XPST" | "XQST") => "Static Error",
        Some("XPTY" | "XQTY") => "Type Error",
        Some("XPDY" | "XQDY") => "Dynamic Error",
        _ if code.starts_with("FO") => "Function Error",
        _ => "XPath Error",
    }
}

/// Adapts an [`xee_xpath::error::Error`] into a [`BunsenError`].
///
/// The message is one line: the kind of error, its code, and its summary.
/// The details hold the code's note and, given the source, the source with
/// the error's span marked and labelled with the kind of error: "Parse
/// Error", "Type Error", and so on.
///
/// The kind follows the error code:
/// - `XPST0003` (text outside the grammar) is
///   [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
///   [`ParseError`] cause for the expression;
/// - every other standard code (static, type, dynamic, function), and an
///   application error the expression raised, is `Illegal`, with the `xee`
///   error as its cause: the expression is wrong;
/// - `xee`'s `Unsupported` is
///   [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), its
///   `StackOverflow` is [`Sys`](crate::errors::BunsenErrorKind::Sys), and its
///   `UsedQueryWithWrongQueries` is
///   [`Internal`](crate::errors::BunsenErrorKind::Internal).
///
/// A caller that runs an expression from outside the program re-marks the
/// `Illegal` errors [`as_policy`](crate::errors::ResultContext::as_policy).
pub fn adapt_xee_error(
    e: xee_xpath::error::Error,
    src: Option<&str>,
) -> BunsenError {
    let label = error_kind_label(&e.error);
    let summary = e
        .error
        .message()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let message = format!("XPath {label}: {}: {summary}", e.error.code());

    let mut details = Vec::new();
    if !e.error.note().is_empty() {
        details.push(e.error.note().to_string());
    }
    if let Some(src) = src
        && let Some(src_span) = e.span
    {
        let rng = src_span.range();

        let section = spanned_error_message::Section {
            start: offset_to_location(src, rng.start),
            end: offset_to_location(src, rng.end),
            document: spanned_error_message::Document::from_content(src),
            label: label.to_string(),
        };

        details.push(SpannedErrorMessage::new().create(&section));
    }

    let error = match &e.error {
        ErrorValue::XPST0003 => {
            let mut parse = ParseError::new("XPath expression");
            if let Some(src) = src {
                parse = parse.input(src);
            }
            BunsenError::illegal(message).with_cause(parse.with_source(e))
        }
        ErrorValue::Unsupported => BunsenError::unsupported(message).with_cause(e),
        ErrorValue::StackOverflow => BunsenError::sys(message).with_cause(e),
        ErrorValue::UsedQueryWithWrongQueries => BunsenError::internal(message).with_cause(e),
        _ => BunsenError::illegal(message).with_cause(e),
    };

    if details.is_empty() {
        error
    } else {
        error.with_details(details.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use burn::nn::{
        Linear,
        LinearConfig,
    };

    use super::*;
    use crate::{
        burner::module::reflection::XmlModuleTree,
        errors::{
            BunsenErrorKind,
            testing::{
                ErrorMatcher,
                predicate,
            },
        },
        support::testing::{
            CpuBackend,
            cpu_device,
        },
    };

    fn linear_tree() -> XmlModuleTree {
        let module: Linear<CpuBackend> = LinearConfig::new(2, 3).init(&cpu_device());
        XmlModuleTree::build(&module)
    }

    /// An error raised while evaluating the expression is labelled with its
    /// own kind on the source span: a type error is a "Type Error".
    #[test]
    fn test_evaluation_error_is_labelled_with_its_kind() {
        let mut mtree = linear_tree();

        // Parses, then fails on evaluation: a two-item sequence has no
        // boolean value.
        let expr = "Linear/*[@name='weight', @rank=2]";
        let err = mtree.select_param_ids(expr).unwrap_err();

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("XPath Type Error: XPTY0004")
            .details_contains("Type Error")
            .has_cause::<xee_xpath::error::Error>()
            .assert(&err);
        assert!(!format!("{err:#}").contains("Parse Error"), "{err:#}");
    }

    /// A syntax error is labelled "Parse Error", and is `Illegal` with a
    /// [`ParseError`] cause.
    #[test]
    fn test_syntax_error_is_labelled_parse_error() {
        let mut mtree = linear_tree();

        let err = mtree.try_select("Linear/*[").err().unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("XPath Parse Error: XPST0003")
            .details_contains("Parse Error")
            .cause(predicate("the expression", |p: &ParseError| {
                p.input
                    .as_deref()
                    .is_some_and(|i| i.ends_with("/Linear/*["))
            }))
            .assert(&err);
    }

    #[test]
    fn test_error_kind_label() {
        for (e, label) in [
            (ErrorValue::XPST0003, "Parse Error"),
            (ErrorValue::XPST0008, "Static Error"),
            (ErrorValue::XPTY0004, "Type Error"),
            (ErrorValue::XPDY0002, "Dynamic Error"),
            (ErrorValue::FORG0006, "Function Error"),
            (ErrorValue::StackOverflow, "XPath Error"),
        ] {
            assert_eq!(error_kind_label(&e), label, "{e}");
        }
    }

    #[test]
    fn test_error() {
        let e = ErrorValue::XPST0003;

        println!("{}", pretty_errorvalue(&e));

        println!("debug:{e:?}, display:{e}");
        println!("msg: {}", e.message());
        println!("note: {}", e.note());
    }
}
