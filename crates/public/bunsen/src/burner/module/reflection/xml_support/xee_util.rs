use spanned_error_message::{
    Location,
    SpannedErrorMessage,
};
use xee_xpath::error::ErrorValue;

use crate::errors::BunsenError;

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
/// Given the source, the message also marks the error's span, labelled
/// with the kind of error: "Parse Error", "Type Error", and so on.
/// `XPST0003` (a parse error) becomes [`BunsenError::ParseError`]; every
/// other error becomes [`BunsenError::External`].
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
            label: error_kind_label(&e.error).to_string(),
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
    use burn::nn::{
        Linear,
        LinearConfig,
    };

    use super::*;
    use crate::{
        burner::module::reflection::XmlModuleTree,
        support::testing::{
            CpuBackend,
            default_device,
        },
    };

    fn linear_tree() -> XmlModuleTree {
        let module: Linear<CpuBackend> = LinearConfig::new(2, 3).init(&default_device());
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
        let msg = mtree.select_param_ids(expr).unwrap_err().to_string();

        assert!(msg.contains("XPTY0004"), "{msg}");
        assert!(msg.contains("Type Error"), "{msg}");
        assert!(!msg.contains("Parse Error"), "{msg}");
    }

    /// A syntax error is labelled "Parse Error", and is a
    /// [`BunsenError::ParseError`].
    #[test]
    fn test_syntax_error_is_labelled_parse_error() {
        let mut mtree = linear_tree();

        let err = mtree.try_select("Linear/*[").err().unwrap();
        assert!(matches!(err, BunsenError::ParseError(_)), "{err:?}");

        let msg = err.to_string();
        assert!(msg.contains("XPST0003"), "{msg}");
        assert!(msg.contains("Parse Error"), "{msg}");
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
