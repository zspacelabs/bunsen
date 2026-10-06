//! The XML attribute codec for shapes.
use burn::prelude::Shape;

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    ParseError,
};

/// Encodes a [`Shape`] as an XML attribute value.
///
/// The dimensions, space-separated and without brackets: `"1"`, `"2 4"`.
/// Module reflection writes a parameter's `shape` attribute with it; see
/// [`shape_from_xml_attr`] for the inverse.
pub fn shape_to_xml_attr(shape: &Shape) -> String {
    shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<String>>()
        .join(" ")
}

/// Decodes a [`Shape`] from an XML attribute value.
///
/// The inverse of [`shape_to_xml_attr`]: space-separated dimensions without
/// brackets, e.g. `"2 4"`.
///
/// # Errors
///
/// [`Illegal`](BunsenErrorKind::Illegal), with a [`ParseError`] cause, if a
/// part is not a `usize`.
pub fn shape_from_xml_attr(val: &str) -> BunsenResult<Shape> {
    Ok(val
        .split_whitespace()
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<usize>, _>>()
        .map_err(|e| {
            BunsenError::from_cause(
                BunsenErrorKind::Illegal,
                ParseError::new("shape attribute").input(val).with_source(e),
            )
        })?
        .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_to_xml_attr() {
        assert_eq!(shape_to_xml_attr(&Shape::new([2])), "2");
        assert_eq!(shape_to_xml_attr(&Shape::new([2, 3, 4])), "2 3 4");
    }

    #[test]
    fn test_shape_from_xml_attr() {
        assert_eq!(shape_from_xml_attr("2").unwrap(), Shape::new([2]));
        assert_eq!(shape_from_xml_attr("2 3 4").unwrap(), Shape::new([2, 3, 4]));
    }

    #[test]
    fn test_shape_from_xml_attr_rejects_non_usize() {
        use crate::errors::testing::ErrorMatcher;

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_eq("cannot parse shape attribute \"2 x\": invalid digit found in string")
            .has_cause::<ParseError>()
            .assert_err(&shape_from_xml_attr("2 x"));
    }
}
