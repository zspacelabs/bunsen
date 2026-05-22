//! Public Encode/Decode Utilites for Burn Types.
use burn::prelude::Shape;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Encode a [`Shape`] to XML/XPath attribute style.
///
/// Bracketless space-seperated style, eg: "1", "2 4"
///
/// See [`shape_from_xml_attr`].
pub fn shape_to_xml_attr(shape: &Shape) -> String {
    shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<String>>()
        .join(" ")
}

/// Decode a [`Shape`] from XML/XPath attribute style.
///
/// Bracketless space-seperated style, eg: "1", "2 4"
///
/// See [`shape_to_xml_attr`].
pub fn shape_from_xml_attr(val: &str) -> BunsenResult<Shape> {
    Ok(val
        .split_whitespace()
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<usize>, _>>()
        .map_err(|e| BunsenError::External(format!("Invalid shape: {}", e)))?
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
}
