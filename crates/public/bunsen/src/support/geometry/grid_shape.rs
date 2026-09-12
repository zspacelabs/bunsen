//! # Config Parsers

use core::str::FromStr;

use serde::{
    Deserialize,
    Serialize,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A representation of a grid shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GridShape2D {
    /// The width of the image.
    pub width: usize,

    /// The height of the image.
    pub height: usize,
}

impl FromStr for GridShape2D {
    type Err = BunsenError;

    /// Parses a shape string into a ``(W, H)`` tuple.
    ///
    /// Accepts:
    /// - ``SHAPE``: ``(SHAPE, SHAPE)``.
    /// - ``[W,H]``: ``(W, H)``.
    ///
    /// # Returns
    ///
    /// a result, or error message.
    fn from_str(s: &str) -> BunsenResult<Self> {
        if s.contains(",") {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() != 2 {
                return Err(BunsenError::ParseError(
                    "Shape must be in the format WIDTH,HEIGHT".to_string(),
                ));
            }
            let width = parts[0]
                .parse::<usize>()
                .map_err(|_| BunsenError::ParseError(format!("Invalid width: {}", parts[0])))?;
            let height = parts[1]
                .parse::<usize>()
                .map_err(|_| BunsenError::ParseError(format!("Invalid height: {}", parts[1])))?;
            Ok(Self { width, height })
        } else {
            let size = s
                .parse::<usize>()
                .map_err(|_| BunsenError::ParseError(format!("Invalid size: {s}")))?;
            Ok(Self::square(size))
        }
    }
}

impl GridShape2D {
    /// Creates a square grid shape.
    pub fn square(size: usize) -> Self {
        Self {
            width: size,
            height: size,
        }
    }

    /// Export as a `[HEIGHT, WIDTH]` array.
    pub fn as_height_width(&self) -> [usize; 2] {
        [self.height, self.width]
    }

    /// Export as a `[WIDTH, HEIGHT]` array.
    pub fn as_width_height(&self) -> [usize; 2] {
        [self.width, self.height]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_grid_shape() {
        assert_eq!(
            GridShape2D::from_str("10,10").unwrap(),
            GridShape2D {
                width: 10,
                height: 10
            }
        );
        assert_eq!(
            GridShape2D::from_str("10").unwrap(),
            GridShape2D {
                width: 10,
                height: 10
            }
        );

        assert_eq!(
            GridShape2D::from_str("10,10,10").expect_err("Expected an error"),
            BunsenError::ParseError("Shape must be in the format WIDTH,HEIGHT".to_string()),
        );
    }
}
