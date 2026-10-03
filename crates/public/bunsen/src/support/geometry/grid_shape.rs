//! # Grid shapes

use core::str::FromStr;

use serde::{
    Deserialize,
    Serialize,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A 2D grid size: `width` by `height` cells.
///
/// It parses from a string ([`FromStr`]), either `"W,H"` or a single `"N"`
/// for an `N`-by-`N` square, so a command line or a config file can name a
/// grid. The sims kits' configs use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GridShape2D {
    /// The width, in cells.
    pub width: usize,

    /// The height, in cells.
    pub height: usize,
}

impl FromStr for GridShape2D {
    type Err = BunsenError;

    /// Parses `"W,H"`, or `"N"` for an `N`-by-`N` square.
    ///
    /// # Errors
    ///
    /// [`BunsenError::ParseError`] if the string has more than two parts, or
    /// a part is not a `usize`.
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
