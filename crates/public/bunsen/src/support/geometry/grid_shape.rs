//! # Grid shapes

use core::str::FromStr;

use serde::{
    Deserialize,
    Serialize,
};

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    ParseError,
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
    /// [`Policy`](BunsenErrorKind::Policy), with a [`ParseError`] cause, if
    /// the string has more than two parts, or a part is not a `usize`: the
    /// string is a request, such as a command-line value.
    fn from_str(s: &str) -> BunsenResult<Self> {
        let parse = |what: &'static str, part: &str| {
            part.parse::<usize>().map_err(|e| {
                BunsenError::from_cause(
                    BunsenErrorKind::Policy,
                    ParseError::new(what).input(part).with_source(e),
                )
            })
        };
        if s.contains(",") {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() != 2 {
                return Err(BunsenError::from_cause(
                    BunsenErrorKind::Policy,
                    ParseError::new("grid shape")
                        .input(s)
                        .because("must be WIDTH,HEIGHT or SIZE"),
                ));
            }
            let width = parse("grid width", parts[0])?;
            let height = parse("grid height", parts[1])?;
            Ok(Self { width, height })
        } else {
            let size = parse("grid size", s)?;
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
    use crate::errors::testing::ErrorMatcher;

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

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq("cannot parse grid shape \"10,10,10\": must be WIDTH,HEIGHT or SIZE")
            .has_cause::<ParseError>()
            .assert_err(&GridShape2D::from_str("10,10,10"));

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq("cannot parse grid height \"x\": invalid digit found in string")
            .assert_err(&GridShape2D::from_str("10,x"));

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("cannot parse grid size \"-1\"")
            .assert_err(&GridShape2D::from_str("-1"));
    }
}
