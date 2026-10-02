//! A size relative to a reference size.

use serde::{
    Deserialize,
    Serialize,
};

/// A size, given relative to a reference size that is only known later.
///
/// [`resolve`](Self::resolve) turns it into a count once the reference is
/// known. Today the one consumer is [`DropBlockOptions`]: each side of its
/// `kernel` is a `SizeConfig`, resolved against the input's height or width,
/// so a block can be "7 pixels" or "a quarter of the image".
///
/// `From<usize>` builds [`Fixed`](Self::Fixed) and `From<f64>` builds
/// [`Ratio`](Self::Ratio), so `with_block_size(7)` and
/// `with_block_size(0.25)` both work.
///
/// [`DropBlockOptions`]: super::DropBlockOptions
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum SizeConfig {
    /// The reference size itself.
    #[default]
    Default,

    /// A ratio of the reference size, truncated toward zero.
    ///
    /// Must be >= 0.
    Ratio(f64),

    /// A fixed size; the reference is ignored.
    Fixed(usize),
}

impl SizeConfig {
    /// The reference size itself; the same as [`SizeConfig::Default`].
    pub fn source() -> Self {
        Self::Default
    }

    /// Resolves the size against a reference size.
    ///
    /// # Arguments
    ///
    /// - `source_size`: the reference size.
    ///
    /// # Returns
    ///
    /// The resolved size.
    ///
    /// # Panics
    ///
    /// If the size is a negative [`Ratio`](Self::Ratio).
    pub fn resolve(
        self,
        source_size: usize,
    ) -> usize {
        match self {
            SizeConfig::Default => source_size,
            SizeConfig::Ratio(ratio) => {
                assert!(ratio >= 0.0, "Ratio must be non-negative: {ratio}");
                ((source_size as f64) * ratio) as usize
            }
            SizeConfig::Fixed(size) => size,
        }
    }
}

impl From<usize> for SizeConfig {
    fn from(size: usize) -> Self {
        Self::Fixed(size)
    }
}

impl From<f64> for SizeConfig {
    fn from(ratio: f64) -> Self {
        Self::Ratio(ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_config() {
        assert_eq!(SizeConfig::default(), SizeConfig::Default);

        assert_eq!(SizeConfig::from(42), SizeConfig::Fixed(42));

        assert_eq!(SizeConfig::from(1.5), SizeConfig::Ratio(1.5));

        assert_eq!(SizeConfig::source(), SizeConfig::Default);
        assert_eq!(SizeConfig::source().resolve(50), 50);
    }
}
