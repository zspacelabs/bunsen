//! A size stated against a reference that is known later.

use serde::{
    Deserialize,
    Serialize,
};

/// A size, stated as the reference size itself, a ratio of it, or a fixed
/// count, where the reference is only known later.
///
/// [`resolve`](Self::resolve) turns it into a count once the reference is
/// known; a [`Fixed`](Self::Fixed) size ignores the reference. In the crate,
/// [`DropBlockOptions`] uses it: each side of its `kernel` is a `SizeSpec`,
/// resolved against the input's height or width, so a block can be "7 pixels"
/// or "a quarter of the image".
///
/// `From<usize>` builds [`Fixed`](Self::Fixed) and `From<f64>` builds
/// [`Ratio`](Self::Ratio), so `with_block_size(7)` and
/// `with_block_size(0.25)` both work.
///
/// [`DropBlockOptions`]: super::DropBlockOptions
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum SizeSpec {
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

impl SizeSpec {
    /// The reference size itself; the same as [`SizeSpec::Default`].
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
            SizeSpec::Default => source_size,
            SizeSpec::Ratio(ratio) => {
                assert!(ratio >= 0.0, "Ratio must be non-negative: {ratio}");
                ((source_size as f64) * ratio) as usize
            }
            SizeSpec::Fixed(size) => size,
        }
    }
}

impl From<usize> for SizeSpec {
    fn from(size: usize) -> Self {
        Self::Fixed(size)
    }
}

impl From<f64> for SizeSpec {
    fn from(ratio: f64) -> Self {
        Self::Ratio(ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_spec() {
        assert_eq!(SizeSpec::default(), SizeSpec::Default);

        assert_eq!(SizeSpec::from(42), SizeSpec::Fixed(42));

        assert_eq!(SizeSpec::from(1.5), SizeSpec::Ratio(1.5));

        assert_eq!(SizeSpec::source(), SizeSpec::Default);
        assert_eq!(SizeSpec::source().resolve(50), 50);
    }
}
