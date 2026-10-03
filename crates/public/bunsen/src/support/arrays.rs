//! # Array Utilities

/// Repeats `v` into a `[T; D]`.
///
/// For passing one value where a per-axis array is expected, such as a
/// kernel size or a stride.
pub fn scalar_to_array<const D: usize, T>(v: T) -> [T; D]
where
    T: Copy,
{
    [v; D]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_narray() {
        assert_eq!(scalar_to_array::<4, usize>(1), [1, 1, 1, 1]);
    }
}
