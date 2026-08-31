//! # Array Utilities

/// Converts a `T` to a `[T; D]`.
pub fn scalar_to_array<const D: usize, T>(v: T) -> [T; D]
where
    T: Copy,
{
    [v; D]
}

#[cfg(test)]
mod tests {
    use crate::support::arrays::scalar_to_array;

    #[test]
    fn test_to_narray() {
        assert_eq!(scalar_to_array::<4, usize>(1), [1, 1, 1, 1]);
    }
}
