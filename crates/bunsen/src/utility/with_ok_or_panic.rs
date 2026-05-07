//! # Result Utilities
//!
//! Methods for [`std::result::Result`] manipulation.

use core::fmt::Debug;

/// Extension trait for `Result<T, E>` to add `ok_or_panic` method.
pub trait WithOkOrPanic<T> {
    /// Unwraps the `Result`, or panics with the error message.
    ///
    /// This differs from the behavior of [`Result::unwrap`]
    /// in that the [`Debug`] format of the wrapped error is used
    /// directly as the panic message; and not escaped.
    fn ok_or_panic(self) -> T;
}

impl<T, E> WithOkOrPanic<T> for Result<T, E>
where
    E: Debug,
{
    fn ok_or_panic(self) -> T {
        match self {
            Ok(t) => t,
            Err(e) => panic!("{:?}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::bail;

    use super::*;

    fn try_example(
        value: i32,
        throw: bool,
    ) -> anyhow::Result<i32> {
        if throw { bail!("throwing") } else { Ok(value) }
    }

    #[test]
    fn test_expect_unwrap() {
        let result = try_example(42, false);
        assert_eq!(result.ok_or_panic(), 42);
    }

    #[should_panic(expected = "throwing")]
    #[test]
    fn test_expect_unwrap_panic() {
        let result = try_example(42, true);
        result.ok_or_panic();
    }
}
