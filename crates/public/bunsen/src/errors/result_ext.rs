//! `WithOkOrPanic`: the panicking half of the `try_x` / `x` convention.

use core::fmt::Display;

/// Adds [`ok_or_panic`](Self::ok_or_panic) to every `Result` whose error is
/// `Display`.
///
/// This is how the `x` half of a `try_x` / `x` pair is written:
/// `try_x(..).ok_or_panic()`. See the
/// [convention](crate::errors#convention-try_x-and-x).
pub trait WithOkOrPanic<T> {
    /// Unwraps the `Result`, or panics with the error message.
    ///
    /// The panic message is the error's `Display` text, as is.
    /// [`Result::unwrap`] instead prints the error's `Debug` form after a
    /// fixed prefix, and `BunsenError`'s derived `Debug` quotes each message
    /// and escapes its newlines, so a multi-line message (an audit mismatch,
    /// say) would print as one escaped line.
    fn ok_or_panic(self) -> T;
}

impl<T, E> WithOkOrPanic<T> for Result<T, E>
where
    E: Display,
{
    fn ok_or_panic(self) -> T {
        match self {
            Ok(t) => t,
            Err(e) => panic!("{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_example(
        value: i32,
        throw: bool,
    ) -> Result<i32, String> {
        if throw {
            Err("throwing".to_string())
        } else {
            Ok(value)
        }
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
