//! `WithOkOrPanic` and `ResultContext`: extension traits on results.

use core::fmt::Display;

use crate::errors::BunsenResult;

/// Adds [`ok_or_panic`](Self::ok_or_panic) to every `Result` whose error is
/// `Display`.
///
/// This is how the `x` half of a `try_x` / `x` pair is written:
/// `try_x(..).ok_or_panic()`. See the
/// [convention](crate::errors#the-try_x-and-x-convention).
pub trait WithOkOrPanic<T> {
    /// Unwraps the `Result`, or panics with the error's alternate `Display`
    /// (`{:#}`).
    ///
    /// For a [`BunsenError`](crate::errors::BunsenError), that is its full
    /// [report](crate::errors::BunsenError::report): kind, message, details,
    /// frames and causes, with their line breaks intact. [`Result::unwrap`]
    /// instead prints the error's `Debug` form after a fixed prefix. The panic
    /// points at the caller of `ok_or_panic`.
    #[track_caller]
    fn ok_or_panic(self) -> T;
}

impl<T, E> WithOkOrPanic<T> for Result<T, E>
where
    E: Display,
{
    #[track_caller]
    fn ok_or_panic(self) -> T {
        match self {
            Ok(t) => t,
            Err(e) => panic!("{e:#}"),
        }
    }
}

/// Adds context and re-marking to a [`BunsenResult`].
///
/// Each method leaves `Ok` untouched and changes only an `Err`. They are the
/// `Result` forms of the methods of the same names on
/// [`BunsenError`](crate::errors::BunsenError), and record the source location
/// of the call. See the [module docs](crate::errors#context-and-details).
pub trait ResultContext<T> {
    /// Adds a frame of context: what the code was doing.
    #[track_caller]
    fn context(
        self,
        frame: impl Display,
    ) -> BunsenResult<T>;

    /// Adds a frame of context built only on error.
    #[track_caller]
    fn with_context<D, F>(
        self,
        f: F,
    ) -> BunsenResult<T>
    where
        D: Display,
        F: FnOnce() -> D;

    /// Adds a frame of context with multi-line details, both built only on
    /// error: `f` returns `(frame, details)`.
    #[track_caller]
    fn context_details<M, D, F>(
        self,
        f: F,
    ) -> BunsenResult<T>
    where
        M: Display,
        D: Display,
        F: FnOnce() -> (M, D);

    /// Re-marks an `Illegal` error as `Policy`, at an input boundary. See
    /// [`BunsenError::as_policy`](crate::errors::BunsenError::as_policy).
    #[track_caller]
    fn as_policy(self) -> BunsenResult<T>;

    /// Re-marks an `Illegal` error as `Internal`. See
    /// [`BunsenError::as_internal`](crate::errors::BunsenError::as_internal).
    #[track_caller]
    fn as_internal(self) -> BunsenResult<T>;

    /// Re-marks a runtime error as `Illegal`. See
    /// [`BunsenError::as_illegal`](crate::errors::BunsenError::as_illegal).
    #[track_caller]
    fn as_illegal(self) -> BunsenResult<T>;
}

impl<T> ResultContext<T> for BunsenResult<T> {
    #[track_caller]
    fn context(
        self,
        frame: impl Display,
    ) -> BunsenResult<T> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(e.context(frame)),
        }
    }

    #[track_caller]
    fn with_context<D, F>(
        self,
        f: F,
    ) -> BunsenResult<T>
    where
        D: Display,
        F: FnOnce() -> D,
    {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(e.context(f())),
        }
    }

    #[track_caller]
    fn context_details<M, D, F>(
        self,
        f: F,
    ) -> BunsenResult<T>
    where
        M: Display,
        D: Display,
        F: FnOnce() -> (M, D),
    {
        match self {
            Ok(t) => Ok(t),
            Err(e) => {
                let (frame, details) = f();
                Err(e.context_details(frame, details))
            }
        }
    }

    #[track_caller]
    fn as_policy(self) -> BunsenResult<T> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(e.as_policy()),
        }
    }

    #[track_caller]
    fn as_internal(self) -> BunsenResult<T> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(e.as_internal()),
        }
    }

    #[track_caller]
    fn as_illegal(self) -> BunsenResult<T> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(e.as_illegal()),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::{
        String,
        ToString,
    };

    use super::*;
    use crate::errors::BunsenError;

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

    #[should_panic(expected = "error[Illegal]: stride is zero")]
    #[test]
    fn test_ok_or_panic_prints_the_report() {
        let result: BunsenResult<()> = Err(BunsenError::illegal("stride is zero"));
        result.ok_or_panic();
    }

    #[test]
    fn test_context_on_ok_is_untouched() {
        let r: BunsenResult<i32> = Ok(3);
        assert_eq!(r.context("x").as_illegal().unwrap(), 3);
    }
}
