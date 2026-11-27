//! `expect_variant!`: a `let .. else` that says what it found.

/// Binds a refutable pattern, or panics naming the pattern and the value it
/// met.
///
/// `expect_variant!(let PAT = EXPR)` is `let PAT = EXPR else { panic!(..) };`
/// whose panic prints the pattern and the value's `Debug`, so a test reaching
/// into an enum needs no message of its own; one may follow, as format
/// arguments: `expect_variant!(let PAT = EXPR, "the {k}th unit")`. The value
/// is evaluated once; its type must be `Debug`.
///
/// ```
/// use bunsen::support::testing::expect_variant;
///
/// let parsed: Result<u32, String> = "42".parse::<u32>().map_err(|e| e.to_string());
/// expect_variant!(let Ok(n) = &parsed);
/// assert_eq!(*n, 42);
/// ```
///
/// A miss panics with `expected Ok(n), got Err("..")`.
pub use crate::__expect_variant as expect_variant;

#[doc(hidden)]
#[macro_export]
macro_rules! __expect_variant {
    (let $pattern:pat = $value:expr $(,)?) => {
        let __expect_variant_value = $value;
        let $pattern = __expect_variant_value else {
            ::core::panic!(
                "expected {}, got {:?}",
                ::core::stringify!($pattern),
                __expect_variant_value
            )
        };
    };
    (let $pattern:pat = $value:expr, $($message:tt)+) => {
        let __expect_variant_value = $value;
        let $pattern = __expect_variant_value else {
            ::core::panic!(
                "{}: expected {}, got {:?}",
                ::core::format_args!($($message)+),
                ::core::stringify!($pattern),
                __expect_variant_value
            )
        };
    };
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use super::*;

    #[derive(Debug)]
    enum Shape {
        Circle(String),
        Square { side: u32 },
    }

    /// The pattern's bindings are the caller's, by reference or by move.
    #[test]
    fn test_binds() {
        let shape = Shape::Square { side: 3 };
        expect_variant!(let Shape::Square { side } = &shape);
        assert_eq!(*side, 3);
        expect_variant!(let Shape::Square { side: owned } = shape, "a square");
        assert_eq!(owned, 3);
    }

    #[test]
    #[should_panic(expected = "expected Shape::Square { side }, got Circle(\"c\")")]
    fn test_miss_names_the_pattern_and_value() {
        expect_variant!(let Shape::Square { side } = Shape::Circle("c".into()));
        let _ = side;
    }

    #[test]
    #[should_panic(expected = "unit 2: expected Shape::Circle(_), got Square { side: 1 }")]
    fn test_miss_with_a_message() {
        let k = 2;
        expect_variant!(let Shape::Circle(_) = Shape::Square { side: 1 }, "unit {k}");
    }
}
