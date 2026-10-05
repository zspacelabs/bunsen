use core::fmt::Debug;

use num_traits::{
    Float,
    One,
    Zero,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    WithOkOrPanic,
};

/// Checks that `prob` is a probability, in the closed range `[0.0, 1.0]`.
///
/// The fallible half of a `try_x` / `x` pair ([errors convention]); the
/// panicking half is [`expect_probability`].
///
/// # Errors
///
/// [`BunsenError::Invalid`] if `prob` is below `0.0`, above `1.0`, or NaN.
///
/// [errors convention]: crate::errors#convention-try_x-and-x
#[inline]
pub fn try_probability<F: Float + Debug>(prob: F) -> BunsenResult<F> {
    // Not `prob < 0 || prob > 1`: both are false for NaN.
    let in_range = prob >= F::zero() && prob <= F::one();
    if !in_range {
        Err(BunsenError::Invalid(format!(
            "probability must be in [0.0, 1.0]: {prob:?}"
        )))
    } else {
        Ok(prob)
    }
}

/// Returns `prob` if it is a probability, in `[0.0, 1.0]`.
///
/// The panicking half of [`try_probability`].
///
/// # Panics
///
/// With the [`try_probability`] error's message, if `prob` is out of range
/// or NaN.
#[inline]
pub fn expect_probability<F: Float + Debug>(prob: F) -> F {
    try_probability(prob).ok_or_panic()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability() {
        assert_eq!(expect_probability(0f32), 0f32);
        assert_eq!(expect_probability(1f32), 1f32);
        assert_eq!(expect_probability(0.5f32), 0.5f32);

        assert_eq!(expect_probability(0f64), 0f64);
        assert_eq!(expect_probability(1f64), 1f64);
        assert_eq!(expect_probability(0.5f64), 0.5f64);

        assert!(try_probability(-1.0f32).is_err());
        assert!(try_probability(2.0f32).is_err());

        assert!(try_probability(-1.0f64).is_err());
        assert!(try_probability(2.0f64).is_err());
    }

    #[should_panic(expected = "probability must be in [0.0, 1.0]: -1.0")]
    #[test]
    fn test_probability_panic() {
        expect_probability(-1.0);
    }

    #[test]
    fn test_probability_rejects_nan() {
        assert!(matches!(
            try_probability(f32::NAN),
            Err(BunsenError::Invalid(_))
        ));
        assert!(matches!(
            try_probability(f64::NAN),
            Err(BunsenError::Invalid(_))
        ));
    }

    #[should_panic(expected = "probability must be in [0.0, 1.0]: NaN")]
    #[test]
    fn test_expect_probability_panics_on_nan() {
        expect_probability(f64::NAN);
    }
}
