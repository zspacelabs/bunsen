use core::fmt::Debug;

use num_traits::{
    Float,
    One,
    Zero,
};

use crate::errors::{
    BunsenResult,
    ConstraintError,
    WithOkOrPanic,
};

/// Checks that `prob` is a probability, in the closed range `[0.0, 1.0]`.
///
/// The fallible half of a `try_x` / `x` pair ([errors convention]); the
/// panicking half is [`expect_probability`].
///
/// # Errors
///
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
/// [`ConstraintError`] cause whose owner is `"probability"`, if `prob` is
/// below `0.0`, above `1.0`, or NaN. The check does not know which field it
/// guards: a caller adds that as context.
///
/// [errors convention]: crate::errors#convention-try_x-and-x
#[inline]
pub fn try_probability<F: Float + Debug>(prob: F) -> BunsenResult<F> {
    // Not `prob < 0 || prob > 1`: both are false for NaN.
    let in_range = prob >= F::zero() && prob <= F::one();
    if !in_range {
        Err(
            ConstraintError::out_of_range("probability", "", format!("{prob:?}"), "[0.0, 1.0]")
                .into(),
        )
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
/// With the [`try_probability`] error's report, if `prob` is out of range
/// or NaN.
#[inline]
pub fn expect_probability<F: Float + Debug>(prob: F) -> F {
    try_probability(prob).ok_or_panic()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

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

    #[should_panic(expected = "probability: -1.0 is outside [0.0, 1.0]")]
    #[test]
    fn test_probability_panic() {
        expect_probability(-1.0);
    }

    #[test]
    fn test_probability_rejects_nan() {
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_eq("probability: NaN is outside [0.0, 1.0]")
            .has_cause::<ConstraintError>()
            .assert_err(&try_probability(f32::NAN));
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .has_cause::<ConstraintError>()
            .assert_err(&try_probability(f64::NAN));
    }

    #[should_panic(expected = "probability: NaN is outside [0.0, 1.0]")]
    #[test]
    fn test_expect_probability_panics_on_nan() {
        expect_probability(f64::NAN);
    }
}
