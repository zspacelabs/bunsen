use std::fmt::Debug;

use anyhow::bail;
use num_traits::Float;

/// Validate a probability in the range ``[0.0, 1.0]``.
///
/// # Arguments
///
/// - `prob`: the prob to check.
///
/// # Returns
///
/// An `anyhow::Result<prob>`: an error if `prob` is below `0.0`, above `1.0`,
/// or NaN.
pub fn try_probability<F: Float + Debug>(prob: F) -> anyhow::Result<F> {
    // Not `prob < 0 || prob > 1`: both are false for NaN.
    let in_range = prob >= F::zero() && prob <= F::one();
    if !in_range {
        bail!("probability must be in [0.0, 1.0]: {prob:?}");
    }
    Ok(prob)
}

/// Expect a probability to be in range ``[0.0, 1.0]``, or panic.
///
/// # Arguments
///
/// - `prob`: the prob to check.
///
/// # Returns
///
/// `prob`.
///
/// # Panics
///
/// With the [`try_probability`] error's message, if `prob` is out of range
/// or NaN.
pub fn expect_probability<F: Float + Debug>(prob: F) -> F {
    match try_probability(prob) {
        Ok(prob) => prob,
        Err(e) => panic!("{}", e),
    }
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
        assert!(try_probability(f32::NAN).is_err());
        assert!(try_probability(f64::NAN).is_err());
    }

    #[should_panic(expected = "probability must be in [0.0, 1.0]: NaN")]
    #[test]
    fn test_expect_probability_panics_on_nan() {
        expect_probability(f64::NAN);
    }
}
