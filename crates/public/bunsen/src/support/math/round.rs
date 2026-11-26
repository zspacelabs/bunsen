//! Rounding as Python and torch round.

/// `value`, non-negative, rounded to the nearest integer, ties to even: what
/// Python's `round` and `torch.round` give, without `std`'s float functions.
/// timm's width arithmetic (`RegNet`'s widths, a squeeze-excite's
/// reduction) rounds this way.
///
/// ```rust
/// use bunsen::support::math::round_ties_even;
///
/// assert_eq!(round_ties_even(2.5), 2);
/// assert_eq!(round_ties_even(3.5), 4);
/// assert_eq!(round_ties_even(2.75), 3);
/// ```
pub fn round_ties_even(value: f64) -> usize {
    let floor = value as usize;
    let frac = value - floor as f64;
    if frac > 0.5 || (frac == 0.5 && floor % 2 == 1) {
        floor + 1
    } else {
        floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_ties_even() {
        for (value, want) in [
            (0.0, 0),
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (2.25, 2),
            (2.75, 3),
            (8.0, 8),
        ] {
            assert_eq!(round_ties_even(value), want, "{value}");
        }
    }
}
