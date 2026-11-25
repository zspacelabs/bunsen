//! Channel counts rounded to a divisor.

/// `value` rounded to the nearest multiple of `divisor`, at least
/// `min_value` (`divisor` when `None`), and, when that rounds below
/// `round_limit * value`, one `divisor` more: timm's (and `TensorFlow`'s
/// `MobileNet`'s) `make_divisible`, which sizes a squeeze-excite's reduction
/// and `RegNet`'s and `EfficientNet`'s widths.
///
/// ```rust
/// use bunsen::support::math::make_divisible;
///
/// // A squeeze-excite's reduction of 256 channels at 1/16, rounded to 8s.
/// assert_eq!(make_divisible(256.0 / 16.0, 8, None, 0.0), 16);
/// // 1.5 rounds up to the floor of 8.
/// assert_eq!(make_divisible(1.5, 8, None, 0.0), 8);
/// // 43 rounds to 40, within 90% of 43.
/// assert_eq!(make_divisible(43.0, 8, None, 0.9), 40);
/// ```
pub fn make_divisible(
    value: f64,
    divisor: usize,
    min_value: Option<usize>,
    round_limit: f64,
) -> usize {
    let min_value = min_value.unwrap_or(divisor);
    // `int(v + divisor / 2) // divisor * divisor`: `int` truncates.
    let rounded = (value + divisor as f64 / 2.0) as usize / divisor * divisor;
    let mut new_value = min_value.max(rounded);
    if (new_value as f64) < round_limit * value {
        new_value += divisor;
    }
    new_value
}

#[cfg(test)]
mod tests {
    use super::*;

    /// timm's values (computed by `timm.layers.make_divisible`).
    #[test]
    fn test_make_divisible_is_timms() {
        for (channels, sixteenth, quarter) in [
            (64usize, 8usize, 16usize),
            (256, 16, 64),
            (2048, 128, 512),
            (96, 8, 24),
            (40, 8, 8),
        ] {
            let c = channels as f64;
            assert_eq!(
                make_divisible(c / 16.0, 8, None, 0.0),
                sixteenth,
                "{channels}"
            );
            assert_eq!(
                make_divisible(c * 0.25, 8, None, 0.0),
                quarter,
                "{channels}"
            );
        }
        // Rounding down by more than the limit adds a divisor; a floor holds.
        assert_eq!(make_divisible(11.0, 8, None, 0.9), 16);
        assert_eq!(make_divisible(3.0, 8, Some(4), 0.9), 4);
    }
}
