//! # Range utilities

use std::ops::Range;

/// Converts a `usize` range to an `i32` range.
///
/// The bounds are cast with `as`, so a bound past `i32::MAX` wraps.
pub fn range_into(r: &Range<usize>) -> Range<i32> {
    r.start as i32..r.end as i32
}

/// Shifts both bounds of a range by `shift`.
pub fn shift_range(
    r: Range<i32>,
    shift: i32,
) -> Range<i32> {
    (r.start + shift)..(r.end + shift)
}
