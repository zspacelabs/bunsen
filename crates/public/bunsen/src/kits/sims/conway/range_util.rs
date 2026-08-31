//! # Range utilities

use std::ops::Range;

/// Convert a Range.
pub fn range_into(r: &Range<usize>) -> Range<i32> {
    r.start as i32..r.end as i32
}

/// Shift a range.
pub fn shift_range(
    r: Range<i32>,
    shift: i32,
) -> Range<i32> {
    (r.start + shift)..(r.end + shift)
}
