//! The z-space partial order, and point-in-box checks.
use core::{
    cmp::Ordering,
    fmt::Debug,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    WithOkOrPanic,
};

/// The z-space partial order: compares two points coordinate by coordinate.
///
/// `a` is `Less` than `b` when no coordinate of `a` is greater than the
/// matching one of `b` and at least one is smaller; `Greater` is the mirror;
/// `Equal` is equality on every axis. A pair that is smaller on one axis and
/// greater on another, or that has an incomparable coordinate (a NaN), is
/// `None`. See the [module docs](crate::zspace).
///
/// For example, the following orderings would hold:
/// * `cmp([1, 2], [1, 2]) == Some(Ordering::Equal)`
/// * `cmp([0, 0], [0, 1]) == Some(Ordering::Less)`
/// * `cmp([1, 0], [0, 0]) == Some(Ordering::Greater)`
/// * `cmp([0, 0], [1, 1]) == Some(Ordering::Less)`
/// * `cmp([1, 0], [0, 1]) == None`
///
/// # Arguments
///
/// - `a`: the first slice to compare.
/// - `b`: the second slice to compare.
///
/// # Returns
///
/// An `Option<Ordering>`, where `None` represents incomparable.
///
/// # Panics
///
/// If called on slices of unequal lengths.
pub fn zspace_partial_cmp<T: PartialOrd>(
    a: &[T],
    b: &[T],
) -> Option<Ordering> {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} != {}",
        a.len(),
        b.len()
    );
    let mut ord = Ordering::Equal;
    for (ai, bi) in a.iter().zip(b.iter()) {
        match ai.partial_cmp(bi) {
            None => return None,
            Some(Ordering::Equal) => (),
            Some(Ordering::Less) => {
                if ord == Ordering::Greater {
                    return None;
                }
                ord = Ordering::Less;
            }
            Some(Ordering::Greater) => {
                if ord == Ordering::Less {
                    return None;
                }
                ord = Ordering::Greater;
            }
        }
    }
    Some(ord)
}

/// Checks that `point` is in the half-open box `[start, end)`.
///
/// The lower bound is `start <= point` in the
/// [partial order](zspace_partial_cmp): every coordinate at least `start`'s.
/// The upper bound is `point < end` in the same order.
///
/// # Known issue
///
/// `point < end` in the partial order holds when *some* coordinate of `point`
/// is below `end`'s and none is above, not when *every* coordinate is below.
/// So a point on a far face of the box passes: `[1, 3]` is accepted in
/// `[[0, 0], [2, 3])`, though `3` is not below `3`. Of the far faces, only
/// the corner `end` itself is rejected. This is tracked for repair.
///
/// # Returns
///
/// `Ok(())` if the point passes, else [`BunsenError::Invalid`] naming the
/// point and the box.
pub fn try_point_bounds_check<T>(
    point: &[T],
    start: &[T],
    end: &[T],
) -> BunsenResult<()>
where
    T: PartialOrd + Debug,
{
    if !matches!(
        zspace_partial_cmp(start, point),
        Some(Ordering::Less) | Some(Ordering::Equal)
    ) || zspace_partial_cmp(point, end) != Some(Ordering::Less)
    {
        Err(BunsenError::Invalid(format!(
            "{point:?} is not in [ {start:?}, {end:?} )"
        )))
    } else {
        Ok(())
    }
}

/// Expects that `point` is in the half-open box `[start, end)`.
///
/// The panicking half of [`try_point_bounds_check`], and it shares that
/// function's known issue.
///
/// # Panics
///
/// With the [`try_point_bounds_check`] error's message, if the check fails.
#[allow(dead_code)]
pub fn expect_point_bounds_check<T>(
    point: &[T],
    start: &[T],
    end: &[T],
) where
    T: PartialOrd + Debug,
{
    try_point_bounds_check(point, start, end).ok_or_panic()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[should_panic(expected = "length mismatch: 2 != 3")]
    #[test]
    fn test_zspace_partial_cmp_panic() {
        zspace_partial_cmp(&[2, 3], &[2, 3, 4]);
    }

    #[test]
    fn test_zspace_partial_cmp() {
        assert_eq!(zspace_partial_cmp(&[2, 3], &[2, 3]), Some(Ordering::Equal));
        assert_eq!(zspace_partial_cmp(&[2, 3], &[2, 4]), Some(Ordering::Less));
        assert_eq!(
            zspace_partial_cmp(&[2, 3], &[2, 2]),
            Some(Ordering::Greater)
        );
        assert_eq!(zspace_partial_cmp(&[4, 3], &[2, 4]), None);

        assert_eq!(
            zspace_partial_cmp(&[2.0, 3.0], &[2.0, 3.0]),
            Some(Ordering::Equal)
        );
        assert_eq!(
            zspace_partial_cmp(&[2.0, 3.0], &[2.0, 4.0]),
            Some(Ordering::Less)
        );
        assert_eq!(
            zspace_partial_cmp(&[2.0, 3.0], &[2.0, 2.0]),
            Some(Ordering::Greater)
        );
        assert_eq!(zspace_partial_cmp(&[4.0, 3.0], &[2.0, 4.0]), None);
    }

    #[test]
    fn test_zspace_bounds_check() {
        assert!(try_point_bounds_check(&[0, 0], &[0, 0], &[2, 3]).is_ok());
        assert!(try_point_bounds_check(&[0, 1], &[0, 0], &[2, 3]).is_ok());
        assert!(try_point_bounds_check(&[1, 0], &[0, 0], &[2, 3]).is_ok());
        assert!(try_point_bounds_check(&[1, 2], &[0, 0], &[2, 3]).is_ok());

        assert!(
            try_point_bounds_check(&[-1, 2], &[0, 0], &[2, 3])
                .unwrap_err()
                .to_string()
                .contains("[-1, 2] is not in [ [0, 0], [2, 3] )")
        );
    }
}
