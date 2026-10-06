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
/// The test is per axis: `start[i] <= point[i] < end[i]` on every axis `i`.
/// The lower bound is `start <= point` in the
/// [partial order](zspace_partial_cmp). The upper bound is stricter than
/// `point < end` in that order, which would admit a point on a far face of
/// the box: `[1, 3]` is not in `[[0, 0], [2, 3])`, since `3` is not below `3`.
///
/// # Returns
///
/// `Ok(())` if the point passes. An incomparable coordinate (a NaN) fails.
///
/// # Errors
///
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), naming the point
/// and the box, if the point is outside the box.
///
/// # Panics
///
/// If `point`, `start` and `end` differ in length.
pub fn try_point_bounds_check<T>(
    point: &[T],
    start: &[T],
    end: &[T],
) -> BunsenResult<()>
where
    T: PartialOrd + Debug,
{
    for (a, b) in [(start, point), (point, end)] {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} != {}",
            a.len(),
            b.len()
        );
    }
    let inside = point
        .iter()
        .zip(start.iter().zip(end.iter()))
        .all(|(p, (s, e))| s <= p && p < e);
    if !inside {
        Err(BunsenError::illegal(format!(
            "{point:?} is not in [ {start:?}, {end:?} )"
        )))
    } else {
        Ok(())
    }
}

/// Expects that `point` is in the half-open box `[start, end)`.
///
/// The panicking half of [`try_point_bounds_check`]: the same per-axis test.
///
/// # Panics
///
/// With the [`try_point_bounds_check`] error's message, if the check fails;
/// and if `point`, `start` and `end` differ in length.
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

    #[test]
    fn test_point_bounds_check_rejects_far_faces() {
        let (start, end) = ([0, 0], [2, 3]);

        // The far face of axis 1 (`p[1] == 3`), off the corner.
        assert!(try_point_bounds_check(&[1, 3], &start, &end).is_err());
        assert!(try_point_bounds_check(&[0, 3], &start, &end).is_err());
        // The far face of axis 0 (`p[0] == 2`), off the corner.
        assert!(try_point_bounds_check(&[2, 0], &start, &end).is_err());
        assert!(try_point_bounds_check(&[2, 2], &start, &end).is_err());
        // The corner `end` itself.
        assert!(try_point_bounds_check(&[2, 3], &start, &end).is_err());

        // Three axes: each far face, with the other coordinates inside.
        let (start, end) = ([0, 0, 0], [2, 3, 4]);
        assert!(try_point_bounds_check(&[1, 2, 3], &start, &end).is_ok());
        assert!(try_point_bounds_check(&[2, 2, 3], &start, &end).is_err());
        assert!(try_point_bounds_check(&[1, 3, 3], &start, &end).is_err());
        assert!(try_point_bounds_check(&[1, 2, 4], &start, &end).is_err());
    }

    #[test]
    #[should_panic(expected = "[1, 3] is not in [ [0, 0], [2, 3] )")]
    fn test_expect_point_bounds_check_panics_on_far_face() {
        expect_point_bounds_check(&[1, 3], &[0, 0], &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "length mismatch: 2 != 3")]
    fn test_point_bounds_check_length_mismatch_panics() {
        let _ = try_point_bounds_check(&[0, 0], &[0, 0], &[2, 3, 4]);
    }
}
