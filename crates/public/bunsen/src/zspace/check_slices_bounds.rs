//! # Tensor Indexing Utilities
use std::{
    fmt::{
        Display,
        Formatter,
    },
    string::ToString,
};

use burn::{
    prelude::Shape,
    tensor::Slice,
};

use crate::errors::SlicingError;

/// Attempts to wrap a (potentially negative) index into a length bound.
///
/// # Arguments
/// - `idx`: The index to wrap, potentially a negative value to count back from
///   the end.
/// - `size`: The length bound.
///
/// # Result
///
/// Either a positive `Some(usize)` <= the `size`, or `None`.
fn maybe_wrap_index(
    idx: isize,
    size: usize,
) -> Option<usize> {
    if idx >= 0 {
        if (idx as usize) < size {
            Some(idx as usize)
        } else {
            None
        }
    } else {
        let idx = size as isize + idx;
        if idx >= 0 { Some(idx as usize) } else { None }
    }
}

/// A custom formatter for `Slice`.
/// [`Slice`] implements [`ToString`] at `burn` head.
struct SliceDisplay<'a>(&'a Slice);

impl Display for SliceDisplay<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> core::fmt::Result {
        let slice = self.0;

        if slice.step == 1
            && let Some(end) = slice.end
            && slice.start == end - 1
        {
            f.write_fmt(format_args!("{}", slice.start))
        } else {
            if slice.start != 0 {
                f.write_fmt(format_args!("{}", slice.start))?;
            }
            f.write_str("..")?;
            if let Some(end) = slice.end {
                f.write_fmt(format_args!("{}", end))?;
            }
            if slice.step != 1 {
                f.write_fmt(format_args!(";{}", slice.step))?;
            }
            Ok(())
        }
    }
}
fn format_slice_list(slices: &[Slice]) -> String {
    slices
        .iter()
        .map(|s| SliceDisplay(s).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_shape(shape: &Shape) -> String {
    let dim_list = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", dim_list)
}

/// Checks that the given slices are valid for the given tensor shape.
///
/// # Arguments
/// - `shape`: The tensor shape.
/// - `slices`: The slices to check.
///
/// # Returns
///
/// A `BunsenResult<()>`.
pub fn check_slices_bounds(
    shape: &Shape,
    slices: &[Slice],
) -> Result<(), SlicingError> {
    let rank = shape.rank();
    let k = slices.len();
    if k > rank {
        return Err(SlicingError::InvalidRank {
            msg: format!(
                "Slices [{}] length ({k}) is greater than shape {} rank ({rank})",
                format_slice_list(slices),
                format_shape(shape)
            ),
            shape: shape.clone(),
            slices: slices.to_vec(),
        });
    }
    for (dim, slice) in slices.iter().enumerate() {
        let bounds = shape[dim];

        if maybe_wrap_index(slice.start, bounds).is_none()
            || (slice.end.is_some() && maybe_wrap_index(slice.end.unwrap(), bounds + 1).is_none())
        {
            return Err(SlicingError::OutOfBounds {
                msg: format!(
                    "Slices [{}] out of bounds for tensor shape {}",
                    format_slice_list(slices),
                    format_shape(shape)
                ),
                shape: shape.clone(),
                slices: slices.to_vec(),
            });
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_wrap_index() {
        assert_eq!(maybe_wrap_index(0, 10), Some(0));
        assert_eq!(maybe_wrap_index(9, 10), Some(9));
        assert_eq!(maybe_wrap_index(10, 10), None);

        assert_eq!(maybe_wrap_index(-1, 10), Some(9));
        assert_eq!(maybe_wrap_index(-10, 10), Some(0));
        assert_eq!(maybe_wrap_index(-11, 10), None);
    }

    #[test]
    fn test_check_slices_bounds() {
        let shape = Shape::new([4, 5, 6]);
        let full = Slice::new(0, None, 1);

        assert!(check_slices_bounds(&shape, &[full.clone()]).is_ok());
        assert!(check_slices_bounds(&shape, &[full.clone(), full.clone()]).is_ok());
        assert!(check_slices_bounds(&shape, &[full.clone(), full.clone(), full.clone()]).is_ok());

        assert!(check_slices_bounds(&shape, &[Slice::new(0, Some(4), 1)]).is_ok());
        assert!(check_slices_bounds(&shape, &[Slice::new(-4, Some(0), -1)]).is_ok());

        match check_slices_bounds(
            &shape,
            &[full.clone(), full.clone(), full.clone(), full.clone()],
        )
        .unwrap_err()
        {
            SlicingError::InvalidRank {
                msg,
                shape: err_shape,
                slices: _,
            } => {
                assert_eq!(
                    msg,
                    "Slices [.., .., .., ..] length (4) is greater than shape [4, 5, 6] rank (3)"
                );
                assert_eq!(&err_shape, &shape);
            }
            err => panic!("Unexpected error type: {err:#?}"),
        }

        match check_slices_bounds(&shape, &[Slice::new(4, None, 1)]).unwrap_err() {
            SlicingError::OutOfBounds {
                msg,
                shape: err_shape,
                slices: _,
            } => {
                assert_eq!(msg, "Slices [4..] out of bounds for tensor shape [4, 5, 6]");
                assert_eq!(&err_shape, &shape);
            }
            err => panic!("Unexpected error type: {err:#?}"),
        }
    }
}
