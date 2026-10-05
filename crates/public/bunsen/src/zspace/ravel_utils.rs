//! Row-major raveling of coordinates.

use burn::{
    prelude::Shape,
    tensor::AsIndex,
};

/// The row-major offset of `coords` in `shape`.
///
/// [`ravel_dims`] over the shape's dimensions; see it for negative
/// coordinates.
///
/// # Panics
///
/// If `coords` does not have one entry per dimension of `shape`, or if a
/// coordinate is out of range for its dimension.
#[track_caller]
pub fn ravel_shape<I: AsIndex>(
    shape: &Shape,
    coords: &[I],
) -> usize {
    ravel_dims(shape.as_slice(), coords)
}

/// The row-major offset of `coords` in a tensor of dimensions `dims`.
///
/// The last axis is contiguous: the offset is the sum of `coords[i]` times
/// the product of the dimensions after `i`.
///
/// A negative coordinate counts back from the end of its axis, so `-1` names
/// the last index. Each coordinate must be in `-dims[i]..dims[i]`.
///
/// # Panics
///
/// If `coords` does not have one entry per dimension, or if a coordinate is
/// outside `-dims[i]..dims[i]`. An out-of-range coordinate is not wrapped onto
/// another element: in `[2, 3]`, the coordinates `[0, 3]` panic.
#[track_caller]
pub fn ravel_dims<I: AsIndex>(
    dims: &[usize],
    coords: &[I],
) -> usize {
    assert_eq!(
        dims.len(),
        coords.len(),
        "Shape rank mismatch: expected {}, got {}",
        dims.len(),
        coords.len(),
    );

    let mut ravel_idx = 0;
    let mut stride = 1;

    for i in (0..dims.len()).rev() {
        let dim = dims[i];
        let idx = coords[i].as_index();
        let coord = if idx < 0 { idx + dim as isize } else { idx };
        if coord < 0 || coord as usize >= dim {
            coord_out_of_range(dims, coords, i);
        }

        ravel_idx += coord as usize * stride;
        stride *= dim;
    }

    ravel_idx
}

#[cold]
#[inline(never)]
#[track_caller]
fn coord_out_of_range<I: AsIndex>(
    dims: &[usize],
    coords: &[I],
    axis: usize,
) -> ! {
    panic!(
        "coordinate {} is out of range for axis {axis} of size {}: coords {coords:?} in dims {dims:?}",
        coords[axis].as_index(),
        dims[axis],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ravel_dims() {
        let dims = [2, 3, 4];
        assert_eq!(ravel_dims(&dims, &[0, 0, 0]), 0);
        assert_eq!(ravel_dims(&dims, &[0, 0, 3]), 3);
        assert_eq!(ravel_dims(&dims, &[0, 2, 0]), 2 * 4);
        assert_eq!(ravel_dims(&dims, &[1, 2, 3]), 12 + 2 * 4 + 3);
        assert_eq!(ravel_shape(&Shape::new(dims), &[1, 2, 3]), 12 + 2 * 4 + 3);
    }

    #[test]
    fn test_ravel_dims_wraps_negative_coords() {
        assert_eq!(ravel_dims(&[2, 3], &[-1, -1]), ravel_dims(&[2, 3], &[1, 2]));
        assert_eq!(ravel_dims(&[2, 3], &[-2, -3]), 0);
    }

    #[test]
    #[should_panic(expected = "coordinate 3 is out of range for axis 1 of size 3")]
    fn test_ravel_dims_rejects_coord_at_size() {
        // `[0, 3]` is past the end of axis 1. It names no element: neither
        // `[0, 0]` (wrapped) nor `[1, 0]` (offset 3).
        let offset = ravel_dims(&[2, 3], &[0, 3]);
        panic!("ravel_dims returned {offset}");
    }

    #[test]
    #[should_panic(expected = "coordinate -3 is out of range for axis 0 of size 2")]
    fn test_ravel_dims_rejects_coord_below_minus_size() {
        let offset = ravel_dims(&[2, 3], &[-3, 0]);
        panic!("ravel_dims returned {offset}");
    }
}
