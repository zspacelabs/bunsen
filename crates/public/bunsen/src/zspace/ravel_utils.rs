//! Row-major raveling of coordinates.

use burn::{
    prelude::Shape,
    tensor::{
        AsIndex,
        wrap_index,
    },
};

/// The row-major offset of `coords` in `shape`.
///
/// [`ravel_dims`] over the shape's dimensions; see it for how coordinates are
/// wrapped.
///
/// # Panics
///
/// If `coords` does not have one entry per dimension of `shape`.
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
/// Each coordinate is first wrapped into its dimension, modulo its size, so
/// `-1` names the last index. An index at or past the size wraps too, rather
/// than panicking: in `[2, 3]`, the coordinates `[0, 3]` give the offset of
/// `[0, 0]`.
///
/// # Panics
///
/// If `coords` does not have one entry per dimension.
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
        let coord = wrap_index(coords[i], dim);

        ravel_idx += coord * stride;
        stride *= dim;
    }

    ravel_idx
}
