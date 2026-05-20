//! # Shape Helpers

use burn::{
    prelude::Shape,
    tensor::{
        AsIndex,
        wrap_index,
    },
};

/// Compute the ravel index for the given coordinates.
///
/// This returns the row-major order raveling.
///
/// # Arguments
/// - `coords`: must be the same size as `self.rank()`.
///
/// # Returns
/// - the ravel offset index.
pub fn ravel_shape<I: AsIndex>(
    shape: &Shape,
    coords: &[I],
) -> usize {
    ravel_dims(shape.as_slice(), coords)
}

/// Compute the ravel index for the given coordinates.
///
/// This returns the row-major order raveling.
///
/// # Arguments
/// - `dims`: the dimensions of the shape.
/// - `coords`: must be the same size as `self.rank()`.
///
/// # Returns
/// - the ravel offset index.
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
