//! # Reflection Operations

use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
    },
};

/// Computes spherical solid reflection updates.
///
/// This point as a sphere, normal to all directions.
///
/// # Arguments
/// - `dist`: ``[H, W, VY=3, VX=3]`` pre-collision distribution
///
/// # Returns
/// - ``[H, W, VY=3, VX=3]`` distribution.
pub fn spherical_reflection<B: Backend>(dist: Tensor<B, 4>) -> Tensor<B, 4> {
    dist.flip([2, 3])
}

/// Applies isotropic spherical solid reflection updates to
/// [`operations::bgk_collision`].
///
/// This models every solid point as a sphere, normal to all directions.
///
/// # Arguments
/// - `pre_dist`: ``[H, W, VY=3, VX=3]`` pre-collision distribution
/// - `naive_dist`: ``[H, W, VY=3, VX=3]`` post-collision distribution
/// - `solid_mask`: ``[H, W]`` mask of solid locations.
///
/// # Returns
/// - ``[H, W, VY=3, VX=3]`` distribution.
pub fn with_spherical_reflection<B: Backend>(
    pre_dist: Tensor<B, 4>,
    naive_dist: Tensor<B, 4>,
    solid_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 4> {
    naive_dist.mask_where(
        solid_mask.unsqueeze_dims::<4>(&[-1, -1]),
        spherical_reflection(pre_dist),
    )
}
