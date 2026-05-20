//! # Collision Operators

use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
    },
};

use crate::simulations::surface::fluids::lbm::d2q9::{
    reflection,
    relaxation,
    relaxation::{
        OmegaSource,
        RelaxationParam,
    },
    space,
    thermal,
};

/// Bhatnagar-Gross-Krook collision operator.
///
/// This operator makes no accounting for solids, reflection,
/// or boundaries.
///
/// ## Correction
///
/// The `correction` term is a scale applied to the result. It is provided
/// as a way for dynamic corrections to be computed as fused operations;
/// and will default to `1.0` for `None`.
///
/// # Arguments
/// - `dist`: ``[H, W, VY=3, VX=3]`` current distribution
/// - `relaxation`: relaxation parameter.
/// - `correction`: fused correction factor for the relaxation operator;
///   defaults to 1.0.
/// - `lbm_tables`: LBM Reference Tables.
///
/// # Returns
/// - `[H, W, VY=3, VX=3]` post-collision distribution
pub fn bgk_collision<B: Backend, S: Into<OmegaSource<B>>>(
    dist: Tensor<B, 4>,
    relaxation: S,
    correction: Option<f64>,
    lbm_tables: &space::LbmTables<B>,
) -> Tensor<B, 4> {
    let (source_rho, u) = space::moments(dist.clone(), lbm_tables);
    let eq_dist = thermal::thermal_equilibrium(source_rho.clone(), u, lbm_tables);
    relaxation::relaxed_sum(dist, eq_dist, relaxation, correction)
}

/// Combined bgk collision and isotropic reflection operator.
///
/// This combines:
/// - [`bgk_collision`]
/// - [`reflection::with_spherical_reflection`]
///
/// # Arguments
/// - `dist`: ``[H, W, VY=3, VX=3]`` pre-collision distribution
/// - `solid_mask`: ``[H, W]`` mask of solid locations.
/// - `correction`: fused correction factor for the relaxation operator;
///   defaults to 1.0.
/// - `lbm_tables`: LBM Reference Tables.
///
/// # Returns
/// - ``[H, W, VY=3, VX=3]`` distribution.
pub fn bgk_collision_with_spherical_reflection<B: Backend>(
    dist: Tensor<B, 4>,
    solid_mask: Tensor<B, 2, Bool>,
    relaxation: RelaxationParam,
    correction: Option<f64>,
    lbm_tables: &space::LbmTables<B>,
) -> Tensor<B, 4> {
    reflection::with_spherical_reflection(
        dist.clone(),
        bgk_collision(dist, relaxation, correction, lbm_tables),
        solid_mask,
    )
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::PerfTestBackend;
    use burn::{
        Tensor,
        tensor::{
            DType::F32,
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::simulations::surface::fluids::lbm::d2q9::{
        relaxation::RelaxationParam,
        space::density,
    };

    #[test]
    fn test_collision_invariants() {
        type B = PerfTestBackend;
        let device = Default::default();

        let dtype = F32;

        let dist = Tensor::<B, 4>::random([20, 20, 3, 3], Distribution::Uniform(0.1, 1.0), &device)
            .cast(dtype);
        let rho = density(dist.clone());

        let lbm_tables = space::LbmTables::for_dist(&dist);

        let col_dist = bgk_collision(dist.clone(), RelaxationParam::Omega(0.5), None, &lbm_tables);

        // Invariant: density(collision(dist, param)) == density(dist)
        density(col_dist.clone())
            .to_data()
            .assert_approx_eq::<f32>(&rho.clone().to_data(), Tolerance::default());

        // With Correction
        {
            let col_dist = bgk_collision(
                dist.clone(),
                RelaxationParam::Omega(0.5),
                Some(1.2),
                &lbm_tables,
            );

            // Invariant: density(collision(dist, param)) == density(dist)
            density(col_dist.clone())
                .to_data()
                .assert_approx_eq::<f32>(&(rho * 1.2).to_data(), Tolerance::default());
        }
    }
}
