//! # Thermal Equilibrium
use burn::{
    Tensor,
    prelude::Backend,
};

use crate::kits::sims::lbm::d2q9::{
    C2,
    C4,
    space,
    space::LbmTables,
};

/// Compute thermal equilibrium.
///
/// # Arguments
/// - `rho`: ``[H, W]`` population density
/// - `u`: ``[H, W, (Y, X)=2]`` macroscopic velocity
/// - `lbm_tables`: LBM Reference Tables.
///
/// # Returns
/// - `[H, W, Y=3, X=3]` equilibrium distribution
#[rustfmt::skip]
pub fn thermal_equilibrium<B: Backend>(
    rho: Tensor<B, 2>,
    u: Tensor<B, 3>,
    lbm_tables: &LbmTables<B>,
) -> Tensor<B, 4> {
    // [H, W, Y, X]
    let e_dot_u = lattice_dot_velocity(u.clone(), lbm_tables.e_vec());

    // [H, W]
    let u_sq = space::velocity_squared(u);

    // [H, W, 1, 1]
    let rho = rho.unsqueeze_dims::<4>(&[2, 3]);

    (lbm_tables.w().unsqueeze() * rho).mul(
        1.0
            + e_dot_u.clone() / C2
            + e_dot_u.square() / (2.0 * C4)
            - u_sq.unsqueeze_dims::<4>(&[2, 3]) / (2.0 * C2)
    )
}

/// Compute e·u for each lattice direction
///
/// # Arguments
/// - `e`: ``[Y=3, X=3, (Y, X)=2]`` direction vectors
/// - `u`: ``[H, W, (Y, X)=2]`` macroscopic velocity
///
/// # Returns
/// - ``[H, W, Y=3, X=3]`` dot product at each grid point and direction.
pub fn lattice_dot_velocity<B: Backend>(
    u: Tensor<B, 3>,
    e: Tensor<B, 3>,
) -> Tensor<B, 4> {
    ldv_projection(e, u).sum_dim(4).squeeze_dims::<4>(&[4])
}

/// The projection component of [`lattice_dot_velocity`].
///
/// # Arguments
/// - `e`: ``[Y=3, X=3, (Y, X)=2]`` direction vectors
/// - `u`: ``[H, W, (Y, X)=2]`` macroscopic velocity
///
/// # Returns
/// - ``[H, W, Y=3, X=3, (Y, X)=2]`` projection.
pub fn ldv_projection<B: Backend>(
    e: Tensor<B, 3>,
    u: Tensor<B, 3>,
) -> Tensor<B, 5> {
    // e[None, None, ... ] * u[..., None, :] -> [H, W, Y, X, 2]
    // e[1, 1, Y, X, (Y, X)=2] * u[H, W, 1, 1, (Y, X)=2]
    // -> [H, W, Y, X, (Y, X)]
    e.unsqueeze() * u.unsqueeze_dims::<5>(&[2, 3])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        DType::F32,
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::{
        kits::sims::lbm::d2q9::{
            space::{
                LbmTables,
                density,
                direction_vectors,
                moments,
                velocity_squared,
            },
            thermal::lattice_dot_velocity,
        },
        support::testing::PerfTestBackend,
    };

    #[test]
    fn test_equilibrium() {
        type B = PerfTestBackend;
        let device = Default::default();

        let dist = Tensor::<B, 4>::random([20, 20, 3, 3], Distribution::Default, &device);

        let lbm_tables: LbmTables<B> = LbmTables::for_dist(&dist);

        let (rho, u) = moments(dist.clone(), &lbm_tables);

        let equi_dist: Tensor<B, 4> = thermal_equilibrium(rho.clone(), u.clone(), &lbm_tables);

        density(equi_dist.clone())
            .to_data()
            .assert_approx_eq::<f32>(&rho.to_data(), Tolerance::default());

        let e_dot_u = lattice_dot_velocity(u.clone(), lbm_tables.e_vec());
        let u_sq = velocity_squared(u);
        let expected_eq = (lbm_tables.w().unsqueeze() * rho.unsqueeze_dim(2)).mul(
            1 + 3.0 * e_dot_u.clone() + 4.5 * e_dot_u.square()
                - 1.5 * u_sq.unsqueeze_dims::<4>(&[2, 3]),
        );

        equi_dist
            .clone()
            .to_data()
            .assert_approx_eq::<f32>(&expected_eq.to_data(), Tolerance::default());
    }

    #[test]
    fn test_equilibrium_invariants() {
        type B = PerfTestBackend;
        let device = Default::default();

        let dtype = F32;

        let dist = Tensor::<B, 4>::random([20, 20, 3, 3], Distribution::Uniform(0.1, 1.0), &device)
            .cast(dtype);

        let lbm_tables = LbmTables::for_dist(&dist);

        let (rho, u) = moments(dist.clone(), &lbm_tables);
        let equi_dist = thermal_equilibrium(rho.clone(), u.clone(), &lbm_tables);

        // Invariant: density(equilibrium(dist)) == density(dist)
        density(equi_dist.clone())
            .to_data()
            .assert_approx_eq::<f32>(&rho.to_data(), Tolerance::default());
    }

    #[test]
    fn test_lattice_dot_velocity() {
        type B = PerfTestBackend;
        let device = Default::default();

        let e: Tensor<B, 3> = direction_vectors(&device);

        let u: Tensor<B, 3> = Tensor::from_data([[[0.1, -2.], [0.5, -1.5]]], &device);

        let parts = ldv_projection(e.clone(), u.clone());

        parts.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 5>::from_data(
                [[
                    [
                        [[-0.1, 2.], [-0.1, 0.], [-0.1, -2.]],
                        [[0., 2.], [0., 0.], [0., -2.]],
                        [[0.1, 2.], [0.1, 0.], [0.1, -2.]],
                    ],
                    [
                        [[-0.5, 1.5], [-0.5, 0.], [-0.5, -1.5]],
                        [[0., 1.5], [0., 0.], [0., -1.5]],
                        [[0.5, 1.5], [0.5, 0.], [0.5, -1.5]],
                    ],
                ]],
                &device,
            )
            .to_data(),
            Tolerance::default(),
        );

        let e_u = lattice_dot_velocity(u.clone(), e.clone());

        e_u.clone().to_data().assert_approx_eq::<f32>(
            &parts.sum_dim(4).squeeze_dims::<4>(&[4]).to_data(),
            Tolerance::default(),
        );
    }
}
