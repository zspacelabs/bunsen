//! # D2Q9 Lattice-Boltzmann Fluid Simulation
//!
//! This module contains the operations used by the D2Q9 LBM simulation.
//!
//! See:
//! * [Wikipedia](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods).
//!
//! ## Distribution Structure
//!
//! This is a "D2Q9" LBM Grid; "D2" for 2-dimension, "Q9" for 9-direction.
//!
//! This library uses a ``[H, W, VY=3, VX=3]`` layout.
//! * ``(H, W)`` determines a cell's spatial location.
//! * at a given ``(H, W)`` point, the 3x3 ``[VY=3, VX=3]`` grid describes the
//!   local moving particle population.
//!
//! The ``[VY=3, VX=3]`` populations are each moving away from the center
//! at ``(1, 1)``, which is stationary.
//!
//! The ``(y, x)`` directions correspond with the direction vectors ``(y-1,
//! x-1)``; so ``[H, W, 0, 0]`` is the population at ``(H, W)`` which is moving
//! in the ``(-1, -1)`` direction.
//!
//! This direction is also available in the `direction_vectors()` interface.

pub use crate::support::math::FRAC_1_SQRT_3;

pub mod collision;
pub mod reflection;
pub mod relaxation;
pub mod simulation;
pub mod space;
pub mod streaming;
pub mod thermal;

/// The speed of sound.
pub const SPEED_OF_SOUND: f64 = FRAC_1_SQRT_3;
/// The speed of sound squared.
pub const C2: f64 = SPEED_OF_SOUND * SPEED_OF_SOUND;
/// The speed of sound cubed.
pub const C4: f64 = C2 * C2;

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        prelude::{
            Bool,
            ElementConversion,
            s,
        },
    };
    use nearly::nearly;

    use crate::{
        kits::sims::surface::fluids::lbm::d2q9::{
            collision::bgk_collision_with_spherical_reflection,
            relaxation::RelaxationParam,
            space,
            space::dbg_dist,
            streaming::outflow_clipping_stream,
        },
        support::testing::PerfTestBackend,
    };

    #[test]
    fn test_debug_flow_loss() {
        type B = PerfTestBackend;
        let device = Default::default();

        let k = 5;
        let height = 6;
        let width = 6;
        let debug = false;

        let solid_mask: Tensor<B, 2, Bool> = Tensor::full([height, width], false, &device)
            .slice_fill(s![0, ..], true)
            .slice_fill(s![-1, ..], true)
            .slice_fill(s![.., 0], true)
            .slice_fill(s![.., -1], true);

        let dist_t0: Tensor<B, 4> = Tensor::zeros([height, width, 3, 3], &device)
            .slice_fill(s![.., .., 1, 1], 1.0)
            .slice_fill(s![1, 1, 1, 1], 3.0)
            .slice_fill(s![1, -2, 1, 1], 5.0)
            .slice_fill(s![-2, -2, 1, 1], 10.0);

        if debug {
            dbg_dist("dist_t0", dist_t0.clone());
        }

        let initial_energy: f64 = dist_t0.clone().sum().into_scalar().elem();

        let lbm_tables = space::LbmTables::init(&device);

        let mut current = dist_t0.clone();

        for t_idx in 1..=k {
            let stream_phase = outflow_clipping_stream(current.clone());
            if debug {
                dbg_dist(
                    format!("stream {t_idx}").to_string().as_str(),
                    stream_phase.clone(),
                );
            }

            if debug {
                dbg_dist(
                    format!("stream {t_idx}").to_string().as_str(),
                    stream_phase.clone(),
                );
            }

            let thermal_phase = bgk_collision_with_spherical_reflection(
                stream_phase,
                solid_mask.clone(),
                RelaxationParam::Tau(1.0),
                None,
                &lbm_tables,
            );
            if debug {
                dbg_dist(
                    format!("thermal {t_idx}").to_string().as_str(),
                    thermal_phase.clone(),
                );
            }

            current = thermal_phase;
            let current_energy: f64 = dist_t0.clone().sum().into_scalar().elem();

            assert!(nearly!(initial_energy == current_energy));
        }
    }
}
