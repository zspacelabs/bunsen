use burn::{
    Tensor,
    prelude::{
        Bool,
        s,
    },
};
use serial_test::serial;

use crate::{
    sims::lbm::d2q9::{
        collision::bgk_collision_with_spherical_reflection,
        relaxation::RelaxationParam,
        space,
        space::dbg_dist,
        streaming::outflow_clipping_stream,
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

#[test]
#[serial]
fn test_closed_box_steps_conserve_mass() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);

    let k = 5;
    let height = 6;
    let width = 6;
    let debug = false;

    let solid_mask: Tensor<2, Bool> = Tensor::full([height, width], false, &device)
        .slice_fill(s![0, ..], true)
        .slice_fill(s![-1, ..], true)
        .slice_fill(s![.., 0], true)
        .slice_fill(s![.., -1], true);

    let dist_t0: Tensor<4> = Tensor::zeros([height, width, 3, 3], &device)
        .slice_fill(s![.., .., 1, 1], 1.0)
        .slice_fill(s![1, 1, 1, 1], 3.0)
        .slice_fill(s![1, -2, 1, 1], 5.0)
        .slice_fill(s![-2, -2, 1, 1], 10.0);

    if debug {
        dbg_dist("dist_t0", dist_t0.clone());
    }

    let initial_mass: f64 = dist_t0.clone().sum().into_scalar();

    let lbm_tables = space::LbmTables::init(&device);

    let mut current = dist_t0.clone();

    for t_idx in 1..=k {
        let stream_phase = outflow_clipping_stream(current.clone());
        if debug {
            dbg_dist(format!("stream {t_idx}").as_str(), stream_phase.clone());
        }

        let thermal_phase = bgk_collision_with_spherical_reflection(
            stream_phase,
            solid_mask.clone(),
            RelaxationParam::Tau(1.0),
            None,
            &lbm_tables,
        );
        if debug {
            dbg_dist(format!("thermal {t_idx}").as_str(), thermal_phase.clone());
        }

        current = thermal_phase;
        let current_mass: f64 = current.clone().sum().into_scalar();

        assert!(
            (current_mass - initial_mass).abs() <= 1e-4 * initial_mass,
            "step {t_idx}: mass {current_mass} drifted from {initial_mass}"
        );
    }
}
