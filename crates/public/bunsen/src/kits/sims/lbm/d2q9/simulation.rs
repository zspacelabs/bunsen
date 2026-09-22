#![allow(dead_code)]
//! # LBM D2Q9 World Module

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Backend,
        Bool,
        ElementConversion,
        s,
    },
    tensor::DType,
};

use super::{
    LbmTables,
    RelaxationParam,
    bgk_collision,
    outflow_clipping_stream,
    with_spherical_reflection,
};
use crate::{
    burner::{
        module::HasDType,
        tensor::TensorOpExt,
    },
    support::geometry::GridShape2D,
};

/// Introspection trait for [`LBMD2Q9State`]
pub trait LBMMeta {
    /// Returns the shape of the simulation: `[HEIGHT, WIDTH]`.
    fn shape(&self) -> GridShape2D;
}

/// Config for [`LBMD2Q9State`]
///
/// Specifies the `[HEIGHT, WIDTH]` grid shape and the [`RelaxationParam`]. Call
/// `.init(device, rho)` to build a relaxed [`LBMD2Q9State`] module ready to be
/// advanced step by step. Implements [`LBMMeta`].
#[derive(Config, Debug)]
pub struct LBMD2Q9Config {
    /// The shape of the simulation.
    pub shape: GridShape2D,

    /// Relaxation Param
    #[config(default = "RelaxationParam::Tau(0.5)")]
    pub relaxation: RelaxationParam,
}

impl LBMMeta for LBMD2Q9Config {
    fn shape(&self) -> GridShape2D {
        self.shape
    }
}

impl LBMD2Q9Config {
    /// Initializes a [`LBMD2Q9State`] module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
        rho: f64,
    ) -> LBMD2Q9State<B> {
        let height = self.shape.height;
        let width = self.shape.width;

        let solid_mask = Tensor::<B, 2>::zeros([height, width], device).bool();

        let lbm_tables = LbmTables::init(device);

        // Start off in a relaxed state. The border ring is zero, and it must
        // be: nothing later writes it from outside, and the mass total covers
        // the whole grid.
        let state = Tensor::<B, 4>::zeros([height, width, 3, 3], device).slice_assign(
            s![1..-1, 1..-1],
            lbm_tables
                .w()
                .unsqueeze::<4>()
                .expand([height - 2, width - 2, 3, 3])
                * rho,
        );
        let total_mass = state.clone().sum().into_scalar().elem();

        self.relaxation.validate();

        let omega =
            Tensor::<B, 2>::ones([height, width], device) * self.relaxation.as_omega_value();

        LBMD2Q9State {
            shape: self.shape,
            step_count: 0,
            dist: state,
            correct_total_mass: total_mass,
            solid_mask,
            lbm_tables,
            omega,
        }
    }
}

/// Lattice-Boltzmann Fluid Simulation State Module
///
/// Holds the D2Q9 population distribution `[H, W, 3, 3]`, the relaxation field,
/// solid mask, and [`LbmTables`] lattice constants. Construct it from a
/// [`LBMD2Q9Config`] via `.init(device, rho)`, then call
/// [`LBMD2Q9State::advance_step`] to stream, collide, and reflect the fluid one
/// step at a time. Implements [`LBMMeta`].
///
/// Built by [`LBMD2Q9Config`].
#[derive(Module, Debug)]
pub struct LBMD2Q9State<B: Backend> {
    /// The grid shape.
    pub shape: GridShape2D,

    /// The current simulation step.
    pub step_count: u64,

    /// Total Mass.
    pub correct_total_mass: f64,

    /// The grid velocity: `[H, W, UY=3, UX=3]`
    /// Here the 0-9 velocity terms are unfolded
    /// into the ``UY`` and ``UX`` dims.
    pub dist: Tensor<B, 4>,

    /// The solid mask: `[H, W]`
    pub solid_mask: Tensor<B, 2, Bool>,

    /// The relaxation field.
    pub omega: Tensor<B, 2>,

    /// Space Constants
    pub lbm_tables: LbmTables<B>,
}

impl<B: Backend> LBMMeta for LBMD2Q9State<B> {
    fn shape(&self) -> GridShape2D {
        self.shape
    }
}

impl<B: Backend> HasDType for LBMD2Q9State<B> {
    fn dtype(&self) -> DType {
        self.dist.dtype()
    }
}

impl<B: Backend> LBMD2Q9State<B> {
    /// Returns the device the module is on.
    pub fn device(&self) -> B::Device {
        self.dist.device()
    }

    /// Returns the datatype of the state.
    pub fn dtype(&self) -> DType {
        self.dist.dtype()
    }

    /// Recasts the datatype of the state.
    pub fn to_dtype(
        self,
        dtype: DType,
    ) -> Self {
        Self {
            dist: self.dist.cast(dtype),
            lbm_tables: self.lbm_tables.to_dtype(dtype),
            ..self
        }
    }

    /// Returns the current simulation step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Sets the current simulation step count.
    pub fn set_step_count(
        &mut self,
        step: u64,
    ) {
        self.step_count = step;
    }

    /// Resets the simulation step count to zero.
    pub fn reset_step_count(&mut self) {
        self.set_step_count(0)
    }

    /// Returns the mass correction term.
    ///
    /// # Panics
    ///
    /// If the current total mass is not finite and positive: the distribution
    /// is empty, or it has gone non-finite.
    pub fn correction_term(&self) -> f64 {
        let current = self.current_total_mass();
        assert!(
            current.is_finite() && current > 0.0,
            "degenerate total mass {current}: the distribution is empty or non-finite"
        );
        self.correct_total_mass / current
    }

    /// Advances the world simulation by one step.
    pub fn advance_step(&mut self) {
        // Everything the step reads from `self` is read here, before the
        // distribution is taken out of `self.dist` for the in-place update.
        let correction = self.correction_term();
        let omega = self.omega.clone();
        let lbm_tables = &self.lbm_tables;
        let solid_mask = self
            .solid_mask
            .clone()
            .slice_fill(s![0, ..], true)
            .slice_fill(s![-1, ..], true)
            .slice_fill(s![.., 0], true)
            .slice_fill(s![.., -1], true);

        self.dist.replace_with(|dist| {
            let stream_phase = outflow_clipping_stream(dist);

            with_spherical_reflection(
                stream_phase.clone(),
                bgk_collision(stream_phase, omega, Some(correction), lbm_tables),
                solid_mask,
            )
        });

        // TODO: better handle of numerical instability.
        // let dist = dist.clone().mask_fill(dist.is_finite().bool_not(), 0.0);

        self.step_count += 1;

        B::sync(&self.device()).unwrap();
    }

    /// Returns the current mass of the simm.
    pub fn current_total_mass(&self) -> f64 {
        self.dist.clone().sum().into_scalar().elem()
    }

    /// Saves the total energy of the system.
    pub fn save_correct_total_mass(&mut self) {
        self.correct_total_mass = self.current_total_mass();
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        kits::sims::lbm::d2q9::{
            SPEED_OF_SOUND,
            macroscopic_momentum,
        },
        support::testing::{
            DeviceMemoryGuard,
            PerformanceBackend,
            default_device,
        },
    };

    const RHO: f64 = SPEED_OF_SOUND / 100.0;

    /// A 16 x 24 world at rest density `RHO`.
    fn small_world<B: Backend>(device: &B::Device) -> LBMD2Q9State<B> {
        LBMD2Q9Config::new(GridShape2D {
            width: 24,
            height: 16,
        })
        .with_relaxation(RelaxationParam::Tau(0.9))
        .init(device, RHO)
    }

    #[test]
    #[serial]
    fn test_init_fills_the_interior_only() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let world = small_world::<B>(&device);

        assert_eq!(world.dist.dims(), [16, 24, 3, 3]);
        assert_eq!(world.step_count(), 0);

        // The border ring is zero, not whatever the allocator held.
        for ring in [
            world.dist.clone().slice(s![0]),
            world.dist.clone().slice(s![-1]),
            world.dist.clone().slice(s![.., 0]),
            world.dist.clone().slice(s![.., -1]),
        ] {
            let total: f64 = ring.abs().sum().into_scalar().elem();
            assert_eq!(total, 0.0);
        }

        // The interior carries `RHO` per cell, and the saved mass is that sum.
        let expected_mass = 14.0 * 22.0 * RHO;
        let mass = world.current_total_mass();
        assert!(
            (mass - expected_mass).abs() <= 1e-4 * expected_mass,
            "mass {mass} != {expected_mass}"
        );
        assert!((world.correct_total_mass - mass).abs() <= 1e-4 * expected_mass);
        assert!((world.correction_term() - 1.0).abs() <= 1e-6);
    }

    #[test]
    #[serial]
    fn test_advance_step_conserves_mass_and_stays_finite() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let mut world = small_world::<B>(&device);

        // A rest-population bump and a wall give the fluid something to do.
        world.dist = world.dist.slice_fill(s![5, 7, 1, 1], 5.0 * RHO);
        world.solid_mask = world.solid_mask.slice_fill(s![10, 4..12], true);
        world.save_correct_total_mass();
        let initial_mass = world.correct_total_mass;

        for _ in 0..20 {
            world.advance_step();
        }
        assert_eq!(world.step_count(), 20);

        // Every population is finite.
        let all_finite: bool = world.dist.clone().is_finite().all().into_scalar().elem();
        assert!(all_finite, "the distribution went non-finite");

        // Mass is conserved to float precision, so the correction stays ~1.
        let mass = world.current_total_mass();
        assert!(
            (mass - initial_mass).abs() <= 1e-4 * initial_mass,
            "mass drifted: {mass} vs {initial_mass}"
        );
        assert!((world.correction_term() - 1.0).abs() <= 1e-4);

        // The bump has spread: momentum is nonzero somewhere.
        let max_momentum: f64 = macroscopic_momentum(world.dist.clone(), world.lbm_tables.e_vec())
            .abs()
            .max()
            .into_scalar()
            .elem();
        assert!(max_momentum > 0.0);
    }

    #[test]
    #[serial]
    #[should_panic(expected = "degenerate total mass")]
    fn test_correction_term_rejects_a_degenerate_mass() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let mut world = small_world::<B>(&device);

        // What `extract()` leaves behind: an empty distribution.
        let _taken = world.dist.extract();
        let _ = world.correction_term();
    }
}
