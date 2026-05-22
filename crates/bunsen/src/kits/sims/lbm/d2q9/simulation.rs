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

/// Introspection trait for [`LBMD2Q9State`]
pub trait LBMMeta {
    /// Get the shape of the simulation: `[HEIGHT, WIDTH]`
    fn shape(&self) -> [usize; 2];

    /// Get the height of the simulation.
    fn height(&self) -> usize {
        self.shape()[0]
    }

    /// Get the width of the simulation.
    fn width(&self) -> usize {
        self.shape()[1]
    }
}

/// Config for [`LBMD2Q9State`]
///
/// Implements [`LBMMeta`].
#[derive(Config, Debug)]
pub struct LBMD2Q9Config {
    /// The shape of the simulation: `[HEIGHT, WIDTH]`
    pub shape: [usize; 2],

    /// Relaxation Param
    #[config(default = "RelaxationParam::Tau(0.5)")]
    pub relaxation: RelaxationParam,
}

impl LBMMeta for LBMD2Q9Config {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl LBMD2Q9Config {
    /// Initialize a [`LBMD2Q9State`] module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
        rho: f64,
    ) -> LBMD2Q9State<B> {
        let [height, width] = self.shape;

        let solid_mask = Tensor::<B, 2>::zeros([height, width], device).bool();

        let lbm_tables = LbmTables::init(device);

        // Start off in a relaxed state.
        let state = Tensor::<B, 4>::empty([height, width, 3, 3], device).slice_assign(
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
#[derive(Module, Debug)]
pub struct LBMD2Q9State<B: Backend> {
    /// The current simulation step.
    pub step_count: u64,

    /// Total Mass.
    pub correct_total_mass: f64,

    /// The grid velocity: ``[H, W, UY=3, UX=3]``
    /// Here the 0-9 velocity terms are unfolded
    /// into the ``UY`` and ``UX`` dims.
    pub dist: Tensor<B, 4>,

    /// The solid mask: ``[H, W]``
    pub solid_mask: Tensor<B, 2, Bool>,

    /// The relaxation field.
    pub omega: Tensor<B, 2>,

    /// Space Constants
    pub lbm_tables: LbmTables<B>,
}

impl<B: Backend> LBMMeta for LBMD2Q9State<B> {
    fn shape(&self) -> [usize; 2] {
        let [h, w, _, _] = self.dist.dims();
        [h, w]
    }
}

impl<B: Backend> LBMD2Q9State<B> {
    /// Get the device the module is on.
    pub fn device(&self) -> B::Device {
        self.dist.device()
    }

    /// Get the datatype of the state.
    pub fn dtype(&self) -> DType {
        self.dist.dtype()
    }

    /// Recast the datatype of the state.
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

    /// Get the current simulation step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Set the current simulation step count.
    pub fn set_step_count(
        &mut self,
        step: u64,
    ) {
        self.step_count = step;
    }

    /// Reset the simulation step count to zero.
    pub fn reset_step_count(&mut self) {
        self.set_step_count(0)
    }

    /// Get the mass correction term.
    pub fn correction_term(&self) -> f64 {
        self.correct_total_mass / self.current_total_mass()
    }

    /// Advance the world simulation by one step.
    pub fn advance_step(&mut self) {
        let dist = self.dist.clone();

        let solid_mask = self
            .solid_mask
            .clone()
            .slice_fill(s![0, ..], true)
            .slice_fill(s![-1, ..], true)
            .slice_fill(s![.., 0], true)
            .slice_fill(s![.., -1], true);

        let stream_phase = outflow_clipping_stream(dist);

        let thermal_phase = with_spherical_reflection(
            stream_phase.clone(),
            bgk_collision(
                stream_phase.clone(),
                self.omega.clone(),
                Some(self.correction_term()),
                &self.lbm_tables,
            ),
            solid_mask,
        );

        // TODO: better handle of numerical instability.
        // let dist = dist.clone().mask_fill(dist.is_finite().bool_not(), 0.0);

        self.dist = thermal_phase;
        self.step_count += 1;

        B::sync(&self.device()).unwrap();
    }

    /// Get the current mass of the simm.
    pub fn current_total_mass(&self) -> f64 {
        self.dist.clone().sum().into_scalar().elem()
    }

    /// Save the total energy of the system.
    pub fn save_correct_total_mass(&mut self) {
        self.correct_total_mass = self.current_total_mass();
    }
}
