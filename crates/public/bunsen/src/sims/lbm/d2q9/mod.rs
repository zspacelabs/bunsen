//! # D2Q9 Lattice-Boltzmann Fluid Simulation
//!
//! A two-dimensional lattice-Boltzmann fluid with nine velocity directions
//! per cell, a BGK collision, bounce-back solid walls, and a correction
//! that holds the total mass. See
//! [Wikipedia](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods) for
//! the method.
//!
//! # Lifecycle
//!
//! 1. [`LBMD2Q9Config`] holds the `[H, W]` grid shape and the
//!    [`RelaxationParam`], the collision's relaxation time or frequency.
//! 2. [`init`](LBMD2Q9Config::init) takes a device and a rest density `rho`,
//!    and builds an [`LBMD2Q9State`] whose interior is fluid at rest at that
//!    density. The outer ring of the grid starts empty, and the total mass is
//!    recorded as the target the correction holds. Its fallible twin,
//!    [`try_init`](LBMD2Q9Config::try_init), returns an `Err` for a grid under
//!    3 cells on a side or a relaxation out of range.
//! 3. Between steps, the caller may edit the state's public tensors: the
//!    distribution, `dist`, to add density or flow; the `[H, W]` `solid_mask`,
//!    to place walls; the per-cell relaxation, `omega`. After an edit that
//!    changes the mass on purpose,
//!    [`save_correct_total_mass`](LBMD2Q9State::save_correct_total_mass) makes
//!    the new total the target.
//! 4. [`advance_step`](LBMD2Q9State::advance_step) runs one step in place, one
//!    per call, and counts it in [`step_count`](LBMD2Q9State::step_count).
//!
//! [`macroscopic_momentum`] and [`moments`] read the flow back out of the
//! distribution. `examples/lbm2d_vis` runs this lifecycle live, drawing the
//! momentum field.
//!
//! A step is:
//!
//! 1. **stream**: [`outflow_clipping_stream`] moves each population one cell
//!    along its direction. The outer ring keeps what streams into it, and what
//!    would stream off the grid is dropped;
//! 2. **collide**: [`bgk_collision`] relaxes each cell toward its
//!    [`thermal_equilibrium`] at the rate `omega`, scaled by the mass
//!    correction;
//! 3. **reflect**: [`with_spherical_reflection`] gives every solid cell its
//!    streamed populations back, reversed (bounce-back), in place of the
//!    collision's result. For the step, the outer ring of the grid counts as
//!    solid whatever the mask says, so what reaches the edge is sent back in,
//!    and the grid is a closed box.
//!
//! # The pieces
//!
//! The files are private, and their items are all here, flat:
//!
//! - **space**: the lattice and the moments. [`LbmTables`] holds the direction
//!   indices, direction vectors and equilibrium weights ([`direction_indices`],
//!   [`direction_vectors`], [`weight_matrix`]), and [`density`],
//!   [`macroscopic_momentum`], [`macroscopic_velocity`] and [`moments`] reduce
//!   a distribution to its fields.
//! - **thermal**: the equilibrium distribution of a density and a velocity,
//!   [`thermal_equilibrium`], with [`lattice_dot_velocity`]. It is the target
//!   of the collision, not a parameter.
//! - **relaxation**: the parameters. [`RelaxationParam`] is the relaxation time
//!   `tau` or frequency `omega = 1 / tau`, [`OmegaSource`] a scalar or a
//!   per-cell field of it, and [`relaxed_sum`] the blend
//!   `(1-omega)*f+omega*f_eq`.
//! - **streaming**: [`stream_interior_windows`] and
//!   [`outflow_clipping_stream`].
//! - **collision**: [`bgk_collision`], and
//!   [`bgk_collision_with_spherical_reflection`], which adds the walls.
//! - **reflection**: [`spherical_reflection`] and
//!   [`with_spherical_reflection`], which treat every solid cell as a sphere,
//!   normal to every direction.
//! - **simulation**: [`LBMD2Q9Config`], [`LBMD2Q9State`] and [`LBMMeta`].
//!
//! # Mass correction
//!
//! The state holds a target mass,
//! [`correct_total_mass`](LBMD2Q9State::correct_total_mass), and every
//! step steers the total back toward it, so a long run neither thickens
//! nor drains:
//!
//! 1. At the start of [`advance_step`](LBMD2Q9State::advance_step),
//!    [`correction_term`](LBMD2Q9State::correction_term) is the target over the
//!    current total, read before the distribution is touched. It panics if the
//!    total is not finite and positive: the distribution is empty or has gone
//!    non-finite.
//! 2. The collision applies it as a fused scale: [`bgk_collision`] hands it to
//!    [`relaxed_sum`], which returns `correction*((1-omega)*f+omega*f_eq)`. The
//!    relaxation alone keeps each cell's density, so the scale is what moves
//!    the total.
//! 3. Solid cells take their reflected populations instead, which the scale
//!    does not reach.
//!
//! The correction is measured before a step and applied during it, as one
//! factor over every fluid cell: drift is paid back a step late, spread
//! over the whole fluid, not where it arose. The target is the mass
//! [`init`](LBMD2Q9Config::init) built, until
//! [`save_correct_total_mass`](LBMD2Q9State::save_correct_total_mass)
//! replaces it; without that call, the steps scale a deliberate change of
//! mass away.
//!
//! # Distribution Structure
//!
//! This is a "D2Q9" LBM Grid; "D2" for 2-dimension, "Q9" for 9-direction.
//!
//! This library uses a `[H, W, VY=3, VX=3]` layout.
//! * ``(H, W)`` determines a cell's spatial location.
//! * at a given ``(H, W)`` point, the 3x3 `[VY=3, VX=3]` grid describes the
//!   local moving particle population.
//!
//! The `[VY=3, VX=3]` populations are each moving away from the center
//! at ``(1, 1)``, which is stationary.
//!
//! The ``(y, x)`` directions correspond with the direction vectors ``(y-1,
//! x-1)``; so `[H, W, 0, 0]` is the population at ``(H, W)`` which is moving
//! in the ``(-1, -1)`` direction.
//!
//! This direction is also available from [`direction_vectors`].
//!
//! # Example
//!
//! ```rust
//! use bunsen::{
//!     sims::lbm::d2q9::{
//!         LBMD2Q9Config,
//!         LBMD2Q9State,
//!         RelaxationParam,
//!         SPEED_OF_SOUND,
//!         macroscopic_momentum,
//!     },
//!     support::{
//!         geometry::GridShape2D,
//!         testing::{
//!             cpu_device,
//!             performance_device,
//!         },
//!     },
//! };
//! use burn::prelude::s;
//!
//! let device = cpu_device();
//! let rho = SPEED_OF_SOUND / 100.0;
//!
//! let mut sim: LBMD2Q9State = LBMD2Q9Config::new(GridShape2D::square(16))
//!     .with_relaxation(RelaxationParam::Tau(0.9))
//!     .init(&device, rho);
//!
//! // A dense spot in the rest population, a wall, and the mass to hold.
//! sim.dist = sim.dist.slice_fill(s![5, 7, 1, 1], 5.0 * rho);
//! sim.solid_mask = sim.solid_mask.slice_fill(s![10, 4..12], true);
//! sim.save_correct_total_mass();
//!
//! for _ in 0..5 {
//!     sim.advance_step();
//! }
//! assert_eq!(sim.step_count(), 5);
//!
//! // The `[H, W, (y, x)=2]` momentum field.
//! let momentum =
//!     macroscopic_momentum(sim.dist.clone(), sim.lbm_tables.e_vec());
//! assert_eq!(momentum.dims(), [16, 16, 2]);
//! ```

mod collision;
mod reflection;
mod relaxation;
mod simulation;
mod space;
mod streaming;
mod thermal;

pub use collision::*;
pub use reflection::*;
pub use relaxation::*;
pub use simulation::*;
pub use space::*;
pub use streaming::*;
pub use thermal::*;

pub use crate::support::math::FRAC_1_SQRT_3;

mod constants;

pub use constants::*;

#[cfg(test)]
mod tests;
