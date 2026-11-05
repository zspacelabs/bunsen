//! # Simulations
//!
//! Simulations written as tensor programs. A simulation step is the same
//! kind of program as a model's forward pass: whole-grid tensor operations,
//! windows, masks and reductions, with no per-cell loop. Written with
//! `burn`'s tensors, it runs on every backend bunsen supports, CPU or GPU,
//! and its state is a tensor that the rest of an ML program can read,
//! train on, or steer.
//!
//! They also earn their place by what they exercise. A simulation runs a
//! few kernels many thousands of times, step after step, so it is a steady
//! workload for the backends and the ops beneath it (`examples/conway` has
//! a throughput benchmark), and a wrong kernel shows up as a wrong
//! picture. `examples/conway` and `examples/lbm2d_vis` draw them live.
//!
//! - [`conway`]: Conway's Game of Life on toroidal 2D and 3D boards.
//! - [`lbm`]: a lattice-Boltzmann fluid, D2Q9.
//!
//! Each follows the same lifecycle: a config builds a state on a device,
//! the caller may seed or edit the state's tensors, and a step method
//! advances it in place, one step per call. The kernels the steps are made
//! of are public too, as functions from tensors to tensors.

pub mod conway;
pub mod lbm;
