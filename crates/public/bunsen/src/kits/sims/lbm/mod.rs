//! # Lattice-Boltzmann Fluid Simulations
//!
//! The lattice-Boltzmann method models a fluid as populations of particles
//! on a grid, each moving in one of a few fixed directions. A step streams
//! each population to the neighbouring cell it is moving toward, then
//! relaxes each cell's populations toward the equilibrium of the cell's
//! density and velocity (collision). Every step is local and the same at
//! every cell, which suits whole-grid tensor operations.
//!
//! - [`d2q9`]: two dimensions, nine directions, with solid walls and a mass
//!   correction.

pub mod d2q9;
