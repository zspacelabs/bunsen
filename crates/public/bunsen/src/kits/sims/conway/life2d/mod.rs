//! # 2D Conway's Game of Life
//!
//! [`ConwayLife2DConfig`] builds a [`ConwayLife2DState`], a `[H, W]` board
//! whose interior is a torus inside a one-cell halo, and
//! [`ConwaySim::step`](super::util::ConwaySim::step) advances it one
//! generation by the fixed B3/S23 rule. The [`conway`](super) docs cover
//! the halo and the kernels.

mod state2d;

pub use state2d::*;
