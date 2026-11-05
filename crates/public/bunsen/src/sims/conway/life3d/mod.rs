//! # 3D Conway's Game of Life
//!
//! [`ConwayLife3DConfig`] builds a [`ConwayLife3DState`], a `[H, W, Z]`
//! board whose interior is a torus inside a one-cell halo, and
//! [`ConwaySim::step`](super::util::ConwaySim::step) advances it one
//! generation by the state's [`ConwayRules`](super::util::ConwayRules). The
//! [`conway`](super) docs cover the halo and the kernels.

mod state3d;

pub use state3d::*;
