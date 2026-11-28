//! What the Conway sims share: the [`ConwaySim`] trait that drives them,
//! the [`ConwayRules`] a 3D sim steps by, and [`slices`], helpers that read
//! and write rectangles of a 2D board.

pub mod slices;

mod sim;

pub use sim::*;
