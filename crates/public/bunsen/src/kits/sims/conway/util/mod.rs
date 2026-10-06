//! What the Conway sims share: the [`ConwaySim`] trait that drives them,
//! the [`ConwayRules`] a 3D sim steps by, and [`slices`], helpers that read
//! and write rectangles of a 2D board.

use std::ops::Range;

use burn::tensor::Device;
use serde::{
    Deserialize,
    Serialize,
};

pub mod slices;

/// Common control for Conway's Game of Life sims.
///
/// Implemented by [`ConwayLife2DState`] and [`ConwayLife3DState`]. Both
/// step in place; the kernels behind them are in
/// [`conway::ops`](super::ops).
///
/// [`ConwayLife2DState`]: super::life2d::ConwayLife2DState
/// [`ConwayLife3DState`]: super::life3d::ConwayLife3DState
pub trait ConwaySim {
    /// Returns the device the board is on.
    fn device(&self) -> Device;

    /// Adds noise to the board: each interior cell is flipped, live to dead
    /// or dead to live, with probability `density`. The halo is then
    /// rewritten from the interior, so the next step wraps.
    fn fuzz(
        &mut self,
        density: f64,
    );

    /// Advances one generation, in place.
    ///
    /// Computes the next interior from each cell's neighbourhood, then
    /// rewrites the halo from the opposite edges, so the board wraps as a
    /// torus. It reads the halo as it stands, which the sims' own edits keep
    /// fresh; the [`conway`](super) docs say how to keep it fresh after
    /// writing `state` directly.
    fn step(&mut self);
}

/// Life Rulesets.
///
/// A cell's neighbours are the live cells around it, not counting itself.
/// A dead cell with a neighbour count in `spawn` is born, and a live cell
/// with a count in `keep` survives; every other cell is dead next
/// generation. The default, `spawn: 3..4` and `keep: 2..4`, is the
/// standard B3/S23. Only [`ConwayLife3DConfig`] takes rules; the 2D sim's
/// rule is fixed at B3/S23.
///
/// [`ConwayLife3DConfig`]: super::life3d::ConwayLife3DConfig
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConwayRules {
    /// The range in which new cells will spawn.
    pub spawn: Range<usize>,

    /// The range in which live cells will survive.
    pub keep: Range<usize>,
}

impl Default for ConwayRules {
    fn default() -> Self {
        Self {
            spawn: 3..4,
            keep: 2..4,
        }
    }
}
