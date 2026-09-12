//! Fuzzing utilities for Conway's Game of Life.

use std::ops::Range;

use burn::prelude::Backend;
use serde::{
    Deserialize,
    Serialize,
};

pub mod slices;

/// Common control for Conway's Game of Life sims.
pub trait ConwaySim<B: Backend> {
    /// Returns the device the module is on.
    fn device(&self) -> B::Device;

    /// Adds uniform positive noise to the board.
    fn fuzz(
        &mut self,
        density: f64,
    );

    /// Advances one step.
    ///
    /// Wraps edges.
    fn step(&mut self);
}

/// Life Rulesets.
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
