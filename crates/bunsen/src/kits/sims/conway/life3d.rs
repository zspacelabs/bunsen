//! # 3D Conway's Game of Life

use std::ops::Range;

use burn::{
    Tensor,
    config::Config,
    prelude::{
        Backend,
        Bool,
        Int,
        s,
    },
    tensor::Distribution,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    kits::sims::conway::range_util::{
        range_into,
        shift_range,
    },
    prelude::{
        TensorBoolOpExt,
        TensorIntOpExt,
    },
};

/// Fuzzes the state.
///
/// Flips bits with probability `density`.
///
/// # Arguments
///
/// - `state`: the `[H, W, Z]` input state.
/// - `density`: the probability of flipping a given bit.
///
/// # Returns
/// - the fuzzed `[H, W, Z]` state.
pub fn fuzz_state_3d<B: Backend>(
    state: Tensor<B, 3, Bool>,
    density: f64,
) -> Tensor<B, 3, Bool> {
    if density == 0.0 {
        return state;
    }

    let noise: Tensor<B, 3, Bool> = Tensor::<B, 3>::random(
        state.shape(),
        Distribution::Bernoulli(density),
        &state.device(),
    )
    .equal_elem(1.0);

    state.bool_xor(noise)
}

/// Wraps the board state.
///
/// This simulates a toroidal space by copying the penultimate rows and columns
/// to the edges of the opposite sides.
pub fn wrap_state_3d<B: Backend>(state: Tensor<B, 3, Bool>) -> Tensor<B, 3, Bool> {
    let top = state.clone().slice(s![1, .., ..]);
    let bottom = state.clone().slice(s![-2, .., ..]);
    let left = state.clone().slice(s![.., 1, ..]);
    let right = state.clone().slice(s![.., -2, ..]);
    let front = state.clone().slice(s![.., .., 1]);
    let back = state.clone().slice(s![.., .., -2]);

    state
        .slice_assign(s![0, .., ..], bottom)
        .slice_assign(s![-1, .., ..], top)
        .slice_assign(s![.., 0, ..], right)
        .slice_assign(s![.., -1, ..], left)
        .slice_assign(s![.., .., 0], back)
        .slice_assign(s![.., .., -1], front)
}

/// Returns the next board.
///
/// # Arguments
///
/// - `state`: a `[H, W, Z]` game state.
/// - `rules`: a ruleset.
///
/// # Returns
/// - the `[H, W, Z]` evolved interior state, with wrapped edges.
pub fn next_state_wrapped_3d<B: Backend>(
    state: Tensor<B, 3, Bool>,
    rules: &LifeRules,
) -> Tensor<B, 3, Bool> {
    let update = next_interior_3d(state.clone(), rules);

    // There's a *significant* performance speedup (+60%) from re-using the state,
    // rather than building a new state with pad-expansion.
    // This appears to mainly be a result of backend optimizations.
    let state = state.slice_assign(s![1..-1, 1..-1, 1..-1], update);

    wrap_state_3d(state)
}

/// Returns the interior board next-state.
///
/// # Arguments
///
/// - `state`: a `[H, W, Z]` game state.
/// - `rules`: the ruleset to use.
///
/// # Returns
/// - the `[H-2, W-2, Z-2]` evolved interior state.
pub fn next_interior_3d<B: Backend>(
    state: Tensor<B, 3, Bool>,
    rules: &LifeRules,
) -> Tensor<B, 3, Bool> {
    #[cfg(debug_assertions)]
    let [h, w, z] = crate::contracts::unpack_shape_contract!(["h", "w", "z"], &state.dims());

    // [H-2, W-2, Z-2]
    let is_live = state.clone().slice(s![1..-1, 1..-1, 1..-1]);

    // [H-2, W-2, Z-2]
    let win_counts = state
        .clone()
        .unfold::<4, _>(0, 3, 1)
        .unfold::<5, _>(1, 3, 1)
        .unfold::<6, _>(2, 3, 1)
        .count_dims(&[3, 4, 5])
        .squeeze_dims::<3>(&[3, 4, 5]);

    let spawn_range = range_into(&rules.spawn);
    let keep_range = shift_range(range_into(&rules.keep), 1);

    let spawn_points = win_counts.clone().bounded_elem(spawn_range);
    let keep_points = win_counts.bounded_elem(keep_range);

    let update = spawn_points.mask_where(is_live, keep_points);

    #[cfg(debug_assertions)]
    crate::contracts::assert_shape_contract_periodically!(
        ["h" - 2, "w" - 2, "z" - 2],
        &update.dims(),
        &[("h", h), ("w", w), ("z", z)],
    );

    update
}

/// Life Rulesets.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct LifeRules {
    /// The range in which new cells will spawn.
    pub spawn: Range<usize>,

    /// The range in which live cells will survive.
    pub keep: Range<usize>,
}

impl Default for LifeRules {
    fn default() -> Self {
        Self {
            spawn: 3..4,
            keep: 2..4,
        }
    }
}

/// Config for [`ConwayLife3DState`]
///
/// Specifies the `[H, W, Z]` board shape and the [`LifeRules`] ruleset. Call
/// `.init(device)` to build a zeroed [`ConwayLife3DState`] module, which can
/// then be seeded and stepped.
#[derive(Config, Debug)]
pub struct ConwayLife3DConfig {
    /// The shape of the board.
    pub shape: [usize; 3],

    /// The ruleset to use.
    pub rules: LifeRules,
}

impl ConwayLife3DConfig {
    /// Initializes a [`ConwayLife3DState`] module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> ConwayLife3DState<B> {
        ConwayLife3DState {
            state: Tensor::<B, 3, Int>::zeros(self.shape, device).bool(),
            rules: self.rules,
        }
    }
}

/// State module for Conway's Game of Life.
///
/// Holds the toroidal `[H, W, Z]` boolean board together with its
/// [`LifeRules`]. Construct it from a [`ConwayLife3DConfig`] via
/// `.init(device)`, optionally seed it with [`ConwayLife3DState::fuzz`], then
/// call [`ConwayLife3DState::step`] to advance the simulation one wrapped
/// generation at a time.
///
/// Built by [`ConwayLife3DConfig`].
pub struct ConwayLife3DState<B: Backend> {
    /// The current state of the board.
    pub state: Tensor<B, 3, Bool>,

    /// The ruleset to use.
    pub rules: LifeRules,
}

impl<B: Backend> ConwayLife3DState<B> {
    /// Returns the device the module is on.
    pub fn device(&self) -> B::Device {
        self.state.device()
    }

    /// Returns the board shape.
    pub fn shape(&self) -> [usize; 3] {
        self.state.shape().dims()
    }

    /// Adds uniform positive noise to the board.
    pub fn fuzz(
        &mut self,
        density: f64,
    ) {
        if density == 0.0 {
            return;
        }

        let noise: Tensor<B, 3, Bool> = Tensor::<B, 3>::random(
            self.shape(),
            Distribution::Bernoulli(density),
            &self.device(),
        )
        .equal_elem(1.0);
        self.state.inplace(|s| s.bool_or(noise));
    }

    /// Advances the game state.
    pub fn step(&mut self) {
        self.state
            .inplace(|s| next_state_wrapped_3d(s, &self.rules));
        B::sync(&self.device()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::support::testing::PerformanceBackend;

    #[test]
    #[serial]
    fn test_smoke() {
        type B = PerformanceBackend;
        let device = Default::default();

        let steps = 100;
        let grid_size = 20;

        let config = ConwayLife3DConfig {
            shape: [grid_size, grid_size, grid_size],
            rules: LifeRules {
                spawn: 3..5,
                keep: 2..4,
            },
        };
        let mut game: ConwayLife3DState<B> = config.init(&device);
        game.fuzz(0.05);

        for _ in 0..steps {
            game.step();
        }
    }
}
