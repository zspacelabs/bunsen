use burn::{
    Tensor,
    config::Config,
    prelude::{
        Backend,
        Bool,
        Int,
    },
    tensor::Distribution,
};

use crate::kits::sims::conway::{
    ops::next_state_wrapped_3d,
    util::{
        ConwayRules,
        ConwaySim,
    },
};

/// Config for [`ConwayLife3DState`]
///
/// Specifies the `[H, W, Z]` board shape, halo included (the torus is the
/// `[H-2, W-2, Z-2]` interior), and the [`ConwayRules`] ruleset. Call
/// `.init(device)` to build an all-dead [`ConwayLife3DState`], which can
/// then be seeded and stepped.
#[derive(Config, Debug)]
pub struct ConwayLife3DConfig {
    /// The shape of the board.
    pub shape: [usize; 3],

    /// The ruleset to use.
    #[config(default_value = "Default::default()")]
    pub rules: ConwayRules,
}

impl ConwayLife3DConfig {
    /// Initializes an all-dead [`ConwayLife3DState`] on `device`.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ConwayLife3DState<B> {
        ConwayLife3DState {
            state: Tensor::<B, 3, Int>::zeros(&self.shape, device).bool(),
            rules: self.rules.clone(),
        }
    }
}

/// The board of a 3D Game of Life.
///
/// Holds the `[H, W, Z]` boolean board, a `[H-2, W-2, Z-2]` torus inside a
/// one-cell halo that each step rewrites from the opposite faces, together
/// with its [`ConwayRules`]. Construct it from a [`ConwayLife3DConfig`] via
/// `.init(device)`, optionally seed it with [`ConwayLife3DState::fuzz`],
/// then call [`ConwayLife3DState::step`] to advance the simulation one
/// wrapped generation at a time.
///
/// A plain struct, not a burn `Module`: move it to another device by
/// moving its `state` tensor.
///
/// Built by [`ConwayLife3DConfig`].
pub struct ConwayLife3DState<B: Backend> {
    /// The current state of the board.
    pub state: Tensor<B, 3, Bool>,

    /// The ruleset to use.
    pub rules: ConwayRules,
}

impl<B: Backend> ConwayLife3DState<B> {
    fn shape(&self) -> [usize; 3] {
        self.state.shape().dims()
    }
}

impl<B: Backend> ConwaySim<B> for ConwayLife3DState<B> {
    fn device(&self) -> B::Device {
        self.state.device()
    }

    fn fuzz(
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
    fn step(&mut self) {
        self.state
            .inplace(|s| next_state_wrapped_3d(s, &self.rules));

        // B::sync(&self.device()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        PerformanceBackend,
        default_device,
    };

    #[test]
    #[serial]
    fn test_smoke() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let steps = 100;
        let grid_size = 20;

        let config = ConwayLife3DConfig {
            shape: [grid_size, grid_size, grid_size],
            rules: ConwayRules {
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
