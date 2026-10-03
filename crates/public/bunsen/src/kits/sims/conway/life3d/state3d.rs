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
    ops::{
        next_state_wrapped_3d,
        project_wrapped_toroidal_boarders,
    },
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
/// one-cell halo that each step and `fuzz` rewrites from the opposite faces,
/// together with its [`ConwayRules`]. Construct it from a
/// [`ConwayLife3DConfig`] via `.init(device)`, optionally seed it with
/// [`ConwayLife3DState::fuzz`], then call [`ConwayLife3DState::step`] to
/// advance the simulation one wrapped generation at a time.
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

        // The noise lands on the halo too; rewrite it from the interior, so
        // the next step wraps.
        self.state
            .inplace(|s| project_wrapped_toroidal_boarders(s.bool_or(noise)));
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
    use burn::prelude::s;
    use serial_test::serial;

    use super::*;
    use crate::{
        prelude::TensorElemOpExt,
        support::testing::{
            DeviceMemoryGuard,
            PerformanceBackend,
            default_device,
        },
    };

    /// One generation of a `[h, w, z]` torus, flat in row-major order, by
    /// `rules`, computed on the host: the reference a wrapped step of the
    /// board's interior must match.
    fn torus_step_3d(
        torus: &[bool],
        [h, w, z]: [usize; 3],
        rules: &ConwayRules,
    ) -> Vec<bool> {
        let at = |i: usize, j: usize, k: usize| torus[(i * w + j) * z + k];
        let mut next = Vec::with_capacity(torus.len());
        for i in 0..h {
            for j in 0..w {
                for k in 0..z {
                    let mut n = 0;
                    for di in [h - 1, 0, 1] {
                        for dj in [w - 1, 0, 1] {
                            for dk in [z - 1, 0, 1] {
                                if (di, dj, dk) != (0, 0, 0)
                                    && at((i + di) % h, (j + dj) % w, (k + dk) % z)
                                {
                                    n += 1;
                                }
                            }
                        }
                    }
                    next.push(if at(i, j, k) {
                        rules.keep.contains(&n)
                    } else {
                        rules.spawn.contains(&n)
                    });
                }
            }
        }
        next
    }

    /// After `fuzz`, the first step wraps the torus: it matches the host
    /// reference, faces included.
    #[test]
    #[serial]
    fn test_step_after_fuzz_wraps() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        // A 6x7x8 board: a 4x5x6 torus. With 26 neighbours, these rules and
        // this density leave most cells' fate to a few neighbours, so a
        // wrong neighbour shows.
        let rules = ConwayRules {
            spawn: 5..7,
            keep: 4..7,
        };
        let mut life: ConwayLife3DState<B> = ConwayLife3DConfig::new([6, 7, 8])
            .with_rules(rules.clone())
            .init(&device);
        life.fuzz(0.2);
        let interior = |life: &ConwayLife3DState<B>| {
            life.state
                .clone()
                .slice(s![1..5, 1..6, 1..7])
                .to_data_as::<bool>()
                .to_vec::<bool>()
                .unwrap()
        };
        let seed = interior(&life);

        life.step();

        assert_eq!(interior(&life), torus_step_3d(&seed, [4, 5, 6], &rules));
    }

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
