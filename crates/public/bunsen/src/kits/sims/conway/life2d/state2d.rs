use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Backend,
        Bool,
        Int,
        SliceArg,
    },
    tensor::Slice,
};

use crate::{
    kits::sims::conway::{
        ops::{
            fuzz_state_2d,
            next_state_wrapped_2d,
            project_wrapped_toroidal_boarders,
        },
        util::{
            ConwaySim,
            slices::{
                read_2d_slice,
                slices_shape,
            },
        },
    },
    support::geometry::GridShape2D,
};

/// Config for [`ConwayLife2DState`]
///
/// Specifies the `[H, W]` board shape, halo included: the torus is the
/// `[H-2, W-2]` interior. Call `.init(device)` to build an all-dead
/// [`ConwayLife2DState`], which can then be seeded and stepped.
#[derive(Config, Debug)]
pub struct ConwayLife2DConfig {
    /// The shape of the board.
    pub shape: GridShape2D,
}

impl ConwayLife2DConfig {
    /// Initializes an all-dead [`ConwayLife2DState`] on `device`.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> ConwayLife2DState<B> {
        ConwayLife2DState {
            shape: self.shape,
            state: Tensor::<B, 2, Int>::zeros(self.shape.as_height_width(), device).bool(),
        }
    }
}

/// The board of a 2D Game of Life.
///
/// Holds the `[H, W]` boolean board: a `[H-2, W-2]` torus inside a
/// one-cell halo, which each step, `fuzz` and `write_slice` rewrites from
/// the opposite edges. Construct it from a [`ConwayLife2DConfig`] via
/// `.init(device)`, optionally seed it with [`ConwayLife2DState::fuzz`] or
/// [`write_slice`](Self::write_slice), then call
/// [`ConwayLife2DState::step`] to advance the simulation one wrapped
/// generation at a time, by the fixed B3/S23 rule.
///
/// A burn `Module` over the bare `state` tensor, so `to_device` and `fork`
/// move the board. The board is not a parameter: it is not written to
/// records, `ModuleMapper` passes skip it, and it does not appear in
/// reflection.
///
/// Built by [`ConwayLife2DConfig`].
#[derive(Module, Debug)]
pub struct ConwayLife2DState<B: Backend> {
    /// The shape of the board.
    pub shape: GridShape2D,

    /// The current state of the board.
    pub state: Tensor<B, 2, Bool>,
}

impl<B: Backend> ConwaySim<B> for ConwayLife2DState<B> {
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

        // The noise lands on the halo too; rewrite it from the interior, so
        // the next step wraps.
        self.state
            .inplace(|s| project_wrapped_toroidal_boarders(fuzz_state_2d(s, density)))
    }

    fn step(&mut self) {
        self.state.inplace(next_state_wrapped_2d);

        // B::sync(&self.device()).unwrap();
    }
}

impl<B: Backend> ConwayLife2DState<B> {
    /// Reads a slice of the current board state.
    pub fn read_slice<R>(
        &self,
        ranges: R,
    ) -> Vec<Vec<bool>>
    where
        R: SliceArg,
    {
        read_2d_slice(self.state.clone(), ranges)
    }

    /// Writes a slice to the current board state, then rewrites the halo
    /// from the interior, so the next step wraps.
    ///
    /// Write the interior: a cell written in the halo is replaced by the
    /// interior cell it mirrors.
    pub fn write_slice<R>(
        &mut self,
        ranges: R,
        data: Vec<Vec<bool>>,
    ) where
        R: SliceArg,
    {
        let slices: [Slice; 2] = ranges.into_slices(&self.state.shape()).try_into().unwrap();
        let [h, w] = slices_shape(&slices);

        assert_eq!(data.len(), h);
        for row in data.iter() {
            assert_eq!(row.len(), w);
        }

        let mut block = Vec::with_capacity(h * w);
        for row in data.iter() {
            for &cell in row.iter() {
                block.push(cell as u32);
            }
        }

        let data = Tensor::<B, 1, Int>::from_ints(block.as_slice(), &self.device())
            .bool()
            .reshape([h, w]);

        self.state
            .inplace(|s| project_wrapped_toroidal_boarders(s.slice_assign(slices, data)));
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::s,
        tensor::TensorData,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        kits::sims::conway::{
            ops::next_interior_2d,
            util::ConwaySim,
        },
        support::testing::{
            DeviceMemoryGuard,
            PerformanceBackend,
            default_device,
        },
    };

    /// One B3/S23 generation of a torus, computed on the host: the reference
    /// a wrapped step of the board's interior must match.
    fn torus_step_2d(torus: &[Vec<bool>]) -> Vec<Vec<bool>> {
        let h = torus.len();
        let w = torus[0].len();
        (0..h)
            .map(|i| {
                (0..w)
                    .map(|j| {
                        let mut n = 0;
                        for di in [h - 1, 0, 1] {
                            for dj in [w - 1, 0, 1] {
                                if (di, dj) != (0, 0) && torus[(i + di) % h][(j + dj) % w] {
                                    n += 1;
                                }
                            }
                        }
                        n == 3 || (n == 2 && torus[i][j])
                    })
                    .collect()
            })
            .collect()
    }

    /// A seed written on the torus's top row wraps on the first step: a
    /// blinker there turns to a column whose top cell is the torus's bottom
    /// row.
    #[test]
    #[serial]
    fn test_step_after_write_slice_wraps() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        // A 7x7 board: a 5x5 torus in rows and columns 1..6.
        let mut life: ConwayLife2DState<B> =
            ConwayLife2DConfig::new(GridShape2D::square(7)).init(&device);
        life.write_slice(s![1, 1..4], vec![vec![true, true, true]]);
        let seed = life.read_slice(s![1..6, 1..6]);

        life.step();

        assert_eq!(life.read_slice(s![1..6, 1..6]), torus_step_2d(&seed));
        assert_eq!(
            life.read_slice(s![1..6, 2]),
            vec![vec![true], vec![true], vec![false], vec![false], vec![true]]
        );
    }

    /// After `fuzz`, the first step wraps the torus: it matches the host
    /// reference, edges included.
    #[test]
    #[serial]
    fn test_step_after_fuzz_wraps() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        // A 9x12 board: a 7x10 torus in rows 1..8, columns 1..11.
        let mut life: ConwayLife2DState<B> = ConwayLife2DConfig::new(GridShape2D {
            width: 12,
            height: 9,
        })
        .init(&device);
        life.fuzz(0.5);
        let seed = life.read_slice(s![1..8, 1..11]);

        life.step();

        assert_eq!(life.read_slice(s![1..8, 1..11]), torus_step_2d(&seed));
    }

    /// `fuzz` flips each cell it hits. At density 1 it hits every cell, so
    /// it inverts the board, halo included, and a second pass restores it.
    #[test]
    #[serial]
    fn test_fuzz_flips_each_hit_cell() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        // A 7x7 board: a 5x5 torus with live and dead cells.
        let mut life: ConwayLife2DState<B> =
            ConwayLife2DConfig::new(GridShape2D::square(7)).init(&device);
        life.write_slice(
            s![1..3, 1..4],
            vec![vec![true, false, true], vec![false, true, true]],
        );
        let seed = life.read_slice(s![.., ..]);
        let inverted: Vec<Vec<bool>> = seed
            .iter()
            .map(|row| row.iter().map(|cell| !cell).collect())
            .collect();

        life.fuzz(1.0);
        assert_eq!(life.read_slice(s![.., ..]), inverted);

        life.fuzz(1.0);
        assert_eq!(life.read_slice(s![.., ..]), seed);
    }

    /// The module traversal reaches the board: it is the one tensor, so
    /// `devices` lists its device, once.
    #[test]
    #[serial]
    fn test_module_reaches_the_board() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let life: ConwayLife2DState<B> =
            ConwayLife2DConfig::new(GridShape2D::square(5)).init(&device);
        assert_eq!(life.devices(), vec![device]);
    }

    #[test]
    #[serial]
    fn test_smoke() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let steps = 100;
        let grid_size = 20;

        let config = ConwayLife2DConfig {
            shape: GridShape2D::square(grid_size),
        };
        let mut game: ConwayLife2DState<B> = config.init(&device);
        game.fuzz(0.05);

        for _ in 0..steps {
            game.step();
        }
    }

    #[test]
    #[serial]
    fn test_logic() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);
        let config = ConwayLife2DConfig {
            shape: GridShape2D::square(5),
        };
        let mut conway: ConwayLife2DState<B> = config.init(&device);

        assert_eq!(
            conway.read_slice(s![1..3, 1..3]),
            vec![vec![false, false], vec![false, false]]
        );

        conway.write_slice(s![1..3, 1..3], vec![vec![true, true], vec![true, false]]);

        assert_eq!(
            conway.read_slice(s![1..3, 1..3]),
            vec![vec![true, true], vec![true, false]]
        );

        // Of these four cells, only `(3, 3)` is interior; the halo cells are
        // rewritten as the interior cells they mirror.
        conway.write_slice(s![-2.., -2..], vec![vec![false, true], vec![true, true]]);

        assert_eq!(
            conway.read_slice(s![1..3, 1..3]),
            vec![vec![true, true], vec![true, false]]
        );
        assert_eq!(
            conway.read_slice(s![-2.., -2..]),
            vec![vec![false, false], vec![false, true]]
        );

        // On a 3x3 torus, each cell's window is the whole torus: three live
        // cells, so every cell is live next.
        next_interior_2d(conway.state.clone())
            .to_data()
            .assert_eq(&TensorData::from([[true; 3]; 3]), false)
    }
}
