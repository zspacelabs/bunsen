use burn::{
    Tensor,
    config::Config,
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
/// Specifies the `[H, W]` board shape. Call `.init(device)` to build a
/// zeroed [`ConwayLife2DState`] module, which can then be seeded and stepped.
#[derive(Config, Debug)]
pub struct ConwayLife2DConfig {
    /// The shape of the board.
    pub shape: GridShape2D,
}

impl ConwayLife2DConfig {
    /// Initializes a [`ConwayLife2DState`] module.
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

/// State module for Conway's Game of Life.
///
/// Holds the toroidal `[H, W]` boolean board. Construct it from a
/// [`ConwayLife2DConfig`] via `.init(device)`, optionally seed it with
/// [`ConwayLife2DState::fuzz`], then call [`ConwayLife2DState::step`] to
/// advance the simulation one wrapped generation at a time.
///
/// Built by [`ConwayLife2DConfig`].
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
        self.state.inplace(|s| fuzz_state_2d(s, density))
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

    /// Writes a slice to the current board state.
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

        self.state.inplace(|s| s.slice_assign(slices, data));
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

        conway.write_slice(s![-2.., -2..], vec![vec![false, true], vec![true, true]]);

        assert_eq!(
            conway.read_slice(s![1..3, 1..3]),
            vec![vec![true, true], vec![true, false]]
        );

        next_interior_2d(conway.state.clone()).to_data().assert_eq(
            &TensorData::from([
                [true, true, false],
                [true, true, false],
                [false, false, true],
            ]),
            false,
        )
    }
}
