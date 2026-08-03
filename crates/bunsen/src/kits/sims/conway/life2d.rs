//! # 2D Conway's Game of Life

use burn::{
    Tensor,
    config::Config,
    prelude::{
        Backend,
        Bool,
        Int,
        SliceArg,
        s,
    },
    tensor::{
        Distribution,
        Slice,
    },
};

use crate::prelude::{
    TensorBoolOpExt,
    TensorElemOpExt,
};

/// Fuzzes the state.
///
/// Flips bits with probability `density`.
///
/// # Arguments
///
/// - `state`: the `[H, W]` input state.
/// - `density`: the probability of flipping a given bit.
///
/// # Returns
/// - the fuzzed `[H, W]` state.
pub fn fuzz_state_2d<B: Backend>(
    state: Tensor<B, 2, Bool>,
    density: f64,
) -> Tensor<B, 2, Bool> {
    if density == 0.0 {
        return state;
    }

    let noise: Tensor<B, 2, Bool> = Tensor::<B, 2>::random(
        state.shape(),
        Distribution::Bernoulli(density),
        &state.device(),
    )
    .bool();

    state.bool_xor(noise)
}

/// Wraps the board state.
///
/// This simulates a toroidal space by copying the penultimate rows and columns
/// to the edges of the opposite sides.
pub fn wrap_state_2d<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    let bottom = state.clone().slice(s![-2, ..]);
    let state = state.slice_assign(s![0, ..], bottom);

    let top = state.clone().slice(s![1, ..]);
    let state = state.slice_assign(s![-1, ..], top);

    let right = state.clone().slice(s![.., -2]);
    let state = state.slice_assign(s![.., 0], right);

    let left = state.clone().slice(s![.., 1]);
    state.slice_assign(s![.., -1], left)
}

fn slice_size(slice: &Slice) -> usize {
    (slice.end.unwrap() - slice.start) as usize
}

fn slices_shape(slices: &[Slice; 2]) -> [usize; 2] {
    [slice_size(&slices[0]), slice_size(&slices[1])]
}

fn read_2d_slice<B: Backend, R>(
    state: Tensor<B, 2, Bool>,
    ranges: R,
) -> Vec<Vec<bool>>
where
    R: SliceArg,
{
    let slices: [Slice; 2] = ranges.into_slices(&state.shape()).try_into().unwrap();
    let [_, w] = slices_shape(&slices);

    state
        .slice(slices)
        .to_data_convert::<bool>()
        .to_vec()
        .unwrap()
        .chunks(w)
        .map(<[_]>::to_vec)
        .collect()
}

/// Returns the next board.
///
/// # Arguments
///
/// - `state`: a `[H, W]` game state.
///
/// # Returns
/// - the `[H, W]` evolved interior state, with wrapped edges.
pub fn next_state_wrapped_2d<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    let update = next_interior_2d(state.clone());

    // This is faster than re-padding; due to in-place update optimizations.
    wrap_state_2d(state.slice_assign(s![1..-1, 1..-1], update))
}

/// Returns the interior board next-state.
///
/// # Arguments
/// - `state`: a `[H, W]` game state.
///
/// # Returns
/// - the `[H-2, W-2]` evolved interior state.
pub fn next_interior_2d<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    #[cfg(any(test, debug_assertions))]
    let [h, w] = crate::contracts::unpack_shape_contract!(["h", "w"], &state.dims());

    // [H-2, W-2]
    let is_live = state.clone().slice(s![1..-1, 1..-1]);

    // [H-2, W-2]
    let window_count = state
        .unfold::<3, _>(0, 3, 1)
        .unfold::<4, _>(1, 3, 1)
        .count_dims(&[2, 3])
        .squeeze_dims::<2>(&[2, 3]);

    let n_is_3 = window_count.clone().equal_elem(3);
    let n_is_4 = window_count.equal_elem(4);

    let inner = n_is_3.bool_or(n_is_4.bool_and(is_live));

    #[cfg(any(test, debug_assertions))]
    crate::contracts::assert_shape_contract_periodically!(
        ["h" - 2, "w" - 2],
        &inner.dims(),
        &[("h", h), ("w", w)],
    );

    // [H-2, W-2]
    inner
}

/// Config for [`ConwayLife2DState`]
///
/// Specifies the `[H, W]` board shape. Call `.init(device)` to build a
/// zeroed [`ConwayLife2DState`] module, which can then be seeded and stepped.
#[derive(Config, Debug)]
pub struct ConwayLife2DConfig {
    /// The shape of the board.
    pub shape: [usize; 2],
}

impl ConwayLife2DConfig {
    /// Initializes a [`ConwayLife2DState`] module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> ConwayLife2DState<B> {
        ConwayLife2DState {
            state: Tensor::<B, 2, Int>::zeros(self.shape, device).bool(),
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
    /// The current state of the board.
    pub state: Tensor<B, 2, Bool>,
}

impl<B: Backend> ConwayLife2DState<B> {
    /// Returns the device the module is on.
    pub fn device(&self) -> B::Device {
        self.state.device()
    }

    /// Returns the board shape.
    pub fn shape(&self) -> [usize; 2] {
        self.state.shape().dims()
    }

    /// Adds uniform positive noise to the board.
    pub fn fuzz(
        &mut self,
        density: f64,
    ) {
        self.state.inplace(|s| fuzz_state_2d(s, density))
    }

    /// Advances one step.
    ///
    /// Wraps edges.
    pub fn step(&mut self) {
        self.state.inplace(next_state_wrapped_2d);
        B::sync(&self.device()).unwrap();
    }

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
    use crate::support::testing::PerformanceBackend;

    #[test]
    #[serial]
    fn test_smoke() {
        type B = PerformanceBackend;
        let device = Default::default();

        let steps = 100;
        let grid_size = 20;

        let config = ConwayLife2DConfig {
            shape: [grid_size, grid_size],
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
        let device = Default::default();
        let config = ConwayLife2DConfig { shape: [5, 5] };
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
