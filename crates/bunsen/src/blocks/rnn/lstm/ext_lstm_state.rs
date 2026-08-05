use alloc::vec::Vec;

use burn::{
    Tensor,
    nn::LstmState,
    prelude::{
        Backend,
        Shape,
        SliceArg,
    },
};

/// State bundle for LSTM implementations.
///
/// This is a stand-in for [`LstmState`].
/// Waiting on: <https://github.com/tracel-ai/burn/pull/5167>
#[derive(Debug, Clone)]
pub struct ExtLstmState<B: Backend, const D: usize> {
    /// The cell state.
    pub cell: Tensor<B, D>,
    /// The hidden state.
    pub hidden: Tensor<B, D>,
}

impl<B: Backend, const D: usize> From<LstmState<B, D>> for ExtLstmState<B, D> {
    fn from(state: LstmState<B, D>) -> Self {
        Self::new(state.cell, state.hidden)
    }
}

impl<B: Backend, const D: usize> From<ExtLstmState<B, D>> for LstmState<B, D> {
    fn from(state: ExtLstmState<B, D>) -> Self {
        Self::new(state.cell, state.hidden)
    }
}

impl<B: Backend, const D: usize> ExtLstmState<B, D> {
    /// Construct a new [`ExtLstmState`].
    ///
    /// This is the inverse to [`Self::unpack`].
    ///
    /// # Arguments
    /// * `cell` - the cell state.
    /// * `hidden` - the hidden state.
    ///
    /// # Debug Assertion
    /// `cell.shape() == hidden.shape()`
    pub fn new(
        cell: Tensor<B, D>,
        hidden: Tensor<B, D>,
    ) -> Self {
        #[cfg(any(test, debug_assertions))]
        assert_eq!(cell.shape(), hidden.shape());

        Self { cell, hidden }
    }

    /// Allocate an initial zero state.
    ///
    /// # Arguments
    /// * `shape` - the shape of the cell and hidden states.
    /// * `device` - the device to allocate the state on.
    pub fn initial<S>(
        shape: S,
        device: &B::Device,
    ) -> Self
    where
        S: Into<Shape>,
    {
        let cell = Tensor::zeros(shape, device);
        let hidden = cell.clone();
        Self { cell, hidden }
    }

    /// Get the shape of the state.
    pub fn shape(&self) -> Shape {
        self.cell.shape()
    }

    /// Get the device of the state.
    pub fn device(&self) -> B::Device {
        self.cell.device()
    }

    /// Unpack the state to (cell, hidden).
    ///
    /// This is the inverse to [`Self::new`].
    pub fn unpack(self) -> (Tensor<B, D>, Tensor<B, D>) {
        (self.cell, self.hidden)
    }

    /// Maps the internal cells to a new state.
    ///
    /// # Arguments
    /// * `f` - a function to map each member.
    ///
    /// # Returns
    /// A newly mapped state.
    pub fn map_state<const D2: usize, F>(
        self,
        f: F,
    ) -> ExtLstmState<B, D2>
    where
        F: Fn(Tensor<B, D>) -> Tensor<B, D2>,
    {
        let Self { cell, hidden } = self;
        ExtLstmState {
            cell: f(cell),
            hidden: f(hidden),
        }
    }

    /// Slice the state.
    ///
    /// See: [`Tensor::slice`].
    pub fn slice<S>(
        self,
        slices: S,
    ) -> Self
    where
        S: SliceArg,
    {
        let slices = slices.into_slices(&self.shape());
        self.map_state(|t| t.slice(&slices))
    }

    /// Squeeze a dimension from the state.
    ///
    /// See: [`Tensor::squeeze_dim`].
    pub fn squeeze_dim<const D2: usize>(
        self,
        dim: usize,
    ) -> ExtLstmState<B, D2> {
        self.map_state(|t| t.squeeze_dim(dim))
    }

    /// Unsqueeze a dimension in the state.
    ///
    /// See: [`Tensor::unsqueeze_dim`].
    pub fn unsqueeze_dim<const D2: usize>(
        self,
        dim: usize,
    ) -> ExtLstmState<B, D2> {
        self.map_state(|t| t.unsqueeze_dim(dim))
    }

    /// Stack the states along the given dimension.
    ///
    /// See: [`Tensor::stack`].
    pub fn stack<const D2: usize>(
        states: Vec<ExtLstmState<B, D>>,
        dim: usize,
    ) -> ExtLstmState<B, D2> {
        let (c_it, h_it): (Vec<_>, Vec<_>) = states.into_iter().map(|s| s.unpack()).unzip();
        ExtLstmState {
            cell: Tensor::stack(c_it, dim),
            hidden: Tensor::stack(h_it, dim),
        }
    }
}

/// Extension trait for attaching an initializer to an [`Option<ExtLstmState>`].
pub trait OptionalInitialLstmState<B: Backend, const D: usize> {
    /// Unwrap the optional state, or allocate initial state.
    ///
    /// # Arguments
    /// * `shape` - the shape of the cell and hidden states.
    /// * `device` - the device to allocate the state on.
    ///
    /// # Returns
    /// Either self unwrapped, or allocate using [`ExtLstmState::initial`].
    fn unwrap_or_initial<S>(
        self,
        shape: S,
        device: &B::Device,
    ) -> ExtLstmState<B, D>
    where
        S: Into<Shape>;
}

impl<B: Backend, const D: usize> OptionalInitialLstmState<B, D> for Option<ExtLstmState<B, D>> {
    fn unwrap_or_initial<S>(
        self,
        shape: S,
        device: &B::Device,
    ) -> ExtLstmState<B, D>
    where
        S: Into<Shape>,
    {
        self.unwrap_or_else(|| ExtLstmState::initial(shape, device))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use burn::tensor::{
        Distribution,
        TensorData,
        s,
    };

    use super::*;
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;

    fn random_state<B: Backend, const D: usize, S>(
        shape: S,
        device: &B::Device,
    ) -> ExtLstmState<B, D>
    where
        S: Into<Shape> + Clone,
    {
        ExtLstmState::new(
            Tensor::random(shape.clone(), Distribution::Default, device),
            Tensor::random(shape, Distribution::Default, device),
        )
    }

    #[test]
    fn test_new() {
        let shape = [2, 3, 4];
        let device = Default::default();

        let cell = Tensor::random(shape, Distribution::Default, &device);
        let hidden = Tensor::random(shape, Distribution::Default, &device);

        let state: ExtLstmState<B, 3> = ExtLstmState::new(cell.clone(), hidden.clone());

        assert_eq!(state.shape(), Shape::from(shape));
        assert_eq!(state.device(), device);

        state.cell.to_data().assert_eq(&cell.to_data(), true);
        state.hidden.to_data().assert_eq(&hidden.to_data(), true);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_new_shape_mismatch() {
        let device = Default::default();

        let cell = Tensor::<B, 2>::zeros([2, 3], &device);
        let hidden = Tensor::<B, 2>::zeros([2, 4], &device);

        let _ = ExtLstmState::new(cell, hidden);
    }

    #[test]
    fn test_initial() {
        let shape = [2, 3];
        let device = Default::default();

        let state: ExtLstmState<B, 2> = ExtLstmState::initial(shape, &device);

        assert_eq!(state.shape(), Shape::from(shape));
        assert_eq!(state.device(), device);

        let zeros = TensorData::zeros::<f32, _>(shape);
        state.cell.to_data().assert_eq(&zeros, true);
        state.hidden.to_data().assert_eq(&zeros, true);
    }

    #[test]
    fn test_unpack() {
        let device = Default::default();
        let state: ExtLstmState<B, 3> = random_state([2, 3, 4], &device);

        let expected_cell = state.cell.to_data();
        let expected_hidden = state.hidden.to_data();

        let (cell, hidden) = state.unpack();

        cell.to_data().assert_eq(&expected_cell, true);
        hidden.to_data().assert_eq(&expected_hidden, true);
    }

    #[test]
    fn test_map_state() {
        let device = Default::default();
        let state: ExtLstmState<B, 2> = random_state([2, 3], &device);

        let expected_cell = state.cell.clone().reshape([3, 2]).to_data();
        let expected_hidden = state.hidden.clone().reshape([3, 2]).to_data();

        let mapped: ExtLstmState<B, 2> = state.map_state(|t| t.reshape([3, 2]));

        assert_eq!(mapped.shape(), Shape::from([3, 2]));
        mapped.cell.to_data().assert_eq(&expected_cell, true);
        mapped.hidden.to_data().assert_eq(&expected_hidden, true);
    }

    #[test]
    fn test_map_state_changes_rank() {
        let device = Default::default();
        let state: ExtLstmState<B, 2> = random_state([2, 3], &device);

        let mapped: ExtLstmState<B, 3> = state.map_state(|t| t.reshape([1, 2, 3]));

        assert_eq!(mapped.shape(), Shape::from([1, 2, 3]));
    }

    #[test]
    fn test_slice() {
        let device = Default::default();
        let state: ExtLstmState<B, 2> = random_state([4, 3], &device);

        let expected_cell = state.cell.clone().slice(s![1..3, ..]).to_data();
        let expected_hidden = state.hidden.clone().slice(s![1..3, ..]).to_data();

        let sliced = state.slice(s![1..3, ..]);

        assert_eq!(sliced.shape(), Shape::from([2, 3]));
        sliced.cell.to_data().assert_eq(&expected_cell, true);
        sliced.hidden.to_data().assert_eq(&expected_hidden, true);
    }

    #[test]
    fn test_squeeze_dim() {
        let device = Default::default();
        let state: ExtLstmState<B, 3> = random_state([2, 1, 3], &device);

        let expected_cell = state.cell.clone().squeeze_dim::<2>(1).to_data();
        let expected_hidden = state.hidden.clone().squeeze_dim::<2>(1).to_data();

        let squeezed: ExtLstmState<B, 2> = state.squeeze_dim(1);

        assert_eq!(squeezed.shape(), Shape::from([2, 3]));
        squeezed.cell.to_data().assert_eq(&expected_cell, true);
        squeezed.hidden.to_data().assert_eq(&expected_hidden, true);
    }

    #[test]
    fn test_unsqueeze_dim() {
        let device = Default::default();
        let state: ExtLstmState<B, 2> = random_state([2, 3], &device);

        let expected_cell = state.cell.clone().unsqueeze_dim::<3>(1).to_data();
        let expected_hidden = state.hidden.clone().unsqueeze_dim::<3>(1).to_data();

        let unsqueezed: ExtLstmState<B, 3> = state.unsqueeze_dim(1);

        assert_eq!(unsqueezed.shape(), Shape::from([2, 1, 3]));
        unsqueezed.cell.to_data().assert_eq(&expected_cell, true);
        unsqueezed
            .hidden
            .to_data()
            .assert_eq(&expected_hidden, true);
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip() {
        let device = Default::default();
        let state: ExtLstmState<B, 2> = random_state([2, 3], &device);

        let expected_cell = state.cell.to_data();

        let roundtrip: ExtLstmState<B, 2> = state.unsqueeze_dim::<3>(1).squeeze_dim(1);

        assert_eq!(roundtrip.shape(), Shape::from([2, 3]));
        roundtrip.cell.to_data().assert_eq(&expected_cell, true);
    }

    #[test]
    fn test_stack() {
        let device = Default::default();
        let a: ExtLstmState<B, 2> = random_state([2, 3], &device);
        let b: ExtLstmState<B, 2> = random_state([2, 3], &device);

        let expected_cell = Tensor::stack::<3>(vec![a.cell.clone(), b.cell.clone()], 1).to_data();
        let expected_hidden =
            Tensor::stack::<3>(vec![a.hidden.clone(), b.hidden.clone()], 1).to_data();

        let stacked: ExtLstmState<B, 3> = ExtLstmState::stack(vec![a, b], 1);

        assert_eq!(stacked.shape(), Shape::from([2, 2, 3]));
        stacked.cell.to_data().assert_eq(&expected_cell, true);
        stacked.hidden.to_data().assert_eq(&expected_hidden, true);
    }

    #[test]
    fn test_unwrap_or_initial_some() {
        let device = Default::default();
        let state: ExtLstmState<B, 2> = random_state([2, 3], &device);

        let expected_cell = state.cell.to_data();
        let expected_hidden = state.hidden.to_data();

        // The shape passed here is intentionally different; it must be ignored.
        let unwrapped = Some(state).unwrap_or_initial([9, 9], &device);

        assert_eq!(unwrapped.shape(), Shape::from([2, 3]));
        unwrapped.cell.to_data().assert_eq(&expected_cell, true);
        unwrapped.hidden.to_data().assert_eq(&expected_hidden, true);
    }

    #[test]
    fn test_unwrap_or_initial_none() {
        let shape = [2, 3];
        let device = Default::default();

        let unwrapped: ExtLstmState<B, 2> = None.unwrap_or_initial(shape, &device);

        assert_eq!(unwrapped.shape(), Shape::from(shape));

        let zeros = TensorData::zeros::<f32, _>(shape);
        unwrapped.cell.to_data().assert_eq(&zeros, true);
        unwrapped.hidden.to_data().assert_eq(&zeros, true);
    }
}
