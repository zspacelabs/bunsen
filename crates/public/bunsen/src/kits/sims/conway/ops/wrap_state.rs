//! Wrap toroidal board state.

use burn::{
    Tensor,
    prelude::{
        Backend,
        s,
    },
    tensor::BasicOps,
};

use crate::prelude::TensorOpExt;

/// Wraps the board state.
///
/// This simulates a toroidal space by copying the penultimate rows and columns
/// to the edges of the opposite sides.
pub fn wrap_state_2d<B, K>(state: Tensor<B, 2, K>) -> Tensor<B, 2, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    state
        .copy_slice(s![-1, ..], s![1, ..])
        .copy_slice(s![0, ..], s![-2, ..])
        .copy_slice(s![.., -1], s![.., 1])
        .copy_slice(s![.., 0], s![.., -2])
}

/// Wraps the board state.
///
/// This simulates a toroidal space by copying the penultimate rows and columns
/// to the edges of the opposite sides.
pub fn wrap_state_3d<B, K>(state: Tensor<B, 3, K>) -> Tensor<B, 3, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    state
        .copy_slice(s![-1, .., ..], s![1, .., ..])
        .copy_slice(s![0, .., ..], s![-2, .., ..])
        .copy_slice(s![.., -1, ..], s![.., 1, ..])
        .copy_slice(s![.., 0, ..], s![.., -2, ..])
        .copy_slice(s![.., .., -1], s![.., .., 1])
        .copy_slice(s![.., .., 0], s![.., .., -2])
}

#[cfg(test)]
mod test {
    use burn::{
        prelude::{
            Int,
            Shape,
            Tensor,
        },
        tensor::TensorData,
    };

    use super::*;
    use crate::{
        burner::tensor::*,
        support::testing::PerformanceBackend,
    };

    #[test]
    #[serial_test::serial]
    fn test_wrap_state_2d() {
        type B = PerformanceBackend;
        let device = Default::default();

        let shape: Shape = [5, 6].into();

        let state: Tensor<B, 2, Int> = Tensor::arange(0..shape.num_elements() as i64, &device)
            .add_scalar(100)
            .reshape(shape.clone());

        let state = state.slice_fill(s![0, ..], -10);
        let state = state.slice_fill(s![-1, ..], -10);
        let state = state.slice_fill(s![.., 0], -10);
        let state = state.slice_fill(s![.., -1], -10);

        state.to_data_as::<i32>().assert_eq(
            &TensorData::from([
                [-10, -10, -10, -10, -10, -10],
                [-10, 107, 108, 109, 110, -10],
                [-10, 113, 114, 115, 116, -10],
                [-10, 119, 120, 121, 122, -10],
                [-10, -10, -10, -10, -10, -10],
            ]),
            false,
        );

        let wrapped = wrap_state_2d(state);

        wrapped.to_data_as::<i32>().assert_eq(
            &TensorData::from([
                [122, 119, 120, 121, 122, 119],
                [110, 107, 108, 109, 110, 107],
                [116, 113, 114, 115, 116, 113],
                [122, 119, 120, 121, 122, 119],
                [110, 107, 108, 109, 110, 107],
            ]),
            false,
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_wrap_state_3d() {
        type B = PerformanceBackend;
        let device = Default::default();

        let shape: Shape = [4, 5, 6].into();

        let state: Tensor<B, 3, Int> = Tensor::arange(0..shape.num_elements() as i64, &device)
            .add_scalar(100)
            .reshape(shape.clone());

        let state = state.slice_fill(s![0, .., ..], -10);
        let state = state.slice_fill(s![-1, .., ..], -10);
        let state = state.slice_fill(s![.., 0, ..], -10);
        let state = state.slice_fill(s![.., -1, ..], -10);
        let state = state.slice_fill(s![.., .., 0], -10);
        let state = state.slice_fill(s![.., .., -1], -10);

        state.to_data_as::<i32>().assert_eq(
            &TensorData::from([
                [
                    [-10, -10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10, -10],
                ],
                [
                    [-10, -10, -10, -10, -10, -10],
                    [-10, 137, 138, 139, 140, -10],
                    [-10, 143, 144, 145, 146, -10],
                    [-10, 149, 150, 151, 152, -10],
                    [-10, -10, -10, -10, -10, -10],
                ],
                [
                    [-10, -10, -10, -10, -10, -10],
                    [-10, 167, 168, 169, 170, -10],
                    [-10, 173, 174, 175, 176, -10],
                    [-10, 179, 180, 181, 182, -10],
                    [-10, -10, -10, -10, -10, -10],
                ],
                [
                    [-10, -10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10, -10],
                ],
            ]),
            false,
        );

        let wrapped = wrap_state_3d(state);

        wrapped.to_data_as::<i32>().assert_eq(
            &TensorData::from([
                [
                    [182, 179, 180, 181, 182, 179],
                    [170, 167, 168, 169, 170, 167],
                    [176, 173, 174, 175, 176, 173],
                    [182, 179, 180, 181, 182, 179],
                    [170, 167, 168, 169, 170, 167],
                ],
                [
                    [152, 149, 150, 151, 152, 149],
                    [140, 137, 138, 139, 140, 137],
                    [146, 143, 144, 145, 146, 143],
                    [152, 149, 150, 151, 152, 149],
                    [140, 137, 138, 139, 140, 137],
                ],
                [
                    [182, 179, 180, 181, 182, 179],
                    [170, 167, 168, 169, 170, 167],
                    [176, 173, 174, 175, 176, 173],
                    [182, 179, 180, 181, 182, 179],
                    [170, 167, 168, 169, 170, 167],
                ],
                [
                    [152, 149, 150, 151, 152, 149],
                    [140, 137, 138, 139, 140, 137],
                    [146, 143, 144, 145, 146, 143],
                    [152, 149, 150, 151, 152, 149],
                    [140, 137, 138, 139, 140, 137],
                ],
            ]),
            false,
        );
    }
}
