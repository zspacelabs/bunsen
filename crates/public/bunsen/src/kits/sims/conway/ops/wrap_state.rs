//! Wrap toroidal board state.

use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        BasicOps,
        Slice,
    },
};

use crate::prelude::TensorOpExt;

/// Project wrapped toroidal boarders.
///
/// This is a utility mechanism for simulation updates in toroidal space.
/// This assumes that the current valid board state is 1-unit in from
/// the edges of the board; as produced by a 3x3 neighborhood window update
/// function.
///
/// Given the 1D space: "ZABCDEZ", this will produce "EABCDEA".
pub fn project_wrapped_toroidal_boarders<B, const R: usize, K>(
    state: Tensor<B, R, K>
) -> Tensor<B, R, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn mk_slices<const R: usize>(
        dim: usize,
        idx: isize,
    ) -> [Slice; R] {
        let mut slices = [Slice::full(); R];
        slices[dim] = Slice::index(idx);
        slices
    }

    let mut state = state;

    for d in 0..R {
        state = state
            .copy_slice(mk_slices::<R>(d, -1), mk_slices::<R>(d, 1))
            .copy_slice(mk_slices::<R>(d, 0), mk_slices::<R>(d, -2));
    }

    state
}

#[cfg(test)]
mod test {
    use burn::{
        prelude::{
            Int,
            Shape,
            Tensor,
            s,
        },
        tensor::TensorData,
    };

    use super::*;
    use crate::{
        burner::tensor::*,
        support::testing::{
            PerformanceBackend,
            default_device,
        },
    };

    #[test]
    #[serial_test::serial]
    fn test_wrap_state_2d() {
        type B = PerformanceBackend;
        let device = default_device();

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

        let wrapped = project_wrapped_toroidal_boarders(state);

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
        let device = default_device();

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

        let wrapped = project_wrapped_toroidal_boarders(state);

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
