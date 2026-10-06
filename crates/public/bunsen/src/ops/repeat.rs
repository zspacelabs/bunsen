//! # Repeat-interleave
//!
//! [`repeat_interleave`] repeats each element in place, so `[a, b]` becomes
//! `[a, a, b, b]`, as `NumPy`'s `repeat` and `PyTorch`'s `repeat_interleave`
//! do. burn's `Tensor::repeat_dim` tiles the whole axis instead, giving
//! `[a, b, a, b]`.

use burn::{
    Tensor,
    tensor::AsIndex,
};

/// Repeat Interleave.
///
/// Repeats elements of a tensor, interleaved from their existing locations.
///
/// # Arguments
/// - `input` - the input tensor.
/// - `repeats` - the number of repeats.
/// - `dim` - the dim to repeat; supports negative indexing.
///
/// # Returns
/// - the interleaved tensor.
///
/// # Examples
/// ```rust
/// use bunsen::{
///     ops::repeat::repeat_interleave,
///     support::testing::cpu_device,
/// };
/// use burn::Tensor;
///
/// let device = cpu_device();
///
/// let input = Tensor::<2>::from_data([[0., 1., 2.], [3., 4., 5.]], &device);
///
/// let result: Tensor<2> = repeat_interleave::<2, 3, _>(input, 3, 1);
///
/// result.to_data().assert_eq(
///     &Tensor::<2>::from_data(
///         [
///             [0., 0., 0., 1., 1., 1., 2., 2., 2.],
///             [3., 3., 3., 4., 4., 4., 5., 5., 5.],
///         ],
///         &device,
///     )
///     .to_data(),
///     true,
/// );
/// ```
pub fn repeat_interleave<const R: usize, const R2: usize, D: AsIndex>(
    input: Tensor<R>,
    repeats: usize,
    dim: D,
) -> Tensor<R> {
    let dim = dim.expect_dim_index(R);

    let x: Tensor<R2> = input.unsqueeze_dim(dim + 1);

    let mut dims = x.dims();
    dims[dim + 1] = repeats;

    let x = x.expand(dims);

    x.flatten(dim, dim + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_repeat_interleave() {
        let device = cpu_device();

        let input = Tensor::<2>::from_data([[0., 1., 2.], [3., 4., 5.]], &device);

        repeat_interleave::<2, 3, _>(input.clone(), 3, 1)
            .to_data()
            .assert_eq(
                &Tensor::<2>::from_data(
                    [
                        [0., 0., 0., 1., 1., 1., 2., 2., 2.],
                        [3., 3., 3., 4., 4., 4., 5., 5., 5.],
                    ],
                    &device,
                )
                .to_data(),
                true,
            );

        repeat_interleave::<2, 3, _>(input.clone(), 3, 0)
            .to_data()
            .assert_eq(
                &Tensor::<2>::from_data(
                    [
                        [0., 1., 2.],
                        [0., 1., 2.],
                        [0., 1., 2.],
                        [3., 4., 5.],
                        [3., 4., 5.],
                        [3., 4., 5.],
                    ],
                    &device,
                )
                .to_data(),
                true,
            );
    }
}
