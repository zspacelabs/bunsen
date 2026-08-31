//! # Padded splitting.

use burn::{
    Tensor,
    prelude::Backend,
    tensor::AsIndex,
};

/// Splits a tensor along `dim` into chunks of exactly `size`, zero-padding the
/// last one when the axis does not divide evenly.
///
/// [`Tensor::split`] returns a short final chunk; this pads it instead, so
/// every chunk has the same length along `dim` and the results are
/// interchangeable — which is what a model with a fixed input width needs. An
/// axis of length zero yields no chunks at all.
///
/// # Arguments
/// - `input` - the input tensor.
/// - `size` - the chunk length along `dim`.
/// - `dim` - the dim to split; supports negative indexing.
///
/// # Returns
/// - the chunks, in order, each of length `size` along `dim`.
///
/// # Panics
/// Panics if `size` is zero.
///
/// # Examples
/// ```rust, ignore
/// // A length-5 axis split into chunks of 2 yields 2 + 2 + (1 padded to 2).
/// let chunks = split_padded(input, 2, 1);
/// assert_eq!(chunks.len(), 3);
/// ```
pub fn split_padded<B: Backend, const R: usize, D: AsIndex>(
    input: Tensor<B, R>,
    size: usize,
    dim: D,
) -> Vec<Tensor<B, R>> {
    let dim = dim.expect_dim_index(R);
    assert_ne!(size, 0, "split_padded size must be non-zero");

    let device = input.device();

    input
        .split(size, dim)
        .into_iter()
        .map(|chunk| {
            let have = chunk.dims()[dim];
            if have == size {
                chunk
            } else {
                let mut pad = chunk.dims();
                pad[dim] = size - have;
                Tensor::cat(vec![chunk, Tensor::zeros(pad, &device)], dim)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;

    #[test]
    fn test_split_padded_pads_the_short_chunk() {
        let device = Default::default();
        let input =
            Tensor::<B, 2>::from_data([[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.]], &device);

        let chunks = split_padded(input, 2, 1);
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.dims(), [2, 2]);
        }

        chunks[2]
            .to_data()
            .assert_eq(&TensorData::from([[4.0_f32, 0.0], [9.0, 0.0]]), true);
    }

    #[test]
    fn test_split_padded_exact_division_does_not_pad() {
        let device = Default::default();
        let input = Tensor::<B, 2>::from_data([[0., 1., 2., 3.], [4., 5., 6., 7.]], &device);

        let chunks = split_padded(input.clone(), 2, 1);
        assert_eq!(chunks.len(), 2);
        Tensor::cat(chunks, 1)
            .to_data()
            .assert_eq(&input.to_data(), true);
    }

    #[test]
    fn test_split_padded_accepts_a_negative_dim() {
        let device = Default::default();
        let input = Tensor::<B, 2>::from_data([[0., 1., 2.], [3., 4., 5.]], &device);

        let by_neg = split_padded(input.clone(), 2, -1);
        let by_pos = split_padded(input, 2, 1);

        assert_eq!(by_neg.len(), by_pos.len());
        for (a, b) in by_neg.into_iter().zip(by_pos) {
            a.to_data().assert_eq(&b.to_data(), true);
        }
    }

    #[test]
    fn test_split_padded_empty_axis_yields_nothing() {
        let device = Default::default();
        let input = Tensor::<B, 2>::zeros([2, 0], &device);
        assert!(split_padded(input, 4, 1).is_empty());
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn test_split_padded_rejects_zero_size() {
        let device = Default::default();
        let input = Tensor::<B, 2>::zeros([2, 4], &device);
        let _ = split_padded(input, 0, 1);
    }
}
