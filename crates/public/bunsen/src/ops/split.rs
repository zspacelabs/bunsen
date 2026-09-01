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

/// Splits `dim` into a `[windows, window]` pair of axes, zero-padding the
/// tail when the axis does not divide evenly.
///
/// This is [`split_padded`] kept as a single tensor: rather than a `Vec` of
/// equal chunks, the split axis becomes two axes — the window index at `dim`,
/// the offset within a window at `dim + 1` — and the rank grows by one. Reach
/// for it when the windows are batched through the same op in one call, and
/// for [`split_padded`] when they are consumed one at a time.
///
/// The padding policy is the same: the last window is zero-filled out to
/// `window`, so every window is interchangeable. An axis of length zero yields
/// zero windows.
///
/// # Arguments
/// - `input` - the input tensor.
/// - `window` - the window length along `dim`.
/// - `dim` - the dim to window; supports negative indexing.
///
/// # Returns
/// - the windowed tensor; `dim` counts the windows, and `dim + 1` is `window`
///   wide.
///
/// # Panics
/// Panics if `window` is zero, or if `R2` is not `R + 1`.
///
/// # Examples
/// ```rust, ignore
/// // A length-5 axis windowed by 2 yields 3 windows, the last one padded.
/// let windows: Tensor<B, 3> = window_padded::<_, 2, 3, _>(input, 2, 1);
/// assert_eq!(windows.dims(), [2, 3, 2]);
/// ```
pub fn window_padded<B: Backend, const R: usize, const R2: usize, D: AsIndex>(
    mut input: Tensor<B, R>,
    window: usize,
    dim: D,
) -> Tensor<B, R2> {
    let dim = dim.expect_dim_index(R);
    assert_ne!(window, 0, "window_padded window must be non-zero");
    assert_eq!(R2, R + 1, "window_padded output rank must be R + 1");

    let mut len = input.dims()[dim];
    let remainder = len % window;
    if remainder != 0 {
        let extra = window - remainder;
        len += extra;

        let mut padding = [(0, 0); R];
        padding[dim] = (0, extra);
        input = input.pad(padding, 0.0);
    }

    let dims = input.dims();
    let mut shape = [0; R2];
    shape[..dim].copy_from_slice(&dims[..dim]);
    shape[dim] = len / window;
    shape[dim + 1] = window;
    shape[dim + 2..].copy_from_slice(&dims[dim + 1..]);

    input.reshape(shape)
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::Distribution,
    };

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

    #[test]
    fn test_window_padded_pads_the_short_window() {
        let device = Default::default();
        let input =
            Tensor::<B, 2>::from_data([[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.]], &device);

        let windows = window_padded::<_, 2, 3, _>(input, 2, 1);
        assert_eq!(windows.dims(), [2, 3, 2]);

        windows.to_data().assert_eq(
            &TensorData::from([
                [[0.0_f32, 1.0], [2.0, 3.0], [4.0, 0.0]],
                [[5.0, 6.0], [7.0, 8.0], [9.0, 0.0]],
            ]),
            true,
        );
    }

    #[test]
    fn test_window_padded_exact_division_does_not_pad() {
        let device = Default::default();
        let input = Tensor::<B, 2>::from_data([[0., 1., 2., 3.], [4., 5., 6., 7.]], &device);

        let windows = window_padded::<_, 2, 3, _>(input.clone(), 2, 1);
        assert_eq!(windows.dims(), [2, 2, 2]);

        // Windowing an evenly-divided axis is a pure reshape.
        windows
            .flatten::<2>(1, 2)
            .to_data()
            .assert_eq(&input.to_data(), true);
    }

    /// The window axis lands at `dim`, and the padding goes there too — not
    /// on whichever axis happens to be last.
    #[test]
    fn test_window_padded_windows_an_interior_dim() {
        let device = Default::default();
        let input = Tensor::<B, 3>::from_data([[[0., 1.], [2., 3.], [4., 5.]]], &device);

        let windows = window_padded::<_, 3, 4, _>(input, 2, 1);
        assert_eq!(windows.dims(), [1, 2, 2, 2]);

        windows.to_data().assert_eq(
            &TensorData::from([[[[0.0_f32, 1.0], [2.0, 3.0]], [[4.0, 5.0], [0.0, 0.0]]]]),
            true,
        );
    }

    #[test]
    fn test_window_padded_accepts_a_negative_dim() {
        let device = Default::default();
        let input = Tensor::<B, 2>::from_data([[0., 1., 2.], [3., 4., 5.]], &device);

        let by_neg = window_padded::<_, 2, 3, _>(input.clone(), 2, -1);
        let by_pos = window_padded::<_, 2, 3, _>(input, 2, 1);

        by_neg.to_data().assert_eq(&by_pos.to_data(), true);
    }

    #[test]
    fn test_window_padded_empty_axis_yields_no_windows() {
        let device = Default::default();
        let input = Tensor::<B, 2>::zeros([2, 0], &device);

        let windows = window_padded::<_, 2, 3, _>(input, 4, 1);
        assert_eq!(windows.dims(), [2, 0, 4]);
    }

    /// `window_padded` is `split_padded` stacked along the same dim.
    #[test]
    fn test_window_padded_matches_stacked_split_padded() {
        let device = Default::default();
        let input = Tensor::<B, 3>::random([2, 5, 3], Distribution::Default, &device);

        let stacked: Tensor<B, 4> = Tensor::stack(split_padded(input.clone(), 2, 1), 1);

        window_padded::<_, 3, 4, _>(input, 2, 1)
            .to_data()
            .assert_eq(&stacked.to_data(), true);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn test_window_padded_rejects_zero_window() {
        let device = Default::default();
        let input = Tensor::<B, 2>::zeros([2, 4], &device);
        let _: Tensor<B, 3> = window_padded(input, 0, 1);
    }

    #[test]
    #[should_panic(expected = "output rank")]
    fn test_window_padded_rejects_a_bad_output_rank() {
        let device = Default::default();
        let input = Tensor::<B, 2>::zeros([2, 4], &device);
        let _: Tensor<B, 4> = window_padded::<_, 2, 4, _>(input, 2, 1);
    }
}
