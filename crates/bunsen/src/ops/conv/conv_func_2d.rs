//! # Surface Convolutions (2D)

use burn::{
    Tensor,
    prelude::Backend,
    tensor::BasicOps,
};

/// Convolve a neighborhood function over a 2D tensor.
///
/// Here ``h_wins`` and ``w_wins`` refer to
/// ``burn::tensor::ops::unfold::calculate_unfold_windows(dim, kernel, 1)``
///
/// # Arguments
///
/// * `input` - a ``[batch, c_in, height, width]`` tensor.
/// * `func` - a func from ``[batch, h_wins, w_wins, c_in, kernel[0],
///   kernel[1]]`` to ``[batch, h_wins, w_wins, c_out]``
/// * `kernel` - a kernel shape, e.g. ``[3, 3]``.
/// * `stride` - a kernel stride, e.g. ``[1, 1]``.
///
/// # Returns
///
/// A tensor in ``[batch, c_out, h_wins, w_wins]``
pub fn convolve_func_2d<B, KIn, KOut, F>(
    input: Tensor<B, 4, KIn>,
    func: F,
    kernel: [usize; 2],
    stride: [usize; 2],
) -> Tensor<B, 4, KOut>
where
    B: Backend,
    KIn: BasicOps<B>,
    KOut: BasicOps<B>,
    F: Fn(Tensor<B, 6, KIn>) -> Tensor<B, 4, KOut>,
{
    #[cfg(debug_assertions)]
    use burn::tensor::ops::unfold::calculate_unfold_windows;

    #[cfg(debug_assertions)]
    use crate::contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    };

    #[cfg(debug_assertions)]
    let [batch, c_in, height, width] =
        unpack_shape_contract!(["batch", "c_in", "height", "width"], &input,);
    #[cfg(debug_assertions)]
    let h_wins = calculate_unfold_windows(height, kernel[0], stride[0]);
    #[cfg(debug_assertions)]
    let w_wins = calculate_unfold_windows(width, kernel[1], stride[1]);

    let x: Tensor<B, 6, KIn> = input
        .unfold::<5, usize>(2, kernel[0], stride[0])
        .unfold::<6, usize>(3, kernel[1], stride[1]);

    #[cfg(debug_assertions)]
    assert_shape_contract_periodically!(
        ["batch", "c_in", "h_wins", "w_wins", "kernel0", "kernel1"],
        x.shape().as_slice(),
        &[
            ("batch", batch),
            ("c_in", c_in),
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("kernel0", kernel[0]),
            ("kernel1", kernel[1]),
        ]
    );

    let x: Tensor<B, 6, KIn> = x.permute([0, 2, 3, 1, 4, 5]);

    #[cfg(debug_assertions)]
    assert_shape_contract_periodically!(
        ["batch", "h_wins", "w_wins", "c_in", "kernel0", "kernel1"],
        x.shape().as_slice(),
        &[
            ("batch", batch),
            ("c_in", c_in),
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("kernel0", kernel[0]),
            ("kernel1", kernel[1]),
        ]
    );

    let x = (func)(x);

    #[cfg(debug_assertions)]
    assert_shape_contract_periodically!(
        ["batch", "h_wins", "w_wins", "c_out"],
        x.shape().as_slice(),
        &[("batch", batch), ("h_wins", h_wins), ("w_wins", w_wins),]
    );

    x.permute([0, 3, 1, 2])
}
