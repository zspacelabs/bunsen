//! Functional 2D convolution.

use burn::{
    Tensor,
    tensor::kind::Basic,
};

/// Convolves a neighborhood function over a 2D tensor.
///
/// Here ``h_wins`` and ``w_wins`` refer to
/// ``burn::tensor::ops::unfold::calculate_unfold_windows(dim, kernel, 1)``
///
/// # Arguments
///
/// * `input` - a `[batch, c_in, height, width]` tensor.
/// * `func` - a func from `[batch, h_wins, w_wins, c_in, kernel[0], kernel[1]]`
///   to `[batch, h_wins, w_wins, c_out]`
/// * `kernel` - a kernel shape, e.g. `[3, 3]`.
/// * `stride` - a kernel stride, e.g. `[1, 1]`.
///
/// # Returns
///
/// A tensor in `[batch, c_out, h_wins, w_wins]`
pub fn convolve_func_2d<KIn, KOut, F>(
    input: Tensor<4, KIn>,
    func: F,
    kernel: [usize; 2],
    stride: [usize; 2],
) -> Tensor<4, KOut>
where
    KIn: Basic,
    KOut: Basic,
    F: Fn(Tensor<6, KIn>) -> Tensor<4, KOut>,
{
    #[cfg(debug_assertions)]
    use crate::contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    };

    #[cfg(debug_assertions)]
    let [batch, c_in, height, width] =
        unpack_shape_contract!(["batch", "c_in", "height", "width"], &input);
    #[cfg(debug_assertions)]
    let h_wins = (height + stride[0]).saturating_sub(kernel[0]) / stride[0];
    #[cfg(debug_assertions)]
    let w_wins = (width + stride[1]).saturating_sub(kernel[1]) / stride[1];

    let x: Tensor<6, KIn> = input
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

    let x: Tensor<6, KIn> = x.permute([0, 2, 3, 1, 4, 5]);

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
