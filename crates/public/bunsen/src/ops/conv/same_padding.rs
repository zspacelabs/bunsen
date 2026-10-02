//! TensorFlow-style "SAME" padding.

use core::cmp::max;

use burn::prelude::{
    Backend,
    Tensor,
};

// # Reference Python
//
// ```python,ignore
// # Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
// def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
//     if isinstance(x, torch.Tensor):
//         return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
//     else:
//         return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)
//
// # Dynamically pad input x with 'SAME' padding for conv with specified args
// def pad_same(
//         x,
//         kernel_size: List[int],
//         stride: List[int],
//         dilation: List[int] = (1, 1),
//         value: float = 0,
//     ):
//     ih, iw = x.size()[-2:]
//     pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
//     pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
//     x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
//     return x
// ```

/// Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution.
///
/// Ported from `timm`'s `get_same_padding` (the Python reference is in this
/// file's source). Used by [`pad_same`].
///
/// # Arguments
///
/// - `size`: the input length along one axis.
/// - `kernel_size`: the window size along that axis.
/// - `stride`: the window stride along that axis.
/// - `dilation`: the window dilation along that axis.
///
/// # Returns
///
/// The total padding for the axis; [`pad_same`] splits it between the two
/// sides.
pub fn get_same_padding(
    size: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    max(
        (((size + (stride / 2)) / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - size,
        0,
    )
}

/// Dynamically pad input x with 'SAME' padding for conv with specified args.
///
/// Each spatial axis gets [`get_same_padding`]'s total, split so that the odd
/// pixel goes to the bottom / right, as TensorFlow does.
///
/// # Arguments
///
/// - `input`: a `[batch, channels, height, width]` input tensor.
/// - `kernel_size`: `[height, width]` window size.
/// - `stride`: `[height, width]` window stride.
/// - `dilation`: `[height, width]` window dilation.
/// - `value`: the fill value for the padding.
///
/// # Returns
///
/// The `[batch, channels, height + pad_h, width + pad_w]` padded tensor.
pub fn pad_same<B: Backend>(
    input: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    value: f32,
) -> Tensor<B, 4> {
    let ih = input.shape()[2];
    let iw = input.shape()[3];
    let pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0]);
    let pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1]);
    input.pad(
        (pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2),
        value,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_same_padding() {
        assert_eq!(get_same_padding(10, 1, 1, 1), 0);

        assert_eq!(get_same_padding(10, 3, 2, 1), 1);

        assert_eq!(get_same_padding(10, 3, 2, 2), 3);
    }
}
