//! TensorFlow-style "SAME" padding.

use burn::prelude::Tensor;

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
/// It is the padding that lets `ceil(size / stride)` windows cover the axis,
/// TensorFlow's "SAME" output length:
///
/// ```text
/// span = (ceil(size / stride) - 1) * stride + (kernel_size - 1) * dilation + 1
/// padding = max(span - size, 0)
/// ```
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
///
/// # Panics
///
/// If `stride` is 0.
pub fn get_same_padding(
    size: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    // `span - size`, with the two `- 1` terms moved to the subtracted side so
    // that no `usize` step goes below zero; saturating is the `max(.., 0)`.
    let reach = size.div_ceil(stride) * stride + kernel_size * dilation + 1;
    reach.saturating_sub(size + stride + dilation)
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
pub fn pad_same(
    input: Tensor<4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    value: f32,
) -> Tensor<4> {
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
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        performance_device,
    };

    /// `(kernel_size, stride, dilation, [padding for size 1..=12])`, from the
    /// Python reference in this file (computed with python3, not by hand).
    const TIMM_REFERENCE: &[(usize, usize, usize, [usize; 12])] = &[
        (1, 1, 1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (1, 2, 1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (1, 3, 1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (1, 4, 1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (2, 1, 1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (2, 2, 1, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        (2, 3, 1, [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
        (2, 4, 1, [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
        (3, 1, 1, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
        (3, 2, 1, [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]),
        (3, 3, 1, [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]),
        (3, 4, 1, [2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0]),
        (7, 1, 1, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
        (7, 2, 1, [6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5]),
        (7, 3, 1, [6, 5, 4, 6, 5, 4, 6, 5, 4, 6, 5, 4]),
        (7, 4, 1, [6, 5, 4, 3, 6, 5, 4, 3, 6, 5, 4, 3]),
        (3, 1, 2, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]),
        (3, 2, 2, [4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3]),
        (3, 3, 2, [4, 3, 2, 4, 3, 2, 4, 3, 2, 4, 3, 2]),
        (3, 4, 2, [4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1]),
    ];

    #[test]
    fn test_get_same_padding_matches_timm_reference() {
        for &(kernel_size, stride, dilation, pads) in TIMM_REFERENCE {
            for (size, expected) in (1..=12).zip(pads) {
                assert_eq!(
                    get_same_padding(size, kernel_size, stride, dilation),
                    expected,
                    "size={size} kernel_size={kernel_size} stride={stride} dilation={dilation}"
                );
            }
        }
    }

    #[test]
    fn test_get_same_padding_counts_a_partial_last_window() {
        // A partial last window counts as an output position:
        // `ceil(size / stride)` positions, not `size / stride` rounded.
        // `(size, kernel_size, stride, dilation, expected)`, from python3.
        for (size, kernel_size, stride, dilation, expected) in [
            (4, 7, 3, 1, 6),
            (10, 7, 3, 1, 6),
            (5, 7, 4, 1, 6),
            (9, 7, 4, 1, 6),
            (4, 3, 3, 2, 4),
            (5, 3, 4, 2, 4),
        ] {
            assert_eq!(
                get_same_padding(size, kernel_size, stride, dilation),
                expected,
                "size={size} kernel_size={kernel_size} stride={stride} dilation={dilation}"
            );
        }
    }

    #[test]
    #[serial]
    fn test_pad_same_puts_the_odd_pixel_bottom_right() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        // Height 10 pads by 2 (1 + 1); width 8 pads by 1 (0 + 1).
        let input = Tensor::<4>::ones([1, 1, 10, 8], &device);
        let output = pad_same(input.clone(), [3, 3], [3, 3], [1, 1], 0.0);

        assert_eq!(output.dims(), [1, 1, 12, 9]);
        output
            .clone()
            .slice([0..1, 0..1, 1..11, 0..8])
            .to_data()
            .assert_eq(&input.to_data(), true);
        assert_eq!(output.sum().into_scalar::<f32>(), 80.0);
    }
}
