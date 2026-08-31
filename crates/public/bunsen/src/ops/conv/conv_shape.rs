//! # Convolution Shape Utilities
//!
//! Utilities for computing the output shape of convolution operations.

use alloc::vec::Vec;

use burn::{
    module::Initializer,
    nn::PaddingConfig2d,
};

use crate::contracts::unpack_shape_contract;

/// Predicts the output size of a 1D convolution operation.
///
/// ```text
/// out_size = floor( ((in_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1 )
/// ```
///
/// # Reference
///
/// - [conv_arithmetic diagram](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
///   visual explanations of these convolution parameters.
/// - [pytorch conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
///
/// # Arguments
///
/// - `input_size`: The input dimension size, must be > 0.
/// - `kernel_size`: The kernel size, must be > 0.
/// - `stride`: The stride of the convolution, must be > 0.
/// - `padding`: The padding of the convolution, added evenly to all sides of
///   the input.
/// - `dilation`: The dilation of the convolution, must be > 0.
///
/// # Returns
///
/// An `Option<usize>` representing the output size; or `None` for <= 0.
pub fn maybe_conv1d_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    assert!(input_size > 0);
    assert!(kernel_size > 0);
    assert!(stride > 0);
    assert!(dilation > 0);

    let effective_size = input_size + 2 * padding;
    let pos = effective_size + stride;
    let kernel_width = 1 + dilation * (kernel_size - 1);

    if pos < kernel_width {
        return None;
    }
    let x = (pos - kernel_width) / stride;
    if x < 1 { None } else { Some(x) }
}

/// Predicts the output size of a 1D convolution operation.
///
/// This is the ``panic``-ing variant of [`maybe_conv1d_output_size`].
///
/// ```text
/// out_size = floor( ((in_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1 )
/// ```
///
/// # Reference
///
/// - [conv_arithmetic diagram](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
///   visual explanations of these convolution parameters.
/// - [pytorch conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
///
/// # Arguments
///
/// - `input_size`: The input dimension size, must be > 0.
/// - `kernel_size`: The kernel size, must be > 0.
/// - `stride`: The stride of the convolution, must be > 0.
/// - `padding`: The padding of the convolution, added evenly to all sides of
///   the input.
/// - `dilation`: The dilation of the convolution, must be > 0.
///
/// # Returns
///
/// The output size of the convolution operation.
///
/// # Panics
///
/// If the output size would be <= 0.
pub fn expect_conv1d_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    match maybe_conv1d_output_size(input_size, kernel_size, stride, padding, dilation) {
        Some(x) => x,
        None => panic!(
            "No legal output size for conv1d with:\n input_size:{input_size}\n kernel_size:{kernel_size}\n stride:{stride}\n dilation:{dilation}\n padding:{padding}",
        ),
    }
}

/// Predicts the output shape of a D convolution operation; for dynamic slices.
///
/// This is the generalization of [`maybe_conv1d_output_size`] to D dimensions.
///
/// # Arguments
///
/// - `input_shape`: The input dimension shape, each dim must be > 0.
/// - `kernel_shape`: The kernel shape; length must match `input_shape`, each
///   dim must be > 0.
/// - `stride`: The stride of the convolution; length must match `input_shape`,
///   each dim must be > 0.
/// - `padding`: The padding of the convolution, added evenly to all sides of
///   the input.
/// - `dilation`: The dilation of the convolution; length must match
///   `input_shape`, each dim must be > 0.
///
/// # Returns
///
/// An `Option<[usize; D]>` representing the output shape; or `None` for <= 0.
pub fn maybe_conv_output_shape_dyn(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> Option<Vec<usize>> {
    let rank = input_shape.len();
    assert_eq!(kernel_shape.len(), rank);
    assert_eq!(stride.len(), rank);
    assert_eq!(dilation.len(), rank);
    assert_eq!(padding.len(), rank);

    let mut output_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        output_shape.push(maybe_conv1d_output_size(
            input_shape[i],
            kernel_shape[i],
            stride[i],
            padding[i],
            dilation[i],
        )?);
    }
    Some(output_shape)
}

/// Predicts the output shape of a D convolution operation.
///
/// This is the ``panic``-ing variant of [`maybe_conv_output_shape_dyn`];
/// which is the generalization of [`maybe_conv1d_output_size`] to D dimensions.
///
/// # Arguments
///
/// - `input_shape`: The input dimension shape, each dim must be > 0.
/// - `kernel_shape`: The kernel shape, each dim must be > 0.
/// - `stride`: The stride of the convolution, each dim must be > 0.
/// - `padding`: The padding of the convolution, added evenly to all sides of
///   the input.
/// - `dilation`: The dilation of the convolution, each dim must be > 0.
///
/// # Returns
///
/// An `Option<Vec<usize>>` representing the output shape; or `None` for <= 0.
pub fn expect_conv_output_shape_dyn(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> Vec<usize> {
    match maybe_conv_output_shape_dyn(input_shape, kernel_shape, stride, padding, dilation) {
        Some(shape) => shape,
        None => panic!(
            "No legal output size for conv with:\n input_shape:{input_shape:?}\n kernel_shape:{kernel_shape:?}\n stride:{stride:?}\n dilation:{dilation:?}\n padding:{padding:?}",
        ),
    }
}

/// Predicts the output shape of a D convolution operation.
///
/// This is the generalization of [`maybe_conv1d_output_size`] to D dimensions.
///
/// # Arguments
///
/// - `input_shape`: The input dimension shape, each dim must be > 0.
/// - `kernel_shape`: The kernel shape, each dim must be > 0.
/// - `stride`: The stride of the convolution, each dim must be > 0.
/// - `padding`: The padding of the convolution, added evenly to all sides of
///   the input.
/// - `dilation`: The dilation of the convolution, each dim must be > 0.
///
/// # Returns
///
/// An `Option<[usize; D]>` representing the output shape; or `None` for <= 0.
pub fn maybe_conv_output_shape<const D: usize>(
    input_shape: [usize; D],
    kernel_shape: [usize; D],
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> Option<[usize; D]> {
    let mut output_shape = input_shape;
    for i in 0..D {
        output_shape[i] = maybe_conv1d_output_size(
            input_shape[i],
            kernel_shape[i],
            stride[i],
            padding[i],
            dilation[i],
        )?;
    }
    Some(output_shape)
}

/// Predicts the output shape of a D convolution operation.
///
/// This is the ``panic``-ing variant of [`maybe_conv_output_shape`];
/// which is the generalization of [`maybe_conv1d_output_size`] to D dimensions.
///
/// # Arguments
///
/// - `input_shape`: The input dimension shape, each dim must be > 0.
/// - `kernel_shape`: The kernel shape, each dim must be > 0.
/// - `stride`: The stride of the convolution, each dim must be > 0.
/// - `padding`: The padding of the convolution, added evenly to all sides of
///   the input.
/// - `dilation`: The dilation of the convolution, each dim must be > 0.
///
/// # Returns
///
/// An `Option<[usize; D]>` representing the output shape; or `None` for <= 0.
pub fn expect_conv_output_shape<const D: usize>(
    input_shape: [usize; D],
    kernel_shape: [usize; D],
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> [usize; D] {
    match maybe_conv_output_shape(input_shape, kernel_shape, stride, padding, dilation) {
        Some(shape) => shape,
        None => panic!(
            "No legal output size for conv with:\n input_shape:{input_shape:?}\n kernel_shape:{kernel_shape:?}\n stride:{stride:?}\n dilation:{dilation:?}\n padding:{padding:?}",
        ),
    }
}

/// Returns the output resolution for a given input resolution.
///
/// The input must be a multiple of the stride.
///
/// # Arguments
///
/// - `input_resolution`: `[in_height=out_height*stride,
///   in_width=out_width*stride]`.
///
/// # Returns
///
/// `[out_height, out_width]`
///
/// # Panics
///
/// If the input resolution is not a multiple of the stride.
pub fn stride_div_output_resolution(
    input_resolution: [usize; 2],
    stride: usize,
) -> [usize; 2] {
    unpack_shape_contract!(
        [
            "in_height" = "out_height" * "stride",
            "in_width" = "out_width" * "stride"
        ],
        &input_resolution,
        &["out_height", "out_width"],
        &[("stride", stride)]
    )
}

/// Recommended initializer for conv layers feeding into a relu.
pub static CONV_INTO_RELU_INITIALIZER: Initializer = Initializer::KaimingNormal {
    gain: core::f64::consts::SQRT_2,
    fan_out_only: true,
};

/// Computes the necessary [`burn::nn::conv::Conv2d`] padding for the given
/// square parameters.
///
/// All parameters are assumed square (the same in height and width).
///
/// # Arguments
///
/// - `kernel`: The size of the kernel; must be odd.
/// - `stride`: The stride of the convolution; must be >= 1.
/// - `dilation`: The dilation of the convolution; must be >= 1.
///
/// # Returns
///
/// Computed padding.
pub fn get_square_conv2d_padding(
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    assert_eq!(kernel % 2, 1, "Kernel size must be odd");
    assert!(stride >= 1, "Stride must be >= 1");
    assert!(dilation >= 1, "Dilation must be >= 1");
    ((stride - 1) + dilation * (kernel - 1)) / 2
}

/// Computes the necessary [`burn::nn::conv::Conv2d`] padding for the given
/// square parameters.
///
/// All parameters are assumed square (the same in height and width).
///
/// # Arguments
///
/// - `kernel`: The size of the kernel; must be odd.
/// - `stride`: The stride of the convolution; must be >= 1.
/// - `dilation`: The dilation of the convolution; must be >= 1.
///
/// # Returns
///
/// Computed [`PaddingConfig2d`].
pub fn build_square_conv2d_padding_config(
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> PaddingConfig2d {
    let p = get_square_conv2d_padding(kernel, stride, dilation);
    PaddingConfig2d::Explicit(p, p, p, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_output_shape() {
        pub fn conv1d_output_size_reference(
            input_shape: usize,
            kernel_shape: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
        ) -> Option<usize> {
            let input_shape = input_shape as f64;
            let kernel_shape = kernel_shape as f64;
            let stride = stride as f64;
            let dilation = dilation as f64;
            let padding = padding as f64;

            let effective_shape = input_shape + 2.0 * padding;
            let kernel_width = 1.0 + dilation * (kernel_shape - 1.0);

            let x = (((effective_shape - kernel_width) / stride) + 1.0).floor();
            if x < 1.0 { None } else { Some(x as usize) }
        }

        for input_shape in 1..10 {
            for stride in 1..3 {
                for kernel_shape in 1..4 {
                    for dilation in 1..2 {
                        for padding in 0..10 {
                            assert_eq!(
                                maybe_conv1d_output_size(
                                    input_shape,
                                    kernel_shape,
                                    stride,
                                    padding,
                                    dilation,
                                ),
                                conv1d_output_size_reference(
                                    input_shape,
                                    kernel_shape,
                                    stride,
                                    padding,
                                    dilation,
                                )
                            )
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_get_padding() {
        assert_eq!(get_square_conv2d_padding(1, 1, 1), 0);
        assert_eq!(get_square_conv2d_padding(3, 1, 1), 1);
        assert_eq!(get_square_conv2d_padding(5, 1, 1), 2);

        assert_eq!(get_square_conv2d_padding(1, 2, 1), 0);
        assert_eq!(get_square_conv2d_padding(3, 2, 1), 1);
        assert_eq!(get_square_conv2d_padding(5, 2, 1), 2);

        assert_eq!(get_square_conv2d_padding(1, 1, 2), 0);
        assert_eq!(get_square_conv2d_padding(3, 1, 2), 2);
        assert_eq!(get_square_conv2d_padding(5, 1, 2), 4);

        assert_eq!(get_square_conv2d_padding(1, 2, 2), 0);
        assert_eq!(get_square_conv2d_padding(3, 2, 2), 2);
        assert_eq!(get_square_conv2d_padding(5, 2, 2), 4);
    }

    #[test]
    #[should_panic(expected = "Kernel size must be odd")]
    fn test_get_padding_panic() {
        get_square_conv2d_padding(2, 1, 1);
    }

    #[test]
    #[should_panic(expected = "Stride must be >= 1")]
    fn test_get_padding_panic_stride() {
        get_square_conv2d_padding(1, 0, 1);
    }

    #[test]
    #[should_panic(expected = "Dilation must be >= 1")]
    fn test_get_padding_panic_dilation() {
        get_square_conv2d_padding(1, 1, 0);
    }

    #[test]
    fn test_build_square_conv2d_padding_config() {
        assert_eq!(
            build_square_conv2d_padding_config(1, 1, 1),
            PaddingConfig2d::Explicit(0, 0, 0, 0)
        );

        assert_eq!(
            build_square_conv2d_padding_config(3, 2, 2),
            PaddingConfig2d::Explicit(2, 2, 2, 2)
        );
    }
}
