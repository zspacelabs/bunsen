//! Convolution shape arithmetic.

use alloc::{
    format,
    string::ToString,
    vec::Vec,
};

use burn::{
    module::Initializer,
    nn::PaddingConfig2d,
};

use crate::{
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
        WithOkOrPanic,
    },
};

/// The output size of a 1D convolution along one axis:
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
/// - `input_size`: The input dimension size, > 0.
/// - `kernel_size`: The kernel size, > 0.
/// - `stride`: The stride of the convolution, > 0.
/// - `padding`: The padding of the convolution, added evenly to both sides of
///   the input.
/// - `dilation`: The dilation of the convolution, > 0.
///
/// # Errors
///
/// A [`ConstraintError`] for a zero size, kernel, stride or dilation, or a
/// kernel that does not fit the padded input (no output).
pub fn try_conv1d_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> BunsenResult<usize> {
    const OWNER: &str = "try_conv1d_output_size";
    for (name, value) in [
        ("input_size", input_size),
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("dilation", dilation),
    ] {
        if value == 0 {
            return Err(ConstraintError::zero_or_empty(OWNER, name).into());
        }
    }
    let pos = input_size + 2 * padding + stride;
    let kernel_width = 1 + dilation * (kernel_size - 1);
    match pos.checked_sub(kernel_width).map(|x| x / stride) {
        Some(x) if x >= 1 => Ok(x),
        _ => Err(ConstraintError::custom(
            OWNER,
            "input_size",
            format!(
                "{input_size} has no output under kernel_size {kernel_size}, stride {stride}, \
                 padding {padding}, dilation {dilation}"
            ),
        )
        .into()),
    }
}

/// [`try_conv1d_output_size`], for a geometry known to be sound.
///
/// # Panics
///
/// When it is not.
pub fn conv1d_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    try_conv1d_output_size(input_size, kernel_size, stride, padding, dilation).ok_or_panic()
}

/// The output shape of a `D`-dimensional convolution:
/// [`try_conv1d_output_size`] along each axis.
///
/// # Errors
///
/// As [`try_conv1d_output_size`], on the first axis that fails, naming it.
pub fn try_conv_output_shape<const D: usize>(
    input_shape: [usize; D],
    kernel_shape: [usize; D],
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> BunsenResult<[usize; D]> {
    let mut output_shape = input_shape;
    for i in 0..D {
        output_shape[i] = try_conv1d_output_size(
            input_shape[i],
            kernel_shape[i],
            stride[i],
            padding[i],
            dilation[i],
        )
        .context(format!("axis {i} of {input_shape:?}"))?;
    }
    Ok(output_shape)
}

/// [`try_conv_output_shape`], for a geometry known to be sound.
///
/// # Panics
///
/// When it is not.
pub fn conv_output_shape<const D: usize>(
    input_shape: [usize; D],
    kernel_shape: [usize; D],
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> [usize; D] {
    try_conv_output_shape(input_shape, kernel_shape, stride, padding, dilation).ok_or_panic()
}

/// [`try_conv_output_shape`] over slices, for a rank known only at run time.
///
/// # Errors
///
/// A [`ConstraintError`] when the slices' lengths differ from
/// `input_shape`'s; as [`try_conv1d_output_size`] on each axis.
pub fn try_conv_output_shape_dyn(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> BunsenResult<Vec<usize>> {
    let rank = input_shape.len();
    for (name, len) in [
        ("kernel_shape", kernel_shape.len()),
        ("stride", stride.len()),
        ("padding", padding.len()),
        ("dilation", dilation.len()),
    ] {
        if len != rank {
            return Err(ConstraintError::new(
                "try_conv_output_shape_dyn",
                name,
                Rule::Relation {
                    lhs: (format!("{name}'s rank"), len.to_string()),
                    op: "==",
                    rhs: ("input_shape's rank".into(), rank.to_string()),
                },
            )
            .into());
        }
    }
    (0..rank)
        .map(|i| {
            try_conv1d_output_size(
                input_shape[i],
                kernel_shape[i],
                stride[i],
                padding[i],
                dilation[i],
            )
            .context(format!("axis {i} of {input_shape:?}"))
        })
        .collect()
}

/// [`try_conv_output_shape_dyn`], for a geometry known to be sound.
///
/// # Panics
///
/// When it is not.
pub fn conv_output_shape_dyn(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> Vec<usize> {
    try_conv_output_shape_dyn(input_shape, kernel_shape, stride, padding, dilation).ok_or_panic()
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
                                try_conv1d_output_size(
                                    input_shape,
                                    kernel_shape,
                                    stride,
                                    padding,
                                    dilation,
                                )
                                .ok(),
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

    /// The `try_` forms refuse what has no output; their twins panic.
    #[test]
    fn test_conv_output_refusals() {
        use crate::errors::testing::ErrorMatcher;

        assert_eq!(conv1d_output_size(8, 3, 2, 1, 1), 4);
        assert_eq!(
            conv_output_shape([8, 6], [3, 3], [2, 1], [1, 0], [1, 1]),
            [4, 4]
        );
        assert_eq!(
            conv_output_shape_dyn(&[8, 6], &[3, 3], &[2, 1], &[1, 0], &[1, 1]),
            [4, 4]
        );
        ErrorMatcher::illegal()
            .constraint("try_conv1d_output_size", "stride")
            .assert_err(&try_conv1d_output_size(8, 3, 0, 0, 1));
        ErrorMatcher::illegal()
            .constraint("try_conv1d_output_size", "input_size")
            .assert_err(&try_conv1d_output_size(2, 5, 1, 0, 1));
        ErrorMatcher::illegal()
            .constraint("try_conv1d_output_size", "input_size")
            .assert_err(&try_conv_output_shape(
                [8, 2],
                [3, 5],
                [1, 1],
                [0, 0],
                [1, 1],
            ));
        ErrorMatcher::illegal()
            .constraint("try_conv_output_shape_dyn", "stride")
            .assert_err(&try_conv_output_shape_dyn(&[8], &[3], &[1, 1], &[0], &[1]));
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
