//! Conv variants: SAME padding and weight standardization.

use burn::{
    nn::conv::Conv2d,
    prelude::Tensor,
};

use super::{
    pad_same,
    weight_standardized_conv2d,
};

/// `conv`'s forward over `input` under timm's conv variants:
///
/// - `same_pad`: the input padded with zeros first, TensorFlow's "SAME" from
///   its own size ([`pad_same`]; timm's `Conv2dSame`), the conv itself
///   unpadded;
/// - `weight_std`: the conv's weight standardized by this `eps`
///   ([`weight_standardized_conv2d`]; timm's `StdConv2d`).
///
/// Both together are timm's `StdConv2dSame`; neither, `conv.forward`.
///
/// # Arguments
///
/// - `conv`: the conv; unpadded (`Valid`) under `same_pad`.
/// - `input`: `[batch, in_channels, height, width]`.
/// - `weight_std`: the standardization's `eps`; none when `None`.
/// - `same_pad`: whether to SAME-pad the input.
///
/// # Returns
///
/// `[batch, out_channels, out_height, out_width]`; under `same_pad`,
/// `ceil(height / stride)` by `ceil(width / stride)`.
pub fn conv2d_variant_forward(
    conv: &Conv2d,
    input: Tensor<4>,
    weight_std: Option<f64>,
    same_pad: bool,
) -> Tensor<4> {
    let input = match same_pad {
        true => pad_same(input, conv.kernel_size, conv.stride, conv.dilation, 0.0),
        false => input,
    };
    match weight_std {
        Some(eps) => weight_standardized_conv2d(conv, input, eps),
        None => conv.forward(input),
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::{
            PaddingConfig2d,
            conv::Conv2dConfig,
        },
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::support::testing::cpu_device;

    /// A SAME-padded stride-2 `3x3` conv over an even side pads only the
    /// bottom and right: it is the unpadded conv over the input with a zero
    /// row and column appended.
    #[test]
    fn test_same_pad_is_bottom_right_on_even_sides() {
        let device = cpu_device();
        let conv = Conv2dConfig::new([2, 3], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .init(&device);
        let x = Tensor::<4>::random([1, 2, 8, 6], Distribution::Default, &device);
        let ours = conv2d_variant_forward(&conv, x.clone(), None, true);
        assert_eq!(ours.dims(), [1, 3, 4, 3]);
        let expected = conv.forward(x.pad((0, 1, 0, 1), 0.0));
        ours.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
    }

    /// With neither variant, the conv's own forward.
    #[test]
    fn test_no_variant_is_the_conv() {
        let device = cpu_device();
        let conv = Conv2dConfig::new([2, 3], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(&device);
        let x = Tensor::<4>::random([1, 2, 5, 5], Distribution::Default, &device);
        conv2d_variant_forward(&conv, x.clone(), None, false)
            .into_data()
            .assert_eq(&conv.forward(x).into_data(), true);
    }

    /// Both: the standardized conv over the SAME-padded input.
    #[test]
    fn test_std_conv_same() {
        let device = cpu_device();
        let conv = Conv2dConfig::new([2, 3], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(false)
            .init(&device);
        let x = Tensor::<4>::random([1, 2, 12, 12], Distribution::Default, &device);
        let ours = conv2d_variant_forward(&conv, x.clone(), Some(1e-8), true);
        assert_eq!(ours.dims(), [1, 3, 6, 6]);
        // `get_same_padding(12, 7, 2, 1)` is 5: 2 before, 3 after.
        let expected = weight_standardized_conv2d(&conv, x.pad((2, 3, 2, 3), 0.0), 1e-8);
        ours.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
    }
}
