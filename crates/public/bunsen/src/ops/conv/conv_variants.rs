//! Conv variants: SAME padding and weight standardization.

use alloc::format;

use burn::{
    config::Config,
    nn::{
        PaddingConfig2d,
        conv::Conv2d,
    },
    prelude::Tensor,
};

use super::{
    get_same_padding,
    pad_same,
    weight_standardized_conv2d,
};
use crate::errors::{
    BunsenResult,
    ConstraintError,
};

/// timm's conv variants, as one value: how a conv's input is padded and
/// its weight standardized.
///
/// - `same_pad`: the input padded with zeros first, TensorFlow's "SAME" from
///   its own size ([`pad_same`]; timm's `Conv2dSame`), the conv itself unpadded
///   (`Valid`);
/// - `weight_std`: the conv's weight standardized by this `eps`
///   ([`weight_standardized_conv2d`]; timm's `StdConv2d`).
///
/// Both together are timm's `StdConv2dSame`; neither, a plain conv. A
/// config embeds it; a module holds it as `#[module(skip)]`.
#[derive(Config, Debug, Copy, PartialEq)]
pub struct Conv2dVariantOptions {
    /// Weight standardization of the conv, by this `eps`; none when `None`.
    #[config(default = "None")]
    pub weight_std: Option<f64>,

    /// Whether the input is SAME-padded from its own size before the conv,
    /// which is then unpadded (`Valid`): timm's `Conv2dSame`, the odd pixel
    /// bottom-right.
    #[config(default = false)]
    pub same_pad: bool,
}

impl Conv2dVariantOptions {
    /// Checks the variant against the conv's own `padding`: a SAME-padded
    /// conv pads its input, so its own padding is `Valid`.
    ///
    /// # Errors
    ///
    /// A [`ConstraintError`] on `owner`'s `conv.padding` otherwise.
    pub fn try_check(
        &self,
        owner: &'static str,
        padding: &PaddingConfig2d,
    ) -> BunsenResult<()> {
        if self.same_pad && !matches!(padding, PaddingConfig2d::Valid) {
            return Err(ConstraintError::custom(
                owner,
                "conv.padding",
                format!(
                    "a SAME-padded conv pads its input, so its own padding is Valid, not {padding:?}"
                ),
            )
            .into());
        }
        Ok(())
    }

    /// The total padding, per spatial axis, the variant adds to an input of
    /// `input_resolution` before a conv of this geometry; `None` when it adds
    /// none.
    pub fn same_padding(
        &self,
        input_resolution: [usize; 2],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
    ) -> Option<[usize; 2]> {
        self.same_pad.then(|| {
            [0, 1].map(|d| {
                get_same_padding(input_resolution[d], kernel_size[d], stride[d], dilation[d])
            })
        })
    }

    /// `conv`'s forward over `input` under the variant.
    ///
    /// # Arguments
    ///
    /// - `conv`: the conv; unpadded (`Valid`) under `same_pad`.
    /// - `input`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`; under `same_pad`,
    /// `ceil(height / stride)` by `ceil(width / stride)`.
    pub fn forward(
        &self,
        conv: &Conv2d,
        input: Tensor<4>,
    ) -> Tensor<4> {
        let input = match self.same_pad {
            true => pad_same(input, conv.kernel_size, conv.stride, conv.dilation, 0.0),
            false => input,
        };
        match self.weight_std {
            Some(eps) => weight_standardized_conv2d(conv, input, eps),
            None => conv.forward(input),
        }
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
        let ours = Conv2dVariantOptions::new()
            .with_weight_std(None)
            .with_same_pad(true)
            .forward(&conv, x.clone());
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
        Conv2dVariantOptions::new()
            .with_weight_std(None)
            .with_same_pad(false)
            .forward(&conv, x.clone())
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
        let ours = Conv2dVariantOptions::new()
            .with_weight_std(Some(1e-8))
            .with_same_pad(true)
            .forward(&conv, x.clone());
        assert_eq!(ours.dims(), [1, 3, 6, 6]);
        // `get_same_padding(12, 7, 2, 1)` is 5: 2 before, 3 after.
        let expected = weight_standardized_conv2d(&conv, x.pad((2, 3, 2, 3), 0.0), 1e-8);
        ours.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
    }
}
