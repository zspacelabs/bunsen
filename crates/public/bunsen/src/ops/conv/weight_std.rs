//! Weight-standardized convolution (timm's `StdConv2d`).

use burn::{
    nn::{
        PaddingConfig2d,
        conv::Conv2d,
    },
    prelude::Tensor,
    tensor::{
        module::conv2d,
        ops::ConvOptions,
    },
};

/// `conv`'s forward over `input` with its weight standardized: each output
/// channel's filter shifted to mean 0 and scaled to variance 1 (the biased
/// variance, `eps` added under the root), as timm's `StdConv2d` does it
/// (weight standardization, [arXiv:1903.10520](https://arxiv.org/abs/1903.10520);
/// `BiT`'s convs, eps 1e-8). The stored weight is unchanged.
///
/// # Arguments
///
/// - `conv`: the conv, explicitly padded (or not at all).
/// - `input`: `[batch, in_channels, height, width]`.
/// - `eps`: added to each filter's variance.
///
/// # Returns
///
/// `[batch, out_channels, out_height, out_width]`, as `conv.forward`.
///
/// # Panics
///
/// For a `Same`-padded conv, whose padding burn computes privately.
pub fn weight_standardized_conv2d(
    conv: &Conv2d,
    input: Tensor<4>,
    eps: f64,
) -> Tensor<4> {
    let weight = conv.weight.val();
    let [out_channels, in_per_group, kh, kw] = weight.dims();
    let flat = weight.reshape([out_channels, in_per_group * kh * kw]);
    let mean = flat.clone().mean_dim(1);
    let centered = flat - mean;
    let var = centered.clone().square().mean_dim(1);
    let weight = (centered / (var + eps).sqrt()).reshape([out_channels, in_per_group, kh, kw]);

    let ((top, bottom), (left, right)) = match conv.padding {
        PaddingConfig2d::Explicit(top, left, bottom, right) => ((top, bottom), (left, right)),
        PaddingConfig2d::Valid => ((0, 0), (0, 0)),
        PaddingConfig2d::Same => panic!(
            "weight_standardized_conv2d: a `Same`-padded conv; bunsen's convs pad explicitly"
        ),
    };
    conv2d(
        input,
        weight,
        conv.bias.as_ref().map(|bias| bias.val()),
        ConvOptions::new_with_padding(
            conv.stride,
            [(top, bottom), (left, right)],
            conv.dilation,
            conv.groups,
        ),
    )
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::conv::Conv2dConfig,
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::support::testing::cpu_device;

    /// A conv whose filters are already standardized convolves as itself;
    /// scaling and shifting a filter leaves the standardized conv unchanged.
    #[test]
    fn test_weight_standardization() {
        let device = cpu_device();
        let conv = Conv2dConfig::new([2, 3], [3, 3])
            .with_bias(false)
            .init(&device);
        let x = Tensor::<4>::random([1, 2, 5, 5], Distribution::Default, &device);
        let ours = weight_standardized_conv2d(&conv, x.clone(), 0.0);

        let mut shifted = conv.clone();
        shifted.weight = shifted.weight.map(|w| w * 3.0 + 0.5);
        weight_standardized_conv2d(&shifted, x.clone(), 0.0)
            .into_data()
            .assert_approx_eq::<f32>(&ours.clone().into_data(), Tolerance::rel_abs(1e-4, 1e-4));

        let mut standardized = conv.clone();
        standardized.weight = standardized.weight.map(|w| {
            let [o, i, kh, kw] = w.dims();
            let flat = w.reshape([o, i * kh * kw]);
            let c = flat.clone() - flat.mean_dim(1);
            let v = c.clone().square().mean_dim(1);
            (c / v.sqrt()).reshape([o, i, kh, kw])
        });
        standardized
            .forward(x)
            .into_data()
            .assert_approx_eq::<f32>(&ours.into_data(), Tolerance::rel_abs(1e-4, 1e-4));
    }
}
