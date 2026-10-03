//! # The `ResNet` Downsample Implementation.

use burn::{
    nn::{
        BatchNormConfig,
        conv::Conv2dConfig,
        norm::NormalizationConfig,
    },
    prelude::Config,
};

use crate::{
    blocks::conv::ConvBlock2dConfig,
    ops::conv::{
        build_square_conv2d_padding_config,
        expect_conv_output_shape,
        get_square_conv2d_padding,
    },
    support::arrays::scalar_to_array,
};

/// [`ResNetDownsampleConfig`] Meta trait.
///
/// Implemented by:
/// * [`ResNetDownsampleConfig`]
///
/// # Missing Features
///
/// - *avg*: support for average pooling is blocked on support for ``ceil_mode``
///   in [`burn`].
pub trait ResNetDownsampleMeta {
    /// The size of the in channels dimension.
    fn in_channels(&self) -> usize;

    /// The size of the out channels dimension.
    fn out_channels(&self) -> usize;

    /// The kernel size of conv.
    fn kernel_size(&self) -> usize;

    /// The dilation of the conv.
    fn dilation(&self) -> usize;

    /// The stride of the downsample layer.
    fn stride(&self) -> usize;

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
    fn output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> [usize; 2] {
        expect_conv_output_shape(
            input_resolution,
            scalar_to_array(self.kernel_size()),
            scalar_to_array(self.stride()),
            scalar_to_array(get_square_conv2d_padding(
                self.kernel_size(),
                self.stride(),
                self.dilation(),
            )),
            scalar_to_array(self.dilation()),
        )
    }
}

/// `ResNet` Downsample configuration.
///
/// Implements [`ResNetDownsampleMeta`].
///
/// # Missing Features
///
/// - *avg*: support for average pooling is blocked on support for ``ceil_mode``
///   in [`burn`].
#[derive(Config, Debug)]
pub struct ResNetDownsampleConfig {
    /// The size of the in channels dimension.
    in_channels: usize,

    /// The size of the out channels dimension.
    out_channels: usize,

    /// The kernel size of conv.
    kernel_size: usize,

    /// The stride of the conv.
    #[config(default = 1)]
    stride: usize,

    /// The dilation of the conv.
    #[config(default = 1)]
    dilation: usize,

    /// The [`Normalization`] config.
    ///
    /// The feature size will be auto-matched.
    #[config(default = "NormalizationConfig::Batch(BatchNormConfig::new(0))")]
    norm: NormalizationConfig,
}

impl ResNetDownsampleMeta for ResNetDownsampleConfig {
    fn in_channels(&self) -> usize {
        self.in_channels
    }

    fn out_channels(&self) -> usize {
        self.out_channels
    }

    fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    fn dilation(&self) -> usize {
        self.dilation
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

impl ResNetDownsampleConfig {
    /// Builds the downsample [`Conv2dConfig`].
    ///
    /// A `stride == 1`, `dilation == 1` downsample collapses to a `1x1` conv;
    /// dilation only applies when the kernel is larger than `1`.
    fn build_conv(&self) -> Conv2dConfig {
        let kernel_size = if self.stride == 1 && self.dilation == 1 {
            1
        } else {
            self.kernel_size
        };
        let dilation = if kernel_size > 1 { self.dilation } else { 1 };
        let padding = build_square_conv2d_padding_config(kernel_size, self.stride, dilation);

        Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            scalar_to_array(kernel_size),
        )
        .with_stride(scalar_to_array(self.stride))
        .with_padding(padding)
        .with_dilation(scalar_to_array(dilation))
        .with_bias(false)
    }

    /// Lifts this downsample into an equivalent [`ConvBlock2dConfig`].
    pub fn to_conv_block_config(&self) -> ConvBlock2dConfig {
        ConvBlock2dConfig::new(self.build_conv())
            .with_norm(Some(self.norm.clone()))
            .with_act(None)
            .match_norm_features()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_config() {
        let config = ResNetDownsampleConfig::new(2, 4, 3);
        assert_eq!(config.in_channels(), 2);
        assert_eq!(config.out_channels(), 4);
        assert_eq!(config.kernel_size(), 3);
        assert_eq!(config.stride(), 1);
        assert_eq!(config.dilation(), 1);
        assert_eq!(config.output_resolution([8, 8]), [8, 8]);

        let config = config.with_stride(2);
        assert_eq!(config.stride(), 2);
        assert_eq!(config.output_resolution([8, 8]), [4, 4]);

        let conv_block = config.to_conv_block_config();
        assert_eq!(conv_block.conv.kernel_size, [3, 3]);
        assert_eq!(conv_block.conv.stride, [2, 2]);
        assert_eq!(conv_block.conv.dilation, [1, 1]);
        assert!(conv_block.act.is_none());

        match &conv_block.norm {
            Some(NormalizationConfig::Batch(bc)) => {
                assert_eq!(bc.num_features, 4);
            }
            _ => panic!("Expected BatchNormConfig"),
        }
    }
}
