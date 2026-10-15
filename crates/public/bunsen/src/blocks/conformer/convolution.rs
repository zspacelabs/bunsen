//! The Conformer's convolution module.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        BatchNormConfig,
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
        norm::NormalizationConfig,
    },
    prelude::Bool,
    tensor::{
        Device,
        activation::{
            glu,
            silu,
        },
    },
};

use crate::{
    blocks::conv::{
        ConvBlock1d,
        ConvBlock1dConfig,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`ConformerConvolution`] and
/// [`ConformerConvolutionConfig`].
pub trait ConformerConvolutionMeta {
    /// The channel width.
    fn d_model(&self) -> usize;

    /// The depthwise convolution's kernel size; odd.
    fn kernel_size(&self) -> usize;
}

/// Config for [`ConformerConvolution`].
///
/// Implements [`ConformerConvolutionMeta`].
#[derive(Config, Debug)]
pub struct ConformerConvolutionConfig {
    /// The channel width.
    pub d_model: usize,

    /// The depthwise convolution's kernel size; odd, so the output keeps the
    /// input's length.
    pub kernel_size: usize,

    /// Whether the three convolutions carry a bias.
    #[config(default = "true")]
    pub bias: bool,
}

impl ConformerConvolutionMeta for ConformerConvolutionConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

impl ModuleInit<ConformerConvolution> for ConformerConvolutionConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConformerConvolution> {
        let c = self.d_model;
        let pad = (self.kernel_size - 1) / 2;
        Ok(ConformerConvolution {
            pointwise_conv1: Conv1dConfig::new(c, 2 * c, 1)
                .with_bias(self.bias)
                .init(device),
            depthwise: ConvBlock1dConfig::new(
                Conv1dConfig::new(c, c, self.kernel_size)
                    .with_groups(c)
                    .with_padding(PaddingConfig1d::Explicit(pad, pad))
                    .with_bias(self.bias),
            )
            .with_norm(Some(NormalizationConfig::Batch(BatchNormConfig::new(c))))
            .with_act(None)
            .try_init(device)?,
            pointwise_conv2: Conv1dConfig::new(c, c, 1).with_bias(self.bias).init(device),
        })
    }
}

/// The Conformer's convolution module.
///
/// Pointwise convolution to twice the width, GLU over the channels, a
/// depthwise convolution with batch normalization, `SiLU`, and a pointwise
/// convolution back: Gulati et al. (2020), figure 2.
///
/// Built by [`ConformerConvolutionConfig`].
/// Implements [`ConformerConvolutionMeta`].
#[derive(Module, Debug)]
pub struct ConformerConvolution {
    /// `d_model -> 2 * d_model`, kernel 1.
    pub pointwise_conv1: Conv1d,

    /// The depthwise convolution and its batch normalization.
    pub depthwise: ConvBlock1d,

    /// `d_model -> d_model`, kernel 1.
    pub pointwise_conv2: Conv1d,
}

impl ConformerConvolutionMeta for ConformerConvolution {
    fn d_model(&self) -> usize {
        self.pointwise_conv2.weight.dims()[0]
    }

    fn kernel_size(&self) -> usize {
        self.depthwise.conv.kernel_size
    }
}

impl ConformerConvolution {
    /// Applies the module.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    /// * `valid` - `[batch, time]`, true at the frames that are not padding.
    ///   Padding frames are zeroed before the depthwise convolution, so they do
    ///   not leak into the frames beside them. `None` means none is.
    ///
    /// # Returns
    /// `[batch, time, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        valid: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let x = glu(self.pointwise_conv1.forward(x.swap_dims(1, 2)), 1);
        let x = match valid {
            Some(valid) => x.mask_fill(valid.bool_not().unsqueeze_dim::<3>(1), 0.0),
            None => x,
        };
        let x = self.depthwise.map_forward(x, silu);
        self.pointwise_conv2.forward(x).swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = ConformerConvolutionConfig::new(6, 5).with_bias(false);
        let module: ConformerConvolution = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.kernel_size(), config.kernel_size());

        let y = module.forward(Tensor::zeros([2, 7, 6], &cpu_device()), None);
        assert_eq!(y.dims(), [2, 7, 6]);
    }
}
