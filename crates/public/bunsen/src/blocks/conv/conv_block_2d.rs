//! # `ConvBlock2d` - conv/norm/activation block.
//!
//! A [`ConvBlock2d`] module is:
//! * a [`Conv2d`] layer,
//! * an optional [`Normalization`] layer,
//! * an optional [`Activation`] layer.
//!
//! With support for hooking the forward method,
//! to run code between the norm and application images.

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        activation::{
            Activation,
            ActivationConfig,
        },
        conv::{
            Conv2d,
            Conv2dConfig,
        },
        norm::{
            Normalization,
            NormalizationConfig,
        },
    },
    prelude::{
        Backend,
        Tensor,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::conv::maybe_conv_output_shape,
};

/// Abstract policy for [`ConvBlock2d`] Config.
///
/// Defines a [`NormalizationConfig`] and [`ActivationConfig`],
/// and can be lifted to a [`ConvBlock2dConfig`] to match a [`Conv2dConfig`].
///
/// The abstract [`NormalizationConfig`] will be feature matched
/// with the target [`Conv2dConfig`].
#[derive(Config, Debug)]
pub struct AbstractConvBlock2dConfig {
    /// The [`Normalization`] config.
    pub norm: Option<NormalizationConfig>,

    /// Activation Config.
    #[config(default = "Some(ActivationConfig::Relu)")]
    pub act: Option<ActivationConfig>,
}

impl AbstractConvBlock2dConfig {
    /// Merges with a [`Conv2dConfig`] to construct a [`ConvBlock2dConfig`].
    ///
    /// The abstract [`NormalizationConfig`] will be feature matched
    /// with the target [`Conv2dConfig`], resulting in a normalization
    /// layer sized appropriately for the input convolution.
    pub fn build_config(
        &self,
        conv: Conv2dConfig,
    ) -> ConvBlock2dConfig {
        ConvBlock2dConfig {
            conv,
            norm: self.norm.clone(),
            act: self.act.clone(),
        }
        .match_norm_features()
    }
}

/// [`ConvBlock2d`] Meta.
pub trait ConvBlock2dMeta {
    /// Number of input channels.
    fn in_channels(&self) -> usize;

    /// Number of output channels.
    fn out_channels(&self) -> usize;

    /// Number of groups.
    fn groups(&self) -> usize;

    /// Returns the stride; `[height, width]`.
    fn stride(&self) -> [usize; 2];

    /// Returns the kernel size; `[height, width]`.
    fn kernel_size(&self) -> [usize; 2];

    /// Returns the dilation; `[height, width]`.
    fn dilation(&self) -> [usize; 2];

    /// Returns the padding configuration.
    fn padding(&self) -> PaddingConfig2d;

    /// Predicts the output resolution for a given input resolution.
    ///
    /// Computes the true 2D convolution output resolution, factoring in the
    /// kernel size, padding, dilation, and stride, independently per spatial
    /// dimension:
    ///
    /// ```text
    /// out = floor((in + total_padding - dilation*(kernel_size - 1) - 1) / stride) + 1
    /// ```
    ///
    /// # Arguments
    ///
    /// * `input_resolution` - The input resolution `[in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// The predicted output resolution `[out_height, out_width]`.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if there is no legal output resolution (the
    /// kernel does not fit the padded input).
    fn try_output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> BunsenResult<[usize; 2]> {
        let stride = self.stride();
        let kernel_size = self.kernel_size();
        let dilation = self.dilation();
        let total_padding = match self.padding() {
            PaddingConfig2d::Valid => [0, 0],
            // `Explicit` is `(top, left, bottom, right)`.
            PaddingConfig2d::Explicit(top, left, bottom, right) => [top + bottom, left + right],
            // Matches burn's same-padding (`calculate_same_padding`), which
            // targets `out = ceil(in / stride)` and ignores dilation.
            PaddingConfig2d::Same => {
                let mut pads = [0; 2];
                for d in 0..2 {
                    let out = input_resolution[d].div_ceil(stride[d]);
                    pads[d] = (out.saturating_sub(1) * stride[d] + kernel_size[d])
                        .saturating_sub(input_resolution[d]);
                }
                pads
            }
        };
        // Fold the (possibly asymmetric) total padding into the effective input
        // resolution so we can reuse the symmetric `maybe_conv_output_shape`.
        let effective = [
            input_resolution[0] + total_padding[0],
            input_resolution[1] + total_padding[1],
        ];
        maybe_conv_output_shape(effective, kernel_size, stride, [0, 0], dilation).ok_or_else(|| {
            BunsenError::Invalid(format!(
                "ConvBlock2d has no legal output resolution for input resolution \
                 ({input_resolution:?})"
            ))
        })
    }
}

/// [`ConvBlock2d`] Config.
///
/// Implements [`ConvBlock2dMeta`].
///
/// Auto-matches the norm layer input channels
/// to the conv layer's output channels.
#[derive(Config, Debug)]
pub struct ConvBlock2dConfig {
    /// The [`Conv2d`] config.
    pub conv: Conv2dConfig,

    /// The [`Normalization`] config.
    pub norm: Option<NormalizationConfig>,

    /// The [`Activation`] config.
    #[config(default = "Some(ActivationConfig::Relu)")]
    pub act: Option<ActivationConfig>,
}

impl ConvBlock2dMeta for ConvBlock2dConfig {
    fn in_channels(&self) -> usize {
        self.conv.channels[0]
    }

    fn out_channels(&self) -> usize {
        self.conv.channels[1]
    }

    fn groups(&self) -> usize {
        self.conv.groups
    }

    fn stride(&self) -> [usize; 2] {
        self.conv.stride
    }

    fn kernel_size(&self) -> [usize; 2] {
        self.conv.kernel_size
    }

    fn dilation(&self) -> [usize; 2] {
        self.conv.dilation
    }

    fn padding(&self) -> PaddingConfig2d {
        self.conv.padding.clone()
    }
}

impl ConvBlock2dConfig {
    /// Adjust the norm features to match the conv output size.
    ///
    /// [`Self::init`] does this automatically.
    pub fn match_norm_features(self) -> Self {
        let features = self.out_channels();
        let norm = self.norm.map(|config| config.with_num_features(features));
        Self { norm, ..self }
    }
}

/// Auto-matches the norm layer input channels
/// to the conv layer's output channels.
impl<B: Backend> ModuleInit<B, ConvBlock2d<B>> for ConvBlock2dConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<ConvBlock2d<B>> {
        let out_channels = self.out_channels();
        Ok(ConvBlock2d {
            conv: self.conv.init(device),
            norm: self
                .norm
                .as_ref()
                .map(|config| config.clone().with_num_features(out_channels).init(device)),
            act: self.act.as_ref().map(|config| config.init(device)),
        })
    }
}

/// Sequenced conv/norm/activation block.
///
/// A [`ConvBlock2d`] module is:
/// * a [`Conv2d`] layer,
/// * an optional [`Normalization`] layer,
/// * an optional [`Activation`] layer.
///
/// With support for hooking the forward method,
/// to run code between the norm and application images.
///
/// Implements [`ConvBlock2dMeta`].
///
/// Built by [`ConvBlock2dConfig`].
#[derive(Module, Debug)]
pub struct ConvBlock2d<B: Backend> {
    /// Internal Conv2d layer.
    pub conv: Conv2d<B>,

    /// Internal Norm Layer.
    pub norm: Option<Normalization<B>>,

    /// Activation layer.
    pub act: Option<Activation<B>>,
}

impl<B: Backend> ConvBlock2dMeta for ConvBlock2d<B> {
    fn in_channels(&self) -> usize {
        self.conv.weight.dims()[1] * self.groups()
    }

    fn out_channels(&self) -> usize {
        self.conv.weight.dims()[0]
    }

    fn groups(&self) -> usize {
        self.conv.groups
    }

    fn stride(&self) -> [usize; 2] {
        self.conv.stride
    }

    fn kernel_size(&self) -> [usize; 2] {
        self.conv.kernel_size
    }

    fn dilation(&self) -> [usize; 2] {
        self.conv.dilation
    }

    fn padding(&self) -> PaddingConfig2d {
        self.conv.padding.clone()
    }
}

impl<B: Backend> ConvBlock2d<B> {
    /// Forward Pass.
    ///
    /// Applies the conv/norm/act images in sequence:
    ///
    /// ```rust,ignore
    /// let x = self.conv.forward(input);
    /// let x = match &self.norm {
    ///     Some(n) => n.forward(x),
    ///     None => x,
    /// };
    /// let x = match &self.act {
    ///     Some(a) => a.forward(x),
    ///     None => x,
    /// };
    /// return x
    /// ```
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`, where the output
    /// resolution is predicted by [`ConvBlock2dMeta::try_output_resolution`].
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.map_forward(input, |x| x)
    }

    /// Mapping Forward Pass.
    ///
    /// Applies the callback fn after normalization but before activation.
    ///
    /// ```rust,ignore
    /// let x = self.conv.forward(input);
    /// let x = match &self.norm {
    ///     Some(n) => n.forward(x),
    ///     None => x,
    /// };
    /// let x = self.norm.forward(x);
    /// let x = match &self.act {
    ///     Some(a) => a.forward(x),
    ///     None => x,
    /// };
    /// return x
    /// ```
    ///
    /// # Arguments
    ///
    /// - `input`: \ `[batch, in_channels, in_height, in_width]`.
    /// - `f`: a callback endofunction, from/to `[batch, in_channels,
    ///   out_height, out_width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`, where the output
    /// resolution is predicted by [`ConvBlock2dMeta::try_output_resolution`].
    pub fn map_forward<F>(
        &self,
        input: Tensor<B, 4>,
        f: F,
    ) -> Tensor<B, 4>
    where
        F: FnOnce(Tensor<B, 4>) -> Tensor<B, 4>,
    {
        #[cfg(debug_assertions)]
        use crate::{
            contracts::{
                assert_shape_contract_periodically,
                unpack_shape_contract,
            },
            errors::WithOkOrPanic,
        };
        #[cfg(debug_assertions)]
        let [batch, in_height, in_width] = unpack_shape_contract!(
            ["batch", "in_channels", "in_height", "in_width"],
            &input.dims(),
            &["batch", "in_height", "in_width"],
            &[("in_channels", self.in_channels())]
        );
        // True conv arithmetic; factors in kernel size, padding, and dilation.
        #[cfg(debug_assertions)]
        let [out_height, out_width] = self
            .try_output_resolution([in_height, in_width])
            .ok_or_panic();
        let x = self.conv.forward(input);

        #[cfg(debug_assertions)]
        assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_height", "out_width"],
            &x.dims(),
            &[
                ("batch", batch),
                ("out_channels", self.out_channels()),
                ("out_height", out_height),
                ("out_width", out_width)
            ]
        );

        let x = match &self.norm {
            Some(norm) => norm.forward(x),
            None => x,
        };

        let x = f(x);

        let x = match &self.act {
            Some(act) => act.forward(x),
            None => x,
        };

        #[cfg(debug_assertions)]
        assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_height", "out_width"],
            &x.dims(),
            &[
                ("batch", batch),
                ("out_channels", self.out_channels()),
                ("out_height", out_height),
                ("out_width", out_width)
            ]
        );

        x
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Autodiff,
        nn::{
            BatchNormConfig,
            PaddingConfig2d,
            activation::ActivationConfig,
            norm::NormalizationConfig,
        },
        tensor::Distribution,
    };

    use super::*;
    use crate::support::testing::CpuBackend;

    #[test]
    fn test_conv_norm_config() {
        let abstract_config = AbstractConvBlock2dConfig::new()
            .with_norm(Some(NormalizationConfig::Batch(BatchNormConfig::new(0))));

        let conv_config = Conv2dConfig::new([2, 4], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false);

        let config: ConvBlock2dConfig = abstract_config.build_config(conv_config.clone());

        assert_eq!(config.in_channels(), 2);
        assert_eq!(config.out_channels(), 4);
        assert_eq!(config.groups(), 1);
        assert_eq!(config.stride(), [2, 2]);
    }

    #[test]
    fn test_output_resolution() {
        let block = |conv: Conv2dConfig| ConvBlock2dConfig::new(conv);

        // kernel=3, stride=1, dilation=2, "same" padding -> resolution
        // preserved.
        let same = block(
            Conv2dConfig::new([2, 4], [3, 3])
                .with_dilation([2, 2])
                .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2)),
        );
        assert_eq!(same.try_output_resolution([10, 12]).unwrap(), [10, 12]);

        // Valid padding shrinks by `dilation * (kernel - 1)` = 4 per dim.
        let valid_dilated = block(
            Conv2dConfig::new([2, 4], [3, 3])
                .with_dilation([2, 2])
                .with_padding(PaddingConfig2d::Valid),
        );
        assert_eq!(
            valid_dilated.try_output_resolution([10, 12]).unwrap(),
            [6, 8]
        );

        // With dilation=1, valid padding only shrinks by `kernel - 1` = 2.
        let valid = block(Conv2dConfig::new([2, 4], [3, 3]).with_padding(PaddingConfig2d::Valid));
        assert_eq!(valid.try_output_resolution([10, 12]).unwrap(), [8, 10]);

        // Stride downsamples: kernel=3, stride=2, "same" -> ceil(in/2) per dim.
        let strided = block(
            Conv2dConfig::new([2, 4], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
        );
        assert_eq!(strided.try_output_resolution([10, 12]).unwrap(), [5, 6]);

        // No legal output when the kernel cannot fit.
        let too_big = block(Conv2dConfig::new([2, 4], [5, 5]).with_padding(PaddingConfig2d::Valid));
        assert!(matches!(
            too_big.try_output_resolution([3, 3]),
            Err(BunsenError::Invalid(_))
        ));
    }

    #[test]
    fn test_dilated_forward_shape() {
        type I = CpuBackend;
        type B = Autodiff<I>;
        let device = Default::default();

        // Dilated, valid-padded block: previously incompatible with the
        // stride-division contract; now modeled by true conv arithmetic.
        let config = ConvBlock2dConfig::new(
            Conv2dConfig::new([2, 4], [3, 3])
                .with_dilation([2, 2])
                .with_padding(PaddingConfig2d::Valid)
                .with_bias(false),
        )
        .with_norm(None)
        .with_act(None);

        let layer: ConvBlock2d<B> = config.init(&device);

        let input = Tensor::random([2, 2, 10, 12], Distribution::Default, &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [2, 4, 6, 8]);
        assert_eq!(
            [output.dims()[2], output.dims()[3]],
            layer.try_output_resolution([10, 12]).unwrap()
        );
    }

    #[test]
    fn test_cb() {
        type I = CpuBackend;
        type B = Autodiff<I>;
        let device = Default::default();

        let config = ConvBlock2dConfig::new(
            Conv2dConfig::new([2, 4], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false),
        )
        .with_norm(Some(NormalizationConfig::Batch(BatchNormConfig::new(0))))
        .with_act(Some(ActivationConfig::Relu));

        let layer: ConvBlock2d<B> = config.init(&device);
        assert_eq!(layer.in_channels(), 2);
        assert_eq!(layer.out_channels(), 4);
        assert_eq!(layer.groups(), 1);
        assert_eq!(layer.stride(), [2, 2]);

        let batch_size = 2;
        let height = 10;
        let width = 10;
        let channels = 2;

        let input = Tensor::random(
            [batch_size, channels, height, width],
            Distribution::Default,
            &device,
        );

        {
            let output = layer.forward(input.clone());
            let expected = {
                let x = layer.conv.forward(input.clone());
                let x = layer.norm.as_ref().unwrap().forward(x);
                let x = layer.act.as_ref().unwrap().forward(x);
                x
            };
            output.to_data().assert_eq(&expected.to_data(), true);
        }

        {
            let hook = |x| x * 2.0;

            let output = layer.map_forward(input.clone(), hook);
            let expected = {
                let x = layer.conv.forward(input.clone());
                let x = layer.norm.as_ref().unwrap().forward(x);
                let x = hook(x);
                let x = layer.act.as_ref().unwrap().forward(x);
                x
            };
            output.to_data().assert_eq(&expected.to_data(), true);
        }
    }
}
