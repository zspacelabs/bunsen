//! `ConvBlock1d`: a conv/norm/activation block.

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::{
        activation::{
            BunsenActivation,
            BunsenActivationConfig,
        },
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
    },
    ops::conv::try_conv1d_output_size,
};

/// [`ConvBlock1d`] Meta.
pub trait ConvBlock1dMeta {
    /// Number of input channels.
    fn in_channels(&self) -> usize;

    /// Number of output channels.
    fn out_channels(&self) -> usize;

    /// Number of groups.
    fn groups(&self) -> usize;

    /// Returns the stride.
    fn stride(&self) -> [usize; 1];

    /// Returns the kernel size.
    fn kernel_size(&self) -> usize;

    /// Returns the dilation.
    fn dilation(&self) -> usize;

    /// Returns the padding configuration.
    fn padding(&self) -> PaddingConfig1d;

    /// Predicts the output length for a given input length.
    ///
    /// Computes the true 1D convolution output length, factoring in the kernel
    /// size, padding, dilation, and stride:
    ///
    /// ```text
    /// out = floor((in + total_padding - dilation*(kernel_size - 1) - 1) / stride) + 1
    /// ```
    ///
    /// # Arguments
    ///
    /// * `in_length` - The input length.
    ///
    /// # Returns
    ///
    /// The predicted output length.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if there is no
    /// legal output length (the kernel does not fit the padded input).
    /// The message names the input length and the convolution geometry.
    fn try_output_length(
        &self,
        in_length: usize,
    ) -> BunsenResult<usize> {
        let stride = self.stride()[0];
        let kernel_size = self.kernel_size();
        let total_padding = match self.padding() {
            PaddingConfig1d::Valid => 0,
            PaddingConfig1d::Explicit(left, right) => left + right,
            // Matches burn's same-padding (`calculate_same_padding`), which
            // targets `out = ceil(in / stride)` and ignores dilation.
            PaddingConfig1d::Same => {
                let out = in_length.div_ceil(stride);
                (out.saturating_sub(1) * stride + kernel_size).saturating_sub(in_length)
            }
        };
        // Fold the (possibly asymmetric) total padding into the effective input
        // length so we can reuse the symmetric `try_conv1d_output_size`.
        try_conv1d_output_size(
            in_length + total_padding,
            kernel_size,
            stride,
            0,
            self.dilation(),
        )
        .map_err(|_| {
            ConstraintError::custom(
                "ConvBlock1d",
                "in_length",
                format!(
                    "{in_length} has no legal output length with kernel_size {kernel_size}, \
                     stride {stride}, dilation {}, padding {:?}",
                    self.dilation(),
                    self.padding(),
                ),
            )
            .into()
        })
    }
}

/// [`ConvBlock1d`] Config.
///
/// Implements [`ConvBlock1dMeta`].
///
/// Auto-matches the norm layer input channels
/// to the conv layer's output channels.
#[derive(Config, Debug)]
pub struct ConvBlock1dConfig {
    /// The [`Conv1d`] config.
    pub conv: Conv1dConfig,

    /// The [`BunsenNormalization`] config.
    pub norm: Option<BunsenNormalizationConfig>,

    /// The [`BunsenActivation`] config.
    #[config(default = "Some(BunsenActivationConfig::Relu)")]
    pub act: Option<BunsenActivationConfig>,
}

impl ConvBlock1dMeta for ConvBlock1dConfig {
    fn in_channels(&self) -> usize {
        self.conv.channels_in
    }

    fn out_channels(&self) -> usize {
        self.conv.channels_out
    }

    fn groups(&self) -> usize {
        self.conv.groups
    }

    fn stride(&self) -> [usize; 1] {
        [self.conv.stride]
    }

    fn kernel_size(&self) -> usize {
        self.conv.kernel_size
    }

    fn dilation(&self) -> usize {
        self.conv.dilation
    }

    fn padding(&self) -> PaddingConfig1d {
        self.conv.padding.clone()
    }
}

impl ConvBlock1dConfig {
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
impl ModuleInit<ConvBlock1d> for ConvBlock1dConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvBlock1d> {
        let out_channels = self.out_channels();
        Ok(ConvBlock1d {
            conv: self.conv.init(device),
            norm: self
                .norm
                .as_ref()
                .map(|config| {
                    config
                        .clone()
                        .with_num_features(out_channels)
                        .try_init(device)
                })
                .transpose()?,
            act: self
                .act
                .as_ref()
                .map(|config| config.try_init(device))
                .transpose()?,
        })
    }
}

/// Sequenced conv/norm/activation block.
///
/// A [`ConvBlock1d`] module is:
/// * a [`Conv1d`] layer,
/// * an optional [`BunsenNormalization`] layer,
/// * an optional [`BunsenActivation`] layer.
///
/// With support for hooking the forward method,
/// to run code between the norm and activation layers.
///
/// Implements [`ConvBlock1dMeta`].
///
/// Built by [`ConvBlock1dConfig`].
#[derive(Module, Debug)]
pub struct ConvBlock1d {
    /// Internal Conv1d layer.
    pub conv: Conv1d,

    /// Internal Norm Layer.
    pub norm: Option<BunsenNormalization>,

    /// Activation layer.
    pub act: Option<BunsenActivation>,
}

impl ConvBlock1dMeta for ConvBlock1d {
    fn in_channels(&self) -> usize {
        self.conv.weight.dims()[1] * self.groups()
    }

    fn out_channels(&self) -> usize {
        self.conv.weight.dims()[0]
    }

    fn groups(&self) -> usize {
        self.conv.groups
    }

    fn stride(&self) -> [usize; 1] {
        [self.conv.stride]
    }

    fn kernel_size(&self) -> usize {
        self.conv.kernel_size
    }

    fn dilation(&self) -> usize {
        self.conv.dilation
    }

    fn padding(&self) -> PaddingConfig1d {
        self.conv.padding.clone()
    }
}

impl ConvBlock1d {
    /// Forward Pass.
    ///
    /// Applies the conv/norm/act layers in sequence:
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
    /// - `input`: `[batch, in_channels, in_length]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_length]`, where `out_length` is predicted by
    /// [`ConvBlock1dMeta::try_output_length`].
    pub fn forward(
        &self,
        input: Tensor<3>,
    ) -> Tensor<3> {
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
    /// let x = f(x);
    /// let x = match &self.act {
    ///     Some(a) => a.forward(x),
    ///     None => x,
    /// };
    /// return x
    /// ```
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_length]`.
    /// - `f`: a callback endofunction over the normalized conv output, from and
    ///   to `[batch, out_channels, out_length]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_length]`, where `out_length` is predicted by
    /// [`ConvBlock1dMeta::try_output_length`].
    pub fn map_forward<F>(
        &self,
        input: Tensor<3>,
        f: F,
    ) -> Tensor<3>
    where
        F: FnOnce(Tensor<3>) -> Tensor<3>,
    {
        let [batch, in_length] = unpack_shape_contract!(
            ["batch", "in_channels", "in_length"],
            &input.dims(),
            &["batch", "in_length"],
            &[("in_channels", self.in_channels())]
        );
        // True conv arithmetic; factors in kernel size, padding, and dilation.
        #[cfg(any(debug_assertions, test))]
        use crate::errors::WithOkOrPanic;
        #[cfg(any(debug_assertions, test))]
        let out_length = self.try_output_length(in_length).ok_or_panic();
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (batch, in_length);
        let x = self.conv.forward(input);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_length"],
            &x.dims(),
            &[
                ("batch", batch),
                ("out_channels", self.out_channels()),
                ("out_length", out_length)
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

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_length"],
            &x.dims(),
            &[
                ("batch", batch),
                ("out_channels", self.out_channels()),
                ("out_length", out_length)
            ]
        );

        x
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::{
            BatchNormConfig,
            PaddingConfig1d,
        },
        tensor::Distribution,
    };

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    #[test]
    fn test_output_length() {
        let block = |conv: Conv1dConfig| ConvBlock1dConfig::new(conv);

        // kernel=3, stride=1, dilation=2, "same" padding -> length preserved.
        let same = block(
            Conv1dConfig::new(2, 4, 3)
                .with_stride(1)
                .with_dilation(2)
                .with_padding(PaddingConfig1d::Explicit(2, 2)),
        );
        assert_eq!(same.try_output_length(10).unwrap(), 10);

        // Valid padding shrinks by `dilation * (kernel - 1)` = 4.
        let valid_dilated = block(
            Conv1dConfig::new(2, 4, 3)
                .with_dilation(2)
                .with_padding(PaddingConfig1d::Valid),
        );
        assert_eq!(valid_dilated.try_output_length(10).unwrap(), 6);

        // With dilation=1, valid padding only shrinks by `kernel - 1` = 2.
        let valid = block(Conv1dConfig::new(2, 4, 3).with_padding(PaddingConfig1d::Valid));
        assert_eq!(valid.try_output_length(10).unwrap(), 8);

        // Stride downsamples: kernel=3, stride=2, "same" -> ceil(10/2) = 5.
        let strided = block(
            Conv1dConfig::new(2, 4, 3)
                .with_stride(2)
                .with_padding(PaddingConfig1d::Explicit(1, 1)),
        );
        assert_eq!(strided.try_output_length(10).unwrap(), 5);

        // No legal output when the kernel cannot fit.
        let too_big = block(Conv1dConfig::new(2, 4, 5).with_padding(PaddingConfig1d::Valid));
        ErrorMatcher::illegal()
            .constraint("ConvBlock1d", "in_length")
            .message_contains("3 has no legal output length")
            .assert_err(&too_big.try_output_length(3));
    }

    #[test]
    fn test_dilated_forward_shape() {
        let device = cpu_device().autodiff();

        // Dilated, valid-padded block: previously incompatible with the
        // stride-division contract; now modeled by true conv arithmetic.
        let config = ConvBlock1dConfig::new(
            Conv1dConfig::new(2, 4, 3)
                .with_dilation(2)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(false),
        )
        .with_norm(None)
        .with_act(None);

        let layer: ConvBlock1d = config.init(&device);

        let input = Tensor::random([2, 2, 10], Distribution::Default, &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [2, 4, layer.try_output_length(10).unwrap()]);
        assert_eq!(output.dims(), [2, 4, 6]);
    }

    #[test]
    fn test_cb() {
        let device = cpu_device().autodiff();

        let config = ConvBlock1dConfig::new(
            Conv1dConfig::new(2, 4, 3)
                .with_stride(2)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .with_bias(false),
        )
        .with_norm(Some(BunsenNormalizationConfig::Batch(
            BatchNormConfig::new(0),
        )))
        .with_act(Some(BunsenActivationConfig::Relu));

        let layer: ConvBlock1d = config.init(&device);
        assert_eq!(layer.in_channels(), 2);
        assert_eq!(layer.out_channels(), 4);
        assert_eq!(layer.groups(), 1);
        assert_eq!(layer.stride(), [2]);

        let batch_size = 2;
        let length = 10;
        let channels = 2;

        let input = Tensor::random(
            [batch_size, channels, length],
            Distribution::Default,
            &device,
        );

        {
            let output = layer.forward(input.clone());
            let expected = {
                let x = layer.conv.forward(input.clone());
                let x = layer.norm.as_ref().unwrap().forward(x);
                layer.act.as_ref().unwrap().forward(x)
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
                layer.act.as_ref().unwrap().forward(x)
            };
            output.to_data().assert_eq(&expected.to_data(), true);
        }
    }
}
