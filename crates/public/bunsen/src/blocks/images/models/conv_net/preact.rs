//! Pre-activation: a norm and an activation before a conv, not after it.

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{
            Conv2d,
            Conv2dConfig,
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
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// A norm and an activation over a map, no conv: the config of a
/// [`NormAct`] (timm's `BatchNormAct2d`, a pre-activation unit's first norm).
#[derive(Config, Debug)]
pub struct NormActConfig {
    /// The norm; its feature count is set to `channels` at init.
    pub norm: BunsenNormalizationConfig,

    /// The activation after the norm; none when `None`.
    #[config(default = "None")]
    pub act: Option<BunsenActivationConfig>,
}

impl NormActConfig {
    /// The norm and activation over `channels`.
    ///
    /// # Errors
    ///
    /// As the norm's `try_init`.
    pub fn try_init(
        &self,
        channels: usize,
        device: &Device,
    ) -> BunsenResult<NormAct> {
        Ok(NormAct {
            norm: self
                .norm
                .clone()
                .with_num_features(channels)
                .try_init(device)?,
            act: self.act.as_ref().map(|a| a.init(device)),
        })
    }
}

/// `act(norm(x))`; built by [`NormActConfig`].
#[derive(Module, Debug)]
pub struct NormAct {
    /// The norm.
    pub norm: BunsenNormalization,

    /// The activation after it.
    pub act: Option<BunsenActivation>,
}

impl NormAct {
    /// The map, normalized and activated.
    ///
    /// # Arguments
    ///
    /// - `x`: `[batch, channels, height, width]`.
    ///
    /// # Returns
    ///
    /// The same shape.
    pub fn forward(
        &self,
        x: Tensor<4>,
    ) -> Tensor<4> {
        let x = self.norm.forward(x);
        match &self.act {
            Some(act) => act.forward(x),
            None => x,
        }
    }
}

/// A pre-activation conv: an optional norm and activation over its input,
/// then the conv (timm's `norm_act_layer` and `create_conv2d`, a
/// pre-activation bottleneck's `norm{k}` and `conv{k}`). The config of a
/// [`PreActConv`].
#[derive(Config, Debug)]
pub struct PreActConvConfig {
    /// The conv.
    pub conv: Conv2dConfig,

    /// The norm before it, over its input; none when `None`. Its feature
    /// count is set to the conv's input channels at init.
    #[config(default = "None")]
    pub norm: Option<BunsenNormalizationConfig>,

    /// The activation after the norm; none when `None`.
    #[config(default = "None")]
    pub act: Option<BunsenActivationConfig>,

    /// Weight standardization of the conv, by this `eps`
    /// ([`weight_standardized_conv2d`](crate::ops::conv::weight_standardized_conv2d));
    /// none when `None`.
    #[config(default = "None")]
    pub weight_std: Option<f64>,

    /// Whether the conv's input is SAME-padded from its own size first, the
    /// conv then unpadded (`Valid`; timm's `Conv2dSame`).
    #[config(default = false)]
    pub same_pad: bool,
}

impl PreActConvConfig {
    /// The conv's input channels.
    pub fn in_channels(&self) -> usize {
        self.conv.channels[0]
    }

    /// The conv's output channels.
    pub fn out_channels(&self) -> usize {
        self.conv.channels[1]
    }

    /// The conv's stride.
    pub fn stride(&self) -> usize {
        self.conv.stride[0]
    }
}

impl ModuleInit<PreActConv> for PreActConvConfig {
    /// Builds the conv and its norm.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a SAME-padded
    /// conv with padding of its own; as the norm's `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<PreActConv> {
        if self.same_pad && !matches!(self.conv.padding, PaddingConfig2d::Valid) {
            return Err(BunsenError::illegal(format!(
                "PreActConvConfig: a SAME-padded conv pads its input, so its own padding is \
                 Valid, not {:?}",
                self.conv.padding
            )));
        }
        Ok(PreActConv {
            norm: self
                .norm
                .clone()
                .map(|n| n.with_num_features(self.in_channels()).try_init(device))
                .transpose()?,
            act: self.act.as_ref().map(|a| a.init(device)),
            conv: self.conv.init(device),
            weight_std: self.weight_std,
            same_pad: self.same_pad,
        })
    }
}

/// `conv(act(norm(x)))`; built by [`PreActConvConfig`].
#[derive(Module, Debug)]
pub struct PreActConv {
    /// The norm over the input.
    pub norm: Option<BunsenNormalization>,

    /// The activation after the norm.
    pub act: Option<BunsenActivation>,

    /// The conv.
    pub conv: Conv2d,

    /// The conv's weight standardization `eps`, if any.
    #[module(skip)]
    pub weight_std: Option<f64>,

    /// Whether the conv's input is SAME-padded first.
    #[module(skip)]
    pub same_pad: bool,
}

impl PreActConv {
    /// The conv's input channels.
    pub fn in_channels(&self) -> usize {
        self.conv.weight.dims()[1] * self.conv.groups
    }

    /// The conv's output channels.
    pub fn out_channels(&self) -> usize {
        self.conv.weight.dims()[0]
    }

    /// The conv's stride.
    pub fn stride(&self) -> usize {
        self.conv.stride[0]
    }

    /// The input, normalized, activated and convolved, with `f` run on the
    /// conv's output (where a regularizer goes).
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_height, in_width]`.
    /// - `f`: an endofunction over the conv's output.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn map_forward<F>(
        &self,
        input: Tensor<4>,
        f: F,
    ) -> Tensor<4>
    where
        F: FnOnce(Tensor<4>) -> Tensor<4>,
    {
        let [batch] = crate::contracts::unpack_shape_contract!(
            ["batch", "in_channels", _, _],
            &input.dims(),
            &["batch"],
            &[("in_channels", self.in_channels())]
        );
        let x = match &self.norm {
            Some(norm) => norm.forward(input),
            None => input,
        };
        let x = match &self.act {
            Some(act) => act.forward(x),
            None => x,
        };
        let x =
            crate::ops::conv::conv2d_variant_forward(&self.conv, x, self.weight_std, self.same_pad);
        let x = f(x);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "out_channels", _, _],
            &x.dims(),
            &[("batch", batch), ("out_channels", self.out_channels())]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;
        x
    }

    /// The input, normalized, activated and convolved.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        self.map_forward(input, |x| x)
    }
}
