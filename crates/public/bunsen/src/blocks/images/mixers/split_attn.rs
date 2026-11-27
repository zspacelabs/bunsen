//! `ResNeSt`'s split attention.

use alloc::format;

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
    tensor::{
        Device,
        activation::{
            sigmoid,
            softmax,
        },
    },
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        conv::{
            ConvBlock2d,
            ConvBlock2dConfig,
            ConvBlock2dMeta,
        },
        norm::BunsenNormalizationConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
    support::math::make_divisible,
};

/// `ResNeSt`'s split attention, the config of a [`SplitAttn`] (timm's
/// `SplitAttn`, "Splat").
///
/// A grouped conv widens the input to `radix` splits of `out_channels`, each
/// split a group of its own ([`conv`](Self::conv), its norm and activation:
/// timm's `conv`, `bn0`, `act0`). The splits' sum, averaged over the map,
/// goes through a `1x1` conv down to the attention width, a norm and the
/// activation ([`fc1`](Self::fc1): `fc1`, `bn1`, `act1`), and a `1x1` conv
/// back to every split's channels ([`fc2`](Self::fc2)). A softmax across the
/// splits of each channel (a sigmoid at a `radix` of 1) weighs them, and the
/// output is their weighted sum: `out_channels` wide.
#[derive(Config, Debug)]
pub struct SplitAttnConfig {
    /// The splits (timm's `radix`).
    pub radix: usize,

    /// The groups of each split (timm's `groups`, a `ResNeSt`'s
    /// `cardinality`): the conv's groups are `groups * radix`, the
    /// attention's `groups`.
    pub groups: usize,

    /// The grouped conv, its norm and its activation: `in_channels` to
    /// `out_channels * radix`.
    pub conv: ConvBlock2dConfig,

    /// The attention's first `1x1` conv (biased), its norm and its
    /// activation: `out_channels` to the attention width.
    pub fc1: ConvBlock2dConfig,

    /// The attention's second `1x1` conv (biased): the attention width to
    /// `out_channels * radix`.
    pub fc2: Conv2dConfig,
}

impl SplitAttnConfig {
    /// timm's `SplitAttn` from `in_channels` to `out_channels`: a `kernel_size`
    /// conv at `stride` and `dilation` (padded by the dilation), `radix`
    /// splits of `groups` groups, `norm` and `act` after the conv and the
    /// first attention conv, and an attention width of
    /// `make_divisible(in_channels * radix * rd_ratio, rd_divisor)`, at least
    /// 32 (timm's `rd_ratio`, `rd_divisor`; the floor of 32 and the rounding
    /// limit of 0.9 are timm's own, not knobs).
    #[allow(clippy::too_many_arguments)]
    pub fn timm(
        channels: [usize; 2],
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        radix: usize,
        rd_ratio: f64,
        rd_divisor: usize,
        norm: BunsenNormalizationConfig,
        act: BunsenActivationConfig,
    ) -> Self {
        let [in_channels, out_channels] = channels;
        let attn = make_divisible(
            (in_channels * radix) as f64 * rd_ratio,
            rd_divisor,
            Some(32),
            0.9,
        );
        let mid = out_channels * radix;
        let block = |conv: Conv2dConfig| {
            ConvBlock2dConfig::new(conv)
                .with_norm(Some(norm.clone()))
                .with_act(Some(act.clone()))
                .match_norm_features()
        };
        Self {
            radix,
            groups,
            conv: block(
                Conv2dConfig::new([in_channels, mid], [kernel_size, kernel_size])
                    .with_stride([stride, stride])
                    .with_dilation([dilation, dilation])
                    .with_padding(PaddingConfig2d::Explicit(
                        dilation, dilation, dilation, dilation,
                    ))
                    .with_groups(groups * radix)
                    .with_bias(false),
            ),
            fc1: block(Conv2dConfig::new([out_channels, attn], [1, 1]).with_groups(groups)),
            fc2: Conv2dConfig::new([attn, mid], [1, 1]).with_groups(groups),
        }
    }

    /// The input channels.
    pub fn in_channels(&self) -> usize {
        self.conv.in_channels()
    }

    /// The output channels: one split's.
    pub fn out_channels(&self) -> usize {
        self.conv.out_channels() / self.radix.max(1)
    }

    /// The stride: the conv's.
    pub fn stride(&self) -> usize {
        self.conv.stride()[0]
    }

    /// Checks the shape: a nonzero radix whose splits the conv's groups and
    /// channels divide into, and an attention from one split's channels back
    /// to all of them.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when it isn't.
    pub fn try_validate(&self) -> BunsenResult<()> {
        let illegal = |field: &'static str, message: alloc::string::String| {
            Err(ConstraintError::custom("SplitAttnConfig", field, message).into())
        };
        let (radix, groups) = (self.radix, self.groups);
        for (field, value) in [("radix", radix), ("groups", groups)] {
            if value == 0 {
                return Err(ConstraintError::zero_or_empty("SplitAttnConfig", field).into());
            }
        }
        let mid = self.conv.out_channels();
        if self.conv.groups() != groups * radix || !mid.is_multiple_of(radix) {
            return illegal(
                "conv",
                format!(
                    "the conv is {mid} wide in {} groups; {radix} splits of {groups} groups",
                    self.conv.groups()
                ),
            );
        }
        let c = mid / radix;
        let attn = self.fc1.out_channels();
        if self.fc1.in_channels() != c || self.fc2.channels != [attn, mid] {
            return illegal(
                "fc1",
                format!(
                    "the attention is {} to {attn}, then {:?}; it reads {c} channels and weighs {mid}",
                    self.fc1.in_channels(),
                    self.fc2.channels
                ),
            );
        }
        Ok(())
    }
}

impl ModuleInit<SplitAttn> for SplitAttnConfig {
    /// Builds the convs.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](SplitAttnConfig::try_validate), and as the conv
    /// blocks' `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SplitAttn> {
        self.try_validate()?;
        Ok(SplitAttn {
            conv: self.conv.try_init(device)?,
            fc1: self.fc1.try_init(device)?,
            fc2: self.fc2.init(device),
            radix: self.radix,
            groups: self.groups,
        })
    }
}

/// `ResNeSt`'s split attention; built by [`SplitAttnConfig`].
#[derive(Module, Debug)]
pub struct SplitAttn {
    /// The grouped conv, its norm and activation.
    pub conv: ConvBlock2d,

    /// The attention's first conv, its norm and activation.
    pub fc1: ConvBlock2d,

    /// The attention's second conv.
    pub fc2: Conv2d,

    /// The splits.
    #[module(skip)]
    pub radix: usize,

    /// The groups of each split.
    #[module(skip)]
    pub groups: usize,
}

impl SplitAttn {
    /// The input channels.
    pub fn in_channels(&self) -> usize {
        self.conv.in_channels()
    }

    /// The output channels: one split's.
    pub fn out_channels(&self) -> usize {
        self.conv.out_channels() / self.radix
    }

    /// The stride.
    pub fn stride(&self) -> usize {
        self.conv.stride()[0]
    }

    /// The split attention, with `f` run on the conv's normalized output,
    /// before its activation (where timm's `drop_block` goes).
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, height, width]`.
    /// - `f`: an endofunction over the conv's normalized output.
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
        let x = self.conv.map_forward(input, f);
        let [b, rc, h, w] = x.dims();
        let (radix, c) = (self.radix, rc / self.radix);
        let splits = (radix > 1).then(|| x.clone().reshape([b, radix, c, h, w]));
        let gap = match &splits {
            Some(splits) => splits.clone().sum_dim(1).reshape([b, c, h, w]),
            None => x.clone(),
        };
        let gap = gap.mean_dim(2).mean_dim(3);
        let attn = self.fc2.forward(self.fc1.forward(gap));
        // timm's `RadixSoftmax`: across the splits of each channel, within
        // each group; the result radix-major, as the splits are.
        let out = match splits {
            Some(splits) => {
                let attn = softmax(
                    attn.reshape([b, self.groups, radix, rc / (self.groups * radix)])
                        .swap_dims(1, 2),
                    1,
                )
                .reshape([b, radix, c, 1, 1]);
                (splits * attn).sum_dim(1).reshape([b, c, h, w])
            }
            None => x * sigmoid(attn),
        };
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "out_channels", "height", "width"],
            &out.dims(),
            &[
                ("batch", b),
                ("out_channels", c),
                ("height", h),
                ("width", w)
            ]
        );
        out
    }

    /// The split attention.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, height, width]`.
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

#[cfg(test)]
mod tests {
    use burn::{
        nn::BatchNormConfig,
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    fn bn() -> BunsenNormalizationConfig {
        BunsenNormalizationConfig::Batch(BatchNormConfig::new(0))
    }

    /// timm's widths: a `ResNeSt-50`'s first mixer, 64 wide at radix 2,
    /// attends over `make_divisible(64 * 2 / 4)`, 32.
    #[test]
    fn test_timm_widths() {
        let config = SplitAttnConfig::timm(
            [64, 64],
            3,
            2,
            1,
            1,
            2,
            0.25,
            8,
            bn(),
            BunsenActivationConfig::Relu,
        );
        assert_eq!(config.conv.out_channels(), 128);
        assert_eq!(config.conv.groups(), 2);
        assert_eq!(config.fc1.out_channels(), 32);
        assert_eq!(config.fc2.channels, [32, 128]);
        assert_eq!((config.in_channels(), config.out_channels()), (64, 64));
        config.try_validate().unwrap();
    }

    /// At radix 1, the attention is a sigmoid gate on the conv's output.
    #[test]
    fn test_radix_one_is_a_gate() {
        let device = cpu_device();
        let config = SplitAttnConfig::timm(
            [8, 8],
            3,
            1,
            1,
            2,
            1,
            0.25,
            8,
            bn(),
            BunsenActivationConfig::Relu,
        );
        let attn: SplitAttn = config.init(&device);
        let x = Tensor::<4>::random([2, 8, 5, 5], Distribution::Default, &device);
        let y = attn.conv.forward(x.clone());
        let gate = sigmoid(
            attn.fc2
                .forward(attn.fc1.forward(y.clone().mean_dim(2).mean_dim(3))),
        );
        attn.forward(x)
            .into_data()
            .assert_approx_eq::<f32>(&(y * gate).into_data(), Tolerance::rel_abs(1e-6, 1e-6));
    }

    /// Weights that are equal across the splits average them: with `fc2`'s
    /// weights and bias zero, the softmax is `1 / radix` everywhere.
    #[test]
    fn test_even_weights_average_the_splits() {
        let device = cpu_device();
        let config = SplitAttnConfig::timm(
            [8, 8],
            3,
            1,
            1,
            1,
            4,
            0.25,
            8,
            bn(),
            BunsenActivationConfig::Relu,
        );
        let mut attn: SplitAttn = config.init(&device);
        attn.fc2.weight = attn.fc2.weight.clone().map(|w| w.zeros_like());
        attn.fc2.bias = attn.fc2.bias.clone().map(|b| b.map(|b| b.zeros_like()));
        let x = Tensor::<4>::random([1, 8, 4, 4], Distribution::Default, &device);
        let y = attn.conv.forward(x.clone());
        let mean = y.reshape([1, 4, 8, 4, 4]).mean_dim(1).reshape([1, 8, 4, 4]);
        attn.forward(x)
            .into_data()
            .assert_approx_eq::<f32>(&mean.into_data(), Tolerance::rel_abs(1e-6, 1e-6));
    }

    #[test]
    fn test_illegal_shapes() {
        let mut config = SplitAttnConfig::timm(
            [8, 8],
            3,
            1,
            1,
            1,
            2,
            0.25,
            8,
            bn(),
            BunsenActivationConfig::Relu,
        );
        config.radix = 4;
        ErrorMatcher::illegal()
            .message_contains("the conv is 16 wide in 2 groups; 4 splits of 1 groups")
            .assert_err(&config.try_validate());
    }
}
