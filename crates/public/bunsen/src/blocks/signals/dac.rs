//! [`DacDecoder`]: DAC's decoder, its blocks and residual units, and their configs.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    prelude::*,
};

use crate::{
    blocks::{
        activation::{
            SnakeActivation,
            SnakeActivationConfig,
        },
        conv::{
            CausalConv1d,
            CausalConv1dConfig,
            CausalConv1dMeta,
            CausalConvTranspose1d,
            CausalConvTranspose1dConfig,
            CausalConvTranspose1dMeta,
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The decoder's activation, `SnakeBeta` in its log-scale parameterization,
/// over `channels`.
fn snake(
    channels: usize,
    device: &Device,
) -> BunsenResult<SnakeActivation> {
    SnakeActivationConfig::snake_beta_log_scale(channels).try_init(device)
}

/// A causal convolution, `channels` `[in, out]`, of `kernel` taps spaced
/// `dilation` apart.
fn causal_conv(
    channels: [usize; 2],
    kernel: usize,
    dilation: usize,
    device: &Device,
) -> BunsenResult<CausalConv1d> {
    CausalConv1dConfig::new(channels, kernel)
        .with_dilation(dilation)
        .try_init(device)
}

/// Common meta for [`DacResidualUnit`] and
/// [`DacResidualUnitConfig`].
pub trait DacResidualUnitMeta {
    /// The channels, `C`, in and out.
    fn channels(&self) -> usize;

    /// The dilated convolution's dilation.
    fn dilation(&self) -> usize;

    /// The dilated convolution's taps.
    fn kernel(&self) -> usize;

    /// The pointwise convolution's taps.
    fn pointwise_kernel(&self) -> usize;
}

/// Config for [`DacResidualUnit`]; the default taps are DAC's.
#[derive(Config, Debug)]
pub struct DacResidualUnitConfig {
    /// The channels, `C`.
    pub channels: usize,

    /// The dilated convolution's dilation.
    pub dilation: usize,

    /// The dilated convolution's taps.
    #[config(default = 7)]
    pub kernel: usize,

    /// The pointwise convolution's taps.
    #[config(default = 1)]
    pub pointwise_kernel: usize,
}

impl DacResidualUnitMeta for DacResidualUnitConfig {
    fn channels(&self) -> usize {
        self.channels
    }

    fn dilation(&self) -> usize {
        self.dilation
    }

    fn kernel(&self) -> usize {
        self.kernel
    }

    fn pointwise_kernel(&self) -> usize {
        self.pointwise_kernel
    }
}

impl ModuleInit<DacResidualUnit> for DacResidualUnitConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<DacResidualUnit> {
        let c = self.channels;
        Ok(DacResidualUnit {
            act1: snake(c, device)?,
            conv1: causal_conv([c, c], self.kernel, self.dilation, device)?,
            act2: snake(c, device)?,
            conv2: causal_conv([c, c], self.pointwise_kernel, 1, device)?,
        })
    }
}

/// A DAC residual unit: `SnakeBeta`, a dilated causal convolution,
/// `SnakeBeta`, a (pointwise) convolution, and the input added.
///
/// Built by [`DacResidualUnitConfig`].
#[derive(Module, Debug)]
pub struct DacResidualUnit {
    /// The first activation.
    pub act1: SnakeActivation,

    /// The dilated convolution.
    pub conv1: CausalConv1d,

    /// The second activation.
    pub act2: SnakeActivation,

    /// The pointwise convolution.
    pub conv2: CausalConv1d,
}

impl DacResidualUnitMeta for DacResidualUnit {
    fn channels(&self) -> usize {
        self.conv1.channels()[0]
    }

    fn dilation(&self) -> usize {
        self.conv1.dilation()
    }

    fn kernel(&self) -> usize {
        self.conv1.kernel_size()
    }

    fn pointwise_kernel(&self) -> usize {
        self.conv2.kernel_size()
    }
}

impl DacResidualUnit {
    /// `[B, C, T]` to `[B, C, T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let h = self.conv1.forward(self.act1.forward(x.clone()));
        x + self.conv2.forward(self.act2.forward(h))
    }
}

/// Common meta for [`DacDecoderBlock`] and
/// [`DacDecoderBlockConfig`].
pub trait DacDecoderBlockMeta {
    /// The channels, `[C_in, C_out]`.
    fn channels(&self) -> [usize; 2];

    /// The factor, `r`: output frames per input frame.
    fn factor(&self) -> usize;

    /// The upsampler's taps.
    fn kernel(&self) -> usize;

    /// The residual units' dilations, in order.
    fn dilations(&self) -> Vec<usize>;
}

/// Config for [`DacDecoderBlock`]: its upsampler, and each residual
/// unit's config.
#[derive(Config, Debug)]
pub struct DacDecoderBlockConfig {
    /// The channels, `[C_in, C_out]`.
    pub channels: [usize; 2],

    /// The factor, `r`.
    pub factor: usize,

    /// The upsampler's taps; at least `r`.
    pub kernel: usize,

    /// The residual units, in order, each over `C_out`.
    pub residuals: Vec<DacResidualUnitConfig>,
}

impl DacDecoderBlockMeta for DacDecoderBlockConfig {
    fn channels(&self) -> [usize; 2] {
        self.channels
    }

    fn factor(&self) -> usize {
        self.factor
    }

    fn kernel(&self) -> usize {
        self.kernel
    }

    fn dilations(&self) -> Vec<usize> {
        self.residuals.iter().map(|u| u.dilation).collect()
    }
}

impl ModuleInit<DacDecoderBlock> for DacDecoderBlockConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a residual
    /// unit not over `C_out`; as the convolutions' builds: a factor of 0, or
    /// a kernel shorter than it, is refused.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<DacDecoderBlock> {
        let [c_in, c_out] = self.channels;
        if let Some(unit) = self.residuals.iter().find(|u| u.channels != c_out) {
            return Err(BunsenError::illegal(format!(
                "a vocoder block writing {c_out} channels with a residual unit over {}",
                unit.channels
            )));
        }
        Ok(DacDecoderBlock {
            act: snake(c_in, device)?,
            upsample: CausalConvTranspose1dConfig::new([c_in, c_out], self.kernel)
                .with_stride(self.factor)
                .try_init(device)?,
            residuals: self
                .residuals
                .iter()
                .map(|u| u.try_init(device))
                .collect::<BunsenResult<_>>()?,
        })
    }
}

/// A DAC decoder block: `SnakeBeta`, a causal transposed convolution
/// upsampling by `r` (DAC's kernel `2 r`) and halving the width, then
/// residual units (DAC's of dilation 1, 3 and 9).
///
/// Built by [`DacDecoderBlockConfig`].
#[derive(Module, Debug)]
pub struct DacDecoderBlock {
    /// The activation before the upsampler.
    pub act: SnakeActivation,

    /// The upsampler.
    pub upsample: CausalConvTranspose1d,

    /// The residual units.
    pub residuals: Vec<DacResidualUnit>,
}

impl DacDecoderBlockMeta for DacDecoderBlock {
    fn channels(&self) -> [usize; 2] {
        self.upsample.channels()
    }

    fn factor(&self) -> usize {
        self.upsample.stride()
    }

    fn kernel(&self) -> usize {
        self.upsample.kernel_size()
    }

    fn dilations(&self) -> Vec<usize> {
        self.residuals.iter().map(|u| u.dilation()).collect()
    }
}

impl DacDecoderBlock {
    /// `[B, C_in, T]` to `[B, C_out, r T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let x = self.upsample.forward(self.act.forward(x));
        self.residuals.iter().fold(x, |x, unit| unit.forward(x))
    }
}

/// Common meta for [`DacDecoder`] and [`DacDecoderConfig`].
pub trait DacDecoderMeta {
    /// The latent width, `L`: its input's.
    fn latent_width(&self) -> usize;

    /// Its width, `V`: its first block's input.
    fn width(&self) -> usize;

    /// Each block's factor.
    fn factors(&self) -> Vec<usize>;

    /// Its last width, its output convolution's input: the last block's
    /// output, or `V` without blocks.
    fn out_width(&self) -> usize;

    /// The input convolution's taps.
    fn in_kernel(&self) -> usize;

    /// The output convolution's taps.
    fn out_kernel(&self) -> usize;

    /// Output samples per input frame: the blocks' factors, multiplied.
    fn factor(&self) -> usize {
        self.factors().iter().product()
    }
}

/// Config for [`DacDecoder`]: its convolutions, and each block's config.
#[derive(Config, Debug)]
pub struct DacDecoderConfig {
    /// The latent width, `L`.
    pub latent_width: usize,

    /// The width, `V`: the first block's input.
    pub width: usize,

    /// The blocks, in order, each reading the one before's output.
    pub blocks: Vec<DacDecoderBlockConfig>,

    /// The input convolution's taps.
    #[config(default = 7)]
    pub in_kernel: usize,

    /// The output convolution's taps.
    #[config(default = 7)]
    pub out_kernel: usize,
}

impl DacDecoderMeta for DacDecoderConfig {
    fn latent_width(&self) -> usize {
        self.latent_width
    }

    fn width(&self) -> usize {
        self.width
    }

    fn factors(&self) -> Vec<usize> {
        self.blocks.iter().map(|b| b.factor).collect()
    }

    fn out_width(&self) -> usize {
        self.blocks.last().map_or(self.width, |b| b.channels[1])
    }

    fn in_kernel(&self) -> usize {
        self.in_kernel
    }

    fn out_kernel(&self) -> usize {
        self.out_kernel
    }
}

impl ModuleInit<DacDecoder> for DacDecoderConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a block that
    /// does not read what the one before it (or the input convolution)
    /// writes; as the blocks' builds.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<DacDecoder> {
        let mut width = self.width;
        for (i, block) in self.blocks.iter().enumerate() {
            if block.channels[0] != width {
                return Err(BunsenError::illegal(format!(
                    "vocoder block {i} reads {} channels; it is given {width}",
                    block.channels[0]
                )));
            }
            width = block.channels[1];
        }
        let out = self.out_width();
        Ok(DacDecoder {
            input: causal_conv([self.latent_width, self.width], self.in_kernel, 1, device)?,
            blocks: self
                .blocks
                .iter()
                .map(|b| b.try_init(device))
                .collect::<BunsenResult<_>>()?,
            output_act: snake(out, device)?,
            output: causal_conv([out, 1], self.out_kernel, 1, device)?,
        })
    }
}

/// DAC's decoder, a vocoder: a causal convolution into its width, its
/// blocks, then `SnakeBeta` and a causal convolution to one channel.
///
/// Built by [`DacDecoderConfig`].
#[derive(Module, Debug)]
pub struct DacDecoder {
    /// `[L, V]`.
    pub input: CausalConv1d,

    /// The blocks, each upsampling and (the release's) halving the width.
    pub blocks: Vec<DacDecoderBlock>,

    /// The activation before the output.
    pub output_act: SnakeActivation,

    /// To one channel.
    pub output: CausalConv1d,
}

impl DacDecoderMeta for DacDecoder {
    fn latent_width(&self) -> usize {
        self.input.channels()[0]
    }

    fn width(&self) -> usize {
        self.input.channels()[1]
    }

    fn factors(&self) -> Vec<usize> {
        self.blocks.iter().map(|b| b.factor()).collect()
    }

    fn out_width(&self) -> usize {
        self.output.channels()[0]
    }

    fn in_kernel(&self) -> usize {
        self.input.kernel_size()
    }

    fn out_kernel(&self) -> usize {
        self.output.kernel_size()
    }
}
