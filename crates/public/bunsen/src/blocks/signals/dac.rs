//! [`DacDecoder`]: DAC's decoder, its blocks and residual units, and their configs.

use alloc::{
    format,
    string::ToString,
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
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::conv::{
        CausalConvStreamContext,
        CausalConvTransposeStreamContext,
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

/// One chunk through a stride-1 causal convolution's stream: every frame is
/// an output.
fn conv_stream(
    conv: &CausalConv1d,
    x: Tensor<3>,
    ctx: &mut CausalConvStreamContext,
) -> Tensor<3> {
    conv.forward_stream(x, ctx)
        .expect("a stride-1 causal convolution completes every frame")
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

    /// A stream through the unit, not yet started.
    pub fn new_stream_context(&self) -> DacResidualUnitStreamContext {
        DacResidualUnitStreamContext {
            conv1: self.conv1.new_stream_context(),
            conv2: self.conv2.new_stream_context(),
        }
    }

    /// The stream's next chunk; joined, the chunks' outputs are
    /// [`forward`](Self::forward)'s.
    ///
    /// # Arguments
    /// - `x`: `[B, C, T]`, `T ≥ 1`.
    ///
    /// # Returns
    /// `[B, C, T]`.
    pub fn forward_stream(
        &self,
        x: Tensor<3>,
        ctx: &mut DacResidualUnitStreamContext,
    ) -> Tensor<3> {
        let h = conv_stream(&self.conv1, self.act1.forward(x.clone()), &mut ctx.conv1);
        x + conv_stream(&self.conv2, self.act2.forward(h), &mut ctx.conv2)
    }
}

/// One stream through a [`DacResidualUnit`]: its convolutions' streams.
///
/// A `Module` over bare tensors (see [`CausalConvStreamContext`]): moved with
/// `to_device`, never recorded or visited.
#[derive(Module, Debug)]
pub struct DacResidualUnitStreamContext {
    /// The dilated convolution's.
    pub conv1: CausalConvStreamContext,

    /// The pointwise convolution's.
    pub conv2: CausalConvStreamContext,
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
            return Err(ConstraintError::new(
                "DacDecoderBlockConfig",
                "residuals",
                Rule::Relation {
                    lhs: (
                        "a residual unit's channels".into(),
                        unit.channels.to_string(),
                    ),
                    op: "==",
                    rhs: ("the block's output channels".into(), c_out.to_string()),
                },
            )
            .into());
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

    /// A stream through the block, not yet started.
    pub fn new_stream_context(&self) -> DacDecoderBlockStreamContext {
        DacDecoderBlockStreamContext {
            upsample: self.upsample.new_stream_context(),
            residuals: self
                .residuals
                .iter()
                .map(DacResidualUnit::new_stream_context)
                .collect(),
        }
    }

    /// The stream's next chunk; joined, the chunks' outputs are
    /// [`forward`](Self::forward)'s.
    ///
    /// # Arguments
    /// - `x`: `[B, C_in, T]`, `T ≥ 1`.
    ///
    /// # Returns
    /// `[B, C_out, T · factor]`.
    pub fn forward_stream(
        &self,
        x: Tensor<3>,
        ctx: &mut DacDecoderBlockStreamContext,
    ) -> Tensor<3> {
        let x = self
            .upsample
            .forward_stream(self.act.forward(x), &mut ctx.upsample);
        self.residuals
            .iter()
            .zip(&mut ctx.residuals)
            .fold(x, |x, (unit, ctx)| unit.forward_stream(x, ctx))
    }
}

/// One stream through a [`DacDecoderBlock`]: its upsampler's and its
/// residual units' streams.
///
/// A `Module` over bare tensors (see [`CausalConvStreamContext`]): moved with
/// `to_device`, never recorded or visited.
#[derive(Module, Debug)]
pub struct DacDecoderBlockStreamContext {
    /// The upsampler's.
    pub upsample: CausalConvTransposeStreamContext,

    /// Each residual unit's, in order.
    pub residuals: Vec<DacResidualUnitStreamContext>,
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
                return Err(ConstraintError::new(
                    "DacDecoderConfig",
                    "blocks",
                    Rule::Chain {
                        index: i,
                        out: format!("{width} channels"),
                        r#in: format!("{} channels", block.channels[0]),
                    },
                )
                .into());
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

impl DacDecoder {
    /// Latents to a waveform, the whole input at once.
    ///
    /// # Arguments
    /// - `z`: `[B, L, T]`, `L` the latent width.
    ///
    /// # Returns
    /// `[B, 1, T · factor]`.
    pub fn forward(
        &self,
        z: Tensor<3>,
    ) -> Tensor<3> {
        let y = self
            .blocks
            .iter()
            .fold(self.input.forward(z), |y, block| block.forward(y));
        self.output.forward(self.output_act.forward(y))
    }

    /// A stream through the decoder, not yet started.
    pub fn new_stream_context(&self) -> DacDecoderStreamContext {
        DacDecoderStreamContext {
            input: self.input.new_stream_context(),
            blocks: self
                .blocks
                .iter()
                .map(DacDecoderBlock::new_stream_context)
                .collect(),
            output: self.output.new_stream_context(),
        }
    }

    /// The stream's next chunk of latents, to its samples; joined, the
    /// chunks' outputs are [`forward`](Self::forward)'s. Every layer is a
    /// causal convolution of stride 1 or an upsampler, so a chunk's samples
    /// are final at once, and a stream needs no finish.
    ///
    /// # Arguments
    /// - `z`: `[B, L, T]`, `T ≥ 1`.
    ///
    /// # Returns
    /// `[B, 1, T · factor]`.
    pub fn forward_stream(
        &self,
        z: Tensor<3>,
        ctx: &mut DacDecoderStreamContext,
    ) -> Tensor<3> {
        let y = conv_stream(&self.input, z, &mut ctx.input);
        let y = self
            .blocks
            .iter()
            .zip(&mut ctx.blocks)
            .fold(y, |y, (block, ctx)| block.forward_stream(y, ctx));
        conv_stream(&self.output, self.output_act.forward(y), &mut ctx.output)
    }
}

/// One stream through a [`DacDecoder`]: each layer's stream.
///
/// A `Module` over bare tensors (see [`CausalConvStreamContext`]): moved with
/// `to_device`, never recorded or visited.
#[derive(Module, Debug)]
pub struct DacDecoderStreamContext {
    /// The input convolution's.
    pub input: CausalConvStreamContext,

    /// Each block's, in order.
    pub blocks: Vec<DacDecoderBlockStreamContext>,

    /// The output convolution's.
    pub output: CausalConvStreamContext,
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    /// A small vocoder: two blocks (x2 and x3), two residual units each.
    fn decoder(device: &Device) -> DacDecoder {
        let residuals = |c: usize| {
            [1, 3]
                .into_iter()
                .map(|d| DacResidualUnitConfig::new(c, d))
                .collect::<Vec<_>>()
        };
        DacDecoderConfig::new(
            3,
            8,
            alloc::vec![
                DacDecoderBlockConfig::new([8, 4], 2, 4, residuals(4)),
                DacDecoderBlockConfig::new([4, 2], 3, 6, residuals(2)),
            ],
        )
        .init(device)
    }

    /// A stream's samples are the whole forward's, whatever the chunks of
    /// latents.
    #[test]
    fn test_stream_matches_forward() {
        let device = cpu_device();
        let decoder = decoder(&device);
        for t in [1, 4, 9] {
            let z = seeded_tensor(t as u64, [2, 3, t], Distribution::Normal(0.0, 1.0), &device);
            let whole = decoder.forward(z.clone());
            assert_eq!(whole.dims(), [2, 1, t * 6]);
            for size in [1, 2, 4, t] {
                let mut ctx = decoder.new_stream_context();
                let chunks: Vec<Tensor<3>> = (0..t)
                    .step_by(size)
                    .map(|at| {
                        let end = (at + size).min(t);
                        decoder.forward_stream(z.clone().slice(s![.., .., at..end]), &mut ctx)
                    })
                    .collect();
                assert_tensors_close(
                    &Tensor::cat(chunks, 2),
                    &whole,
                    Tolerance::rel_abs(1e-5, 1e-6),
                );
            }
        }
    }
}
