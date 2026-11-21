//! The codec decoder's modules.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    nn::{
        Linear,
        LinearConfig,
        RmsNormConfig,
    },
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
            CausalConvNext1d,
            CausalConvNext1dConfig,
            CausalConvNext1dMeta,
            CausalConvTranspose1d,
            CausalConvTranspose1dConfig,
            CausalConvTranspose1dMeta,
        },
        mlp::BunsenGatedMlpConfig,
        norm::BunsenNormalizationConfig,
        quantization::{
            EuclideanCodebookMeta,
            SplitResidualVectorQuantizer,
            SplitResidualVectorQuantizerConfig,
            SplitResidualVectorQuantizerMeta,
        },
        transformers::{
            attention::csa::CausalSelfAttentionConfig,
            decoders::{
                causal::{
                    CausalDecoderBlockConfig,
                    CausalDecoderBlockMeta,
                },
                stack::{
                    DecoderStack,
                    UniformStackConfig,
                },
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
            },
        },
    },
    burner::{
        module::{
            CanonicalKeys,
            ModuleInit,
            Rekey,
            ToStructureConfig,
        },
        tensor::Widening,
    },
    contracts::unpack_shape_contract,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The `SnakeBeta` of the vocoder's, over `channels`.
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

/// Common meta for [`Qwen3TtsCodecTransformer`] and
/// [`Qwen3TtsCodecTransformerConfig`].
pub trait Qwen3TtsCodecTransformerMeta {
    /// The latent width, `L`: its input's and output's.
    fn latent_width(&self) -> usize;

    /// The stack's width, `W`.
    fn width(&self) -> usize;
}

/// Config for [`Qwen3TtsCodecTransformer`].
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecTransformerConfig {
    /// The latent width, `L`.
    pub latent_width: usize,

    /// The stack; its layers' width is the transformer's
    /// [`width`](Qwen3TtsCodecTransformerMeta::width).
    pub stack: UniformStackConfig,
}

impl Qwen3TtsCodecTransformerMeta for Qwen3TtsCodecTransformerConfig {
    fn latent_width(&self) -> usize {
        self.latent_width
    }

    fn width(&self) -> usize {
        self.stack.layer.n_embed()
    }
}

impl ModuleInit<Qwen3TtsCodecTransformer> for Qwen3TtsCodecTransformerConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsCodecTransformer> {
        let [l, w] = [self.latent_width, self.width()];
        Ok(Qwen3TtsCodecTransformer {
            input_proj: LinearConfig::new(l, w).init(device),
            stack: self.stack.try_init(device)?,
            output_proj: LinearConfig::new(w, l).init(device),
        })
    }
}

/// The transformer between the pre-conv and the upsamplers: a projection
/// in, a decoder stack, a projection out.
///
/// Built by [`Qwen3TtsCodecTransformerConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsCodecTransformer {
    /// `[L, W]`, with a bias.
    pub input_proj: Linear,

    /// The layers and the final norm.
    pub stack: DecoderStack,

    /// `[W, L]`, with a bias.
    pub output_proj: Linear,
}

impl Qwen3TtsCodecTransformerMeta for Qwen3TtsCodecTransformer {
    fn latent_width(&self) -> usize {
        self.input_proj.weight.dims()[0]
    }

    fn width(&self) -> usize {
        self.input_proj.weight.dims()[1]
    }
}

impl Qwen3TtsCodecTransformer {
    /// `[B, T, L]` to `[B, T, L]`, positions from 0.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let t = x.dims()[1];
        let x = self.input_proj.forward(x);
        let x = self
            .stack
            .forward_hidden(x.clone(), &self.stack.rotary(t, &x.device()), None);
        self.output_proj.forward(x)
    }
}

/// Common meta for [`Qwen3TtsCodecUpsample`] and
/// [`Qwen3TtsCodecUpsampleConfig`].
pub trait Qwen3TtsCodecUpsampleMeta {
    /// The width, `L`, in and out.
    fn width(&self) -> usize;

    /// The factor, `r`: output frames per input frame.
    fn factor(&self) -> usize;

    /// The `ConvNeXt` block's hidden width.
    fn hidden(&self) -> usize;

    /// The `ConvNeXt` block's depthwise taps.
    fn block_kernel(&self) -> usize;
}

/// Config for [`Qwen3TtsCodecUpsample`]: its factor, and its `ConvNeXt`
/// block's config.
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecUpsampleConfig {
    /// The width, `L`.
    pub width: usize,

    /// The factor, `r`.
    pub factor: usize,

    /// The `ConvNeXt` block, over `L`.
    pub block: CausalConvNext1dConfig,
}

impl Qwen3TtsCodecUpsampleMeta for Qwen3TtsCodecUpsampleConfig {
    fn width(&self) -> usize {
        self.width
    }

    fn factor(&self) -> usize {
        self.factor
    }

    fn hidden(&self) -> usize {
        self.block.hidden
    }

    fn block_kernel(&self) -> usize {
        self.block.kernel
    }
}

impl ModuleInit<Qwen3TtsCodecUpsample> for Qwen3TtsCodecUpsampleConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a block not
    /// over `L`; as the convolutions' builds: a factor of 0 is refused.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsCodecUpsample> {
        let [w, r] = [self.width, self.factor];
        if self.block.width != w {
            return Err(BunsenError::illegal(format!(
                "an upsampler of width {w} with a ConvNeXt block of width {}",
                self.block.width
            )));
        }
        Ok(Qwen3TtsCodecUpsample {
            up: CausalConvTranspose1dConfig::new([w, w], r)
                .with_stride(r)
                .try_init(device)?,
            block: self.block.try_init(device)?,
        })
    }
}

/// One upsampler: a causal transposed convolution (kernel and stride its
/// factor), then a causal `ConvNeXt` block (the release's with a layer
/// scale).
///
/// Built by [`Qwen3TtsCodecUpsampleConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsCodecUpsample {
    /// The transposed convolution.
    pub up: CausalConvTranspose1d,

    /// The `ConvNeXt` block, channels last.
    pub block: CausalConvNext1d,
}

impl Qwen3TtsCodecUpsampleMeta for Qwen3TtsCodecUpsample {
    fn width(&self) -> usize {
        self.up.channels()[0]
    }

    fn factor(&self) -> usize {
        self.up.stride()
    }

    fn hidden(&self) -> usize {
        self.block.hidden()
    }

    fn block_kernel(&self) -> usize {
        self.block.kernel()
    }
}

impl Qwen3TtsCodecUpsample {
    /// `[B, L, T]` to `[B, L, r T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        self.block
            .forward(self.up.forward(x).swap_dims(1, 2))
            .swap_dims(1, 2)
    }
}

/// Common meta for [`Qwen3TtsVocoderResidualUnit`] and
/// [`Qwen3TtsVocoderResidualUnitConfig`].
pub trait Qwen3TtsVocoderResidualUnitMeta {
    /// The channels, `C`, in and out.
    fn channels(&self) -> usize;

    /// The dilated convolution's dilation.
    fn dilation(&self) -> usize;

    /// The dilated convolution's taps.
    fn kernel(&self) -> usize;

    /// The pointwise convolution's taps.
    fn pointwise_kernel(&self) -> usize;
}

/// Config for [`Qwen3TtsVocoderResidualUnit`]; the default taps are DAC's.
#[derive(Config, Debug)]
pub struct Qwen3TtsVocoderResidualUnitConfig {
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

impl Qwen3TtsVocoderResidualUnitMeta for Qwen3TtsVocoderResidualUnitConfig {
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

impl ModuleInit<Qwen3TtsVocoderResidualUnit> for Qwen3TtsVocoderResidualUnitConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsVocoderResidualUnit> {
        let c = self.channels;
        Ok(Qwen3TtsVocoderResidualUnit {
            act1: snake(c, device)?,
            conv1: causal_conv([c, c], self.kernel, self.dilation, device)?,
            act2: snake(c, device)?,
            conv2: causal_conv([c, c], self.pointwise_kernel, 1, device)?,
        })
    }
}

/// A vocoder residual unit: `SnakeBeta`, a dilated causal convolution,
/// `SnakeBeta`, a (pointwise) convolution, and the input added.
///
/// Built by [`Qwen3TtsVocoderResidualUnitConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsVocoderResidualUnit {
    /// The first activation.
    pub act1: SnakeActivation,

    /// The dilated convolution.
    pub conv1: CausalConv1d,

    /// The second activation.
    pub act2: SnakeActivation,

    /// The pointwise convolution.
    pub conv2: CausalConv1d,
}

impl Qwen3TtsVocoderResidualUnitMeta for Qwen3TtsVocoderResidualUnit {
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

impl Qwen3TtsVocoderResidualUnit {
    /// `[B, C, T]` to `[B, C, T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let h = self.conv1.forward(self.act1.forward(x.clone()));
        x + self.conv2.forward(self.act2.forward(h))
    }
}

/// Common meta for [`Qwen3TtsVocoderBlock`] and
/// [`Qwen3TtsVocoderBlockConfig`].
pub trait Qwen3TtsVocoderBlockMeta {
    /// The channels, `[C_in, C_out]`.
    fn channels(&self) -> [usize; 2];

    /// The factor, `r`: output frames per input frame.
    fn factor(&self) -> usize;

    /// The upsampler's taps.
    fn kernel(&self) -> usize;

    /// The residual units' dilations, in order.
    fn dilations(&self) -> Vec<usize>;
}

/// Config for [`Qwen3TtsVocoderBlock`]: its upsampler, and each residual
/// unit's config.
#[derive(Config, Debug)]
pub struct Qwen3TtsVocoderBlockConfig {
    /// The channels, `[C_in, C_out]`.
    pub channels: [usize; 2],

    /// The factor, `r`.
    pub factor: usize,

    /// The upsampler's taps; at least `r`.
    pub kernel: usize,

    /// The residual units, in order, each over `C_out`.
    pub residuals: Vec<Qwen3TtsVocoderResidualUnitConfig>,
}

impl Qwen3TtsVocoderBlockMeta for Qwen3TtsVocoderBlockConfig {
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

impl ModuleInit<Qwen3TtsVocoderBlock> for Qwen3TtsVocoderBlockConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a residual
    /// unit not over `C_out`; as the convolutions' builds: a factor of 0, or
    /// a kernel shorter than it, is refused.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsVocoderBlock> {
        let [c_in, c_out] = self.channels;
        if let Some(unit) = self.residuals.iter().find(|u| u.channels != c_out) {
            return Err(BunsenError::illegal(format!(
                "a vocoder block writing {c_out} channels with a residual unit over {}",
                unit.channels
            )));
        }
        Ok(Qwen3TtsVocoderBlock {
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

/// A vocoder block (DAC's): `SnakeBeta`, a causal transposed convolution
/// upsampling by `r` (the release's kernel `2 r`) and halving the width,
/// then residual units (the release's of dilation 1, 3 and 9).
///
/// Built by [`Qwen3TtsVocoderBlockConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsVocoderBlock {
    /// The activation before the upsampler.
    pub act: SnakeActivation,

    /// The upsampler.
    pub upsample: CausalConvTranspose1d,

    /// The residual units.
    pub residuals: Vec<Qwen3TtsVocoderResidualUnit>,
}

impl Qwen3TtsVocoderBlockMeta for Qwen3TtsVocoderBlock {
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

impl Qwen3TtsVocoderBlock {
    /// `[B, C_in, T]` to `[B, C_out, r T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let x = self.upsample.forward(self.act.forward(x));
        self.residuals.iter().fold(x, |x, unit| unit.forward(x))
    }
}

/// Common meta for [`Qwen3TtsVocoder`] and [`Qwen3TtsVocoderConfig`].
pub trait Qwen3TtsVocoderMeta {
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

/// Config for [`Qwen3TtsVocoder`]: its convolutions, and each block's config.
#[derive(Config, Debug)]
pub struct Qwen3TtsVocoderConfig {
    /// The latent width, `L`.
    pub latent_width: usize,

    /// The width, `V`: the first block's input.
    pub width: usize,

    /// The blocks, in order, each reading the one before's output.
    pub blocks: Vec<Qwen3TtsVocoderBlockConfig>,

    /// The input convolution's taps.
    #[config(default = 7)]
    pub in_kernel: usize,

    /// The output convolution's taps.
    #[config(default = 7)]
    pub out_kernel: usize,
}

impl Qwen3TtsVocoderMeta for Qwen3TtsVocoderConfig {
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

impl ModuleInit<Qwen3TtsVocoder> for Qwen3TtsVocoderConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a block that
    /// does not read what the one before it (or the input convolution)
    /// writes; as the blocks' builds.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsVocoder> {
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
        Ok(Qwen3TtsVocoder {
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

/// The vocoder: a causal convolution into its width, its blocks, then
/// `SnakeBeta` and a causal convolution to one channel.
///
/// Built by [`Qwen3TtsVocoderConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsVocoder {
    /// `[L, V]`.
    pub input: CausalConv1d,

    /// The blocks, each upsampling and (the release's) halving the width.
    pub blocks: Vec<Qwen3TtsVocoderBlock>,

    /// The activation before the output.
    pub output_act: SnakeActivation,

    /// To one channel.
    pub output: CausalConv1d,
}

impl Qwen3TtsVocoderMeta for Qwen3TtsVocoder {
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

/// Common meta for [`Qwen3TtsCodecDecoder`],
/// [`Qwen3TtsCodecDecoderStructureConfig`] and its policy,
/// [`Qwen3TtsCodecDecoderContractConfig`](super::Qwen3TtsCodecDecoderContractConfig).
pub trait Qwen3TtsCodecDecoderMeta {
    /// The codebooks, `K`.
    fn codebooks(&self) -> usize;

    /// The quantizer's first codebooks, apart from the rest (the semantic
    /// ones).
    fn semantic_codebooks(&self) -> usize;

    /// Codes per codebook.
    fn codebook_size(&self) -> usize;

    /// Audio samples per frame of codes: every upsampler's and vocoder
    /// block's factor, multiplied.
    fn samples_per_frame(&self) -> usize;
}

/// Structure config for [`Qwen3TtsCodecDecoder`]: each part's config.
///
/// The decoder's policy,
/// [`Qwen3TtsCodecDecoderContractConfig`](super::Qwen3TtsCodecDecoderContractConfig),
/// lowers to it.
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecDecoderStructureConfig {
    /// The split quantizer.
    pub quantizer: SplitResidualVectorQuantizerConfig,

    /// The convolution from the quantizer's latent, `[Q, L]`.
    pub pre_conv: CausalConv1dConfig,

    /// The transformer.
    pub pre_transformer: Qwen3TtsCodecTransformerConfig,

    /// The upsamplers.
    pub upsample: Vec<Qwen3TtsCodecUpsampleConfig>,

    /// The vocoder.
    pub decoder: Qwen3TtsVocoderConfig,

    /// The bound the audio is clamped to, `[-c, c]`; `None`, unclamped.
    #[config(default = "Some(1.0)")]
    pub output_clamp: Option<f64>,
}

impl Qwen3TtsCodecDecoderMeta for Qwen3TtsCodecDecoderStructureConfig {
    fn codebooks(&self) -> usize {
        self.quantizer.codebooks
    }

    fn semantic_codebooks(&self) -> usize {
        self.quantizer.first_codebooks
    }

    fn codebook_size(&self) -> usize {
        self.quantizer.size
    }

    fn samples_per_frame(&self) -> usize {
        self.upsample.iter().map(|u| u.factor).product::<usize>() * self.decoder.factor()
    }
}

impl ModuleInit<Qwen3TtsCodecDecoder> for Qwen3TtsCodecDecoderStructureConfig {
    /// # Errors
    /// As the parts' builds.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsCodecDecoder> {
        Ok(Qwen3TtsCodecDecoder {
            quantizer: self.quantizer.try_init(device)?,
            pre_conv: self.pre_conv.try_init(device)?,
            pre_transformer: self.pre_transformer.try_init(device)?,
            upsample: self
                .upsample
                .iter()
                .map(|u| u.try_init(device))
                .collect::<BunsenResult<_>>()?,
            decoder: self.decoder.try_init(device)?,
            samples_per_frame: self.samples_per_frame(),
            output_clamp: self.output_clamp,
        })
    }
}

/// What [`Qwen3TtsCodecDecoder::forward_traced`] computed on the way to the
/// audio, channels first unless said.
#[derive(Clone, Debug)]
pub struct Qwen3TtsCodecTrace {
    /// `[B, Q, T]`: the quantizer's latent.
    pub quantized: Tensor<3>,

    /// `[B, L, T]`: the pre-conv's.
    pub pre_conv: Tensor<3>,

    /// `[B, T, L]`, channels last: the transformer's.
    pub transformer: Tensor<3>,

    /// Each upsampler's, `[B, L, T_i]`.
    pub upsample: Vec<Tensor<3>>,

    /// The vocoder's input convolution's, each block's, the output
    /// activation's and the output convolution's (before the clamp).
    pub vocoder: Vec<Tensor<3>>,
}

/// Qwen3-TTS's 12 Hz codec decoder (`Qwen3TTSTokenizerV2Decoder`): `[B, K, T]`
/// codes at 12.5 Hz to 24 kHz audio, 1920 samples a frame, every step causal.
///
/// ```text
/// z = pre_conv(rvq(codes))                    (split RVQ, then a causal conv of 3 taps)
/// z = transformer(z)                          (windowed causal attention, layer scales)
/// z = upsample_n(… upsample_1(z))             (x2 each: a transposed conv and a ConvNeXt block)
/// y = clamp(vocoder(z), -c, c)                (DAC: SnakeBeta, transposed convs, residual units)
/// ```
///
/// A take longer than a chunk is decoded in chunks with a little left
/// context ([`chunked_decode`](Self::chunked_decode)), as upstream decodes:
/// past the first chunk that is not a whole decode, since the context is
/// shorter than the decoder's reach.
///
/// Built by [`Qwen3TtsCodecDecoderStructureConfig`], which its policy,
/// [`Qwen3TtsCodecDecoderContractConfig`](super::Qwen3TtsCodecDecoderContractConfig),
/// lowers to.
#[derive(Module, Debug)]
pub struct Qwen3TtsCodecDecoder {
    /// The split quantizer: the first codebooks apart from the rest.
    pub quantizer: SplitResidualVectorQuantizer,

    /// `[Q, L]`, 3 taps.
    pub pre_conv: CausalConv1d,

    /// The transformer.
    pub pre_transformer: Qwen3TtsCodecTransformer,

    /// The upsamplers.
    pub upsample: Vec<Qwen3TtsCodecUpsample>,

    /// The vocoder.
    pub decoder: Qwen3TtsVocoder,

    /// Audio samples per frame of codes.
    #[module(skip)]
    pub samples_per_frame: usize,

    /// The bound the audio is clamped to, `[-c, c]`; `None`, unclamped.
    #[module(skip)]
    pub output_clamp: Option<f64>,
}

impl Qwen3TtsCodecDecoderMeta for Qwen3TtsCodecDecoder {
    fn codebooks(&self) -> usize {
        self.quantizer.codebooks()
    }

    fn semantic_codebooks(&self) -> usize {
        self.quantizer.first_codebooks()
    }

    fn codebook_size(&self) -> usize {
        self.quantizer.rvq_first.codebooks[0].size()
    }

    fn samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }
}

impl Qwen3TtsCodecDecoder {
    /// Codes to audio, the whole take at once.
    ///
    /// # Arguments
    /// - `codes`: `[B, K, T]`, `K` the codebooks.
    ///
    /// # Returns
    /// `[B, 1, T · samples_per_frame]`, in `[-c, c]` for an
    /// [`output_clamp`](Self::output_clamp) of `c` (the release's 1).
    pub fn forward(
        &self,
        codes: Tensor<3, Int>,
    ) -> Tensor<3> {
        self.run(codes, None)
    }

    /// [`forward`](Self::forward), and what it computed on the way.
    pub fn forward_traced(
        &self,
        codes: Tensor<3, Int>,
    ) -> (Tensor<3>, Qwen3TtsCodecTrace) {
        let mut trace = None;
        let audio = self.run(codes, Some(&mut trace));
        (audio, trace.expect("a trace"))
    }

    fn run(
        &self,
        codes: Tensor<3, Int>,
        trace: Option<&mut Option<Qwen3TtsCodecTrace>>,
    ) -> Tensor<3> {
        let k = self.quantizer.rvq_first.codebooks.len() + self.quantizer.rvq_rest.codebooks.len();
        let [b, t] =
            unpack_shape_contract!(["B", "K", "T"], &codes.dims(), &["B", "T"], &[("K", k)]);
        let quantized = self.quantizer.decode(codes);
        let pre_conv = self.pre_conv.forward(quantized.clone());
        let transformer = self
            .pre_transformer
            .forward(pre_conv.clone().swap_dims(1, 2));
        let mut upsample = Vec::with_capacity(self.upsample.len());
        let mut z = transformer.clone().swap_dims(1, 2);
        for stage in &self.upsample {
            z = stage.forward(z);
            upsample.push(z.clone());
        }
        let vocoder = &self.decoder;
        let mut steps = Vec::with_capacity(vocoder.blocks.len() + 3);
        let mut y = vocoder.input.forward(z);
        steps.push(y.clone());
        for block in &vocoder.blocks {
            y = block.forward(y);
            steps.push(y.clone());
        }
        y = vocoder.output_act.forward(y);
        steps.push(y.clone());
        y = vocoder.output.forward(y);
        steps.push(y.clone());
        let audio = match self.output_clamp {
            Some(c) => y.clamp(-c, c),
            None => y,
        };
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "C", "N"],
            &audio.dims(),
            &[("B", b), ("C", 1), ("N", t * self.samples_per_frame)]
        );
        if let Some(slot) = trace {
            *slot = Some(Qwen3TtsCodecTrace {
                quantized,
                pre_conv,
                transformer,
                upsample,
                vocoder: steps,
            });
        }
        audio
    }

    /// Codes to audio as upstream decodes a take (`chunked_decode`): chunks
    /// of `chunk` frames, each decoded with up to `context` frames before it
    /// and their audio dropped. Upstream's are 300 and 25
    /// ([`QWEN3_TTS_DECODE_CHUNK`], [`QWEN3_TTS_DECODE_CONTEXT`]).
    ///
    /// The first chunk's audio is the whole decode's; a later one's is not,
    /// when the context is shorter than the decoder's reach (the release's
    /// window alone is 72 frames).
    ///
    /// # Arguments
    /// - `codes`: `[B, K, T]`.
    ///
    /// # Returns
    /// `[B, 1, T · samples_per_frame]`, in `[-c, c]` for an
    /// [`output_clamp`](Self::output_clamp) of `c` (the release's 1).
    ///
    /// # Panics
    /// For a chunk of no frames.
    pub fn chunked_decode(
        &self,
        codes: Tensor<3, Int>,
        chunk: usize,
        context: usize,
    ) -> Tensor<3> {
        assert!(chunk >= 1, "a decode chunk of at least one frame");
        let t = codes.dims()[2];
        let mut parts = Vec::with_capacity(t.div_ceil(chunk));
        let mut start = 0;
        while start < t {
            let end = (start + chunk).min(t);
            // upstream takes the context only when a whole one fits after frame
            // 0
            let left = if start > context { context } else { start };
            let audio = self.forward(codes.clone().slice(s![.., .., start - left..end]));
            let drop = left * self.samples_per_frame;
            let n = audio.dims()[2];
            parts.push(audio.slice(s![.., .., drop..n]));
            start = end;
        }
        Tensor::cat(parts, 2)
    }
}

/// Upstream's decode chunk, in frames (`chunked_decode`'s `chunk_size`).
pub const QWEN3_TTS_DECODE_CHUNK: usize = 300;

/// Upstream's left context per chunk, in frames (`left_context_size`).
pub const QWEN3_TTS_DECODE_CONTEXT: usize = 25;

impl CanonicalKeys for Qwen3TtsCodecDecoder {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("pre_transformer.stack.", &self.pre_transformer.stack)
    }
}

/// The policy of [`Qwen3TtsCodecDecoder`](super::Qwen3TtsCodecDecoder), the
/// 12 Hz codec's decoder (codes in, audio out): its geometry, as a
/// `decoder_config` gives it
/// ([`codec_decoder_config`](crate::kits::speech::tts::qwen3_tts::interop::qwen_tts::Qwen3TtsCodecDecoderMirror::codec_decoder_config)).
///
///
/// Lowers to the [`Qwen3TtsCodecDecoderStructureConfig`] that builds the
/// decoder ([`ToStructureConfig`]): the vocoder's widths halving block by
/// block, the transformer's layers from the geometry.
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecDecoderContractConfig {
    /// The codebooks, `K`; the first
    /// [`semantic_codebooks`](Self::semantic_codebooks) apart from the rest.
    pub codebooks: usize,

    /// The quantizer's first codebooks, apart from the rest (upstream's
    /// `n_q_semantic`).
    #[config(default = 1)]
    pub semantic_codebooks: usize,

    /// Codes per codebook.
    pub codebook_size: usize,

    /// The quantizer's latent width, the pre-conv's input; each codebook's
    /// vectors are half of it.
    pub codebook_dim: usize,

    /// The width the pre-conv writes and the upsamplers keep.
    pub latent_width: usize,

    /// The transformer's width.
    pub width: usize,

    /// Its layers.
    pub n_layers: usize,

    /// Its attention heads.
    pub n_heads: usize,

    /// Its key/value heads.
    pub n_kv_heads: usize,

    /// Its head width.
    pub head_dim: usize,

    /// Its feed-forward's hidden width.
    pub d_ff: usize,

    /// Its causal window, in frames, the frame's own included.
    pub window: usize,

    /// Its rotary base.
    pub rope_theta: f64,

    /// Its RMS norms' epsilon.
    pub rms_eps: f64,

    /// Whether its attention projections carry biases.
    #[config(default = false)]
    pub attention_bias: bool,

    /// The upsamplers' factors, each a transposed convolution and a
    /// `ConvNeXt` block.
    pub upsampling_ratios: Vec<usize>,

    /// The vocoder's blocks' factors.
    pub upsample_rates: Vec<usize>,

    /// The vocoder's first width, halved by each block.
    pub vocoder_width: usize,

    /// Each vocoder block's residual units' dilations (DAC's).
    #[config(default = "alloc::vec![1, 3, 9]")]
    pub vocoder_dilations: Vec<usize>,

    /// Each vocoder block's upsampler's taps per unit of its factor: kernel
    /// `ratio · r` (DAC's 2).
    #[config(default = 2)]
    pub vocoder_kernel_ratio: usize,

    /// The pre-conv's taps.
    #[config(default = 3)]
    pub pre_conv_kernel: usize,

    /// Each upsampler's `ConvNeXt` hidden width per unit of the latent
    /// width: `ratio · L`.
    #[config(default = 4)]
    pub upsample_hidden_ratio: usize,

    /// The transformer's query/key normalization (QK-norm); `None`, none.
    #[config(default = "None")]
    pub qk_norm: Option<BunsenNormalizationConfig>,

    /// The widening the transformer's softmax runs in (`transformers`' eager
    /// attention's f32).
    #[config(default = "Widening::F32")]
    pub softmax_widening: Widening,

    /// The transformer's rotary convention (`rotate_half`'s).
    #[config(default = "RotaryDirection::Forward")]
    pub rotary_direction: RotaryDirection,
}

impl Qwen3TtsCodecDecoderContractConfig {
    /// The transformer's layers: pre-RMS-norm, attention over a causal sliding
    /// window with half-split rotary in [`rotary_direction`](Self::rotary_direction)
    /// and [`qk_norm`](Self::qk_norm) (the release's none; its softmax in
    /// [`softmax_widening`](Self::softmax_widening), the release's f32), a
    /// `SwiGLU` feed-forward without biases, a layer scale on each sublayer,
    /// and a final RMS norm.
    pub fn stack(&self) -> UniformStackConfig {
        let norm = BunsenNormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(self.rms_eps));
        let layer = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, self.width)
                .with_head_dim(Some(self.head_dim))
                .with_norm(self.qk_norm.clone())
                .with_qkv_bias(self.attention_bias)
                .with_output_bias(self.attention_bias)
                .with_sliding_window(Some(self.window))
                .with_softmax_widening(self.softmax_widening)
                .into(),
            BunsenGatedMlpConfig::new(self.width, self.d_ff).into(),
        )
        .with_norm(norm.clone())
        .with_layer_scale(true);
        UniformStackConfig::new(
            self.n_layers,
            layer,
            norm,
            RotaryEmbeddingConfig::new(0, self.head_dim)
                .with_base(self.rope_theta)
                .with_direction(self.rotary_direction),
        )
    }

    /// The vocoder, unrolled: block `i` from `vocoder_width >> i` to half
    /// that, upsampling by `upsample_rates[i]` with a kernel of
    /// `vocoder_kernel_ratio` times it, a residual unit per
    /// `vocoder_dilations`.
    pub fn vocoder(&self) -> Qwen3TtsVocoderConfig {
        let v = self.vocoder_width;
        let blocks = self
            .upsample_rates
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                let out = v >> (i + 1);
                Qwen3TtsVocoderBlockConfig::new(
                    [v >> i, out],
                    r,
                    self.vocoder_kernel_ratio * r,
                    self.vocoder_dilations
                        .iter()
                        .map(|&d| Qwen3TtsVocoderResidualUnitConfig::new(out, d))
                        .collect(),
                )
            })
            .collect();
        Qwen3TtsVocoderConfig::new(self.latent_width, v, blocks)
    }
}

impl Qwen3TtsCodecDecoderMeta for Qwen3TtsCodecDecoderContractConfig {
    fn codebooks(&self) -> usize {
        self.codebooks
    }

    fn semantic_codebooks(&self) -> usize {
        self.semantic_codebooks
    }

    fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    fn samples_per_frame(&self) -> usize {
        self.upsampling_ratios
            .iter()
            .chain(&self.upsample_rates)
            .product()
    }
}

impl ToStructureConfig for Qwen3TtsCodecDecoderContractConfig {
    type Structure = Qwen3TtsCodecDecoderStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<Qwen3TtsCodecDecoderStructureConfig> {
        let latent = self.latent_width;
        Ok(Qwen3TtsCodecDecoderStructureConfig::new(
            SplitResidualVectorQuantizerConfig::new(
                self.codebooks,
                self.codebook_size,
                self.codebook_dim / 2,
                self.codebook_dim,
            )
            .with_first_codebooks(self.semantic_codebooks),
            CausalConv1dConfig::new([self.codebook_dim, latent], self.pre_conv_kernel),
            Qwen3TtsCodecTransformerConfig::new(latent, self.stack()),
            self.upsampling_ratios
                .iter()
                .map(|&r| {
                    Qwen3TtsCodecUpsampleConfig::new(
                        latent,
                        r,
                        CausalConvNext1dConfig::new(latent, self.upsample_hidden_ratio * latent)
                            .with_layer_scale(true),
                    )
                })
                .collect(),
            self.vocoder(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use burn::module::Module;

    use super::*;
    use crate::{
        blocks::transformers::mixers::TokenMixerConfig,
        burner::module::ModuleInit,
        kits::speech::tts::qwen3_tts::{
            codec::{
                Qwen3TtsCodecDecoder,
                Qwen3TtsCodecTransformerMeta,
                Qwen3TtsCodecUpsampleMeta,
                Qwen3TtsVocoderBlockMeta,
                Qwen3TtsVocoderMeta,
                Qwen3TtsVocoderResidualUnitMeta,
            },
            interop::qwen_tts::*,
        },
        support::testing::cpu_device,
    };

    /// A small decoder, every width distinct.
    fn small() -> Qwen3TtsCodecDecoderContractConfig {
        Qwen3TtsCodecDecoderMirror::codec_decoder_config(&Qwen3TtsCodecDecoderMirror {
            codebook_size: 9,
            codebook_dim: 4,
            hidden_size: 8,
            latent_dim: 6,
            rope_theta: 10000.0,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(4),
            attention_bias: false,
            sliding_window: 4,
            intermediate_size: 16,
            hidden_act: "silu".into(),
            rms_norm_eps: 1e-5,
            num_hidden_layers: 1,
            num_quantizers: 3,
            num_semantic_quantizers: 1,
            upsample_rates: vec![2, 3],
            upsampling_ratios: vec![2],
            decoder_dim: 12,
        })
        .expect("a small config")
    }

    /// The policy, its lowering and the decoders built from either answer
    /// the meta alike, and the decoders have the same parameters; every
    /// field with a default is set to another value.
    #[test]
    fn test_lowering_agrees() {
        let device = cpu_device();
        let policy = small()
            .with_semantic_codebooks(2)
            .with_vocoder_dilations(vec![2, 5])
            .with_vocoder_kernel_ratio(3)
            .with_pre_conv_kernel(2)
            .with_upsample_hidden_ratio(3)
            .with_qk_norm(Some(BunsenNormalizationConfig::Rms(RmsNormConfig::new(0))))
            .with_softmax_widening(Widening::NONE)
            .with_rotary_direction(RotaryDirection::Reverse);
        let structure = policy.to_structure();
        let direct: Qwen3TtsCodecDecoder = policy.init(&device);
        let lowered: Qwen3TtsCodecDecoder = structure.init(&device);
        for meta in [
            &policy as &dyn Qwen3TtsCodecDecoderMeta,
            &structure,
            &direct,
            &lowered,
        ] {
            assert_eq!(
                [
                    meta.codebooks(),
                    meta.semantic_codebooks(),
                    meta.codebook_size(),
                    meta.samples_per_frame()
                ],
                [3, 2, 9, 12]
            );
        }
        assert_eq!(direct.num_params(), lowered.num_params());
    }

    /// Each part's config and the part built from it answer its meta alike.
    #[test]
    fn test_part_meta() {
        let device = cpu_device();
        let structure = small().to_structure();
        let decoder: Qwen3TtsCodecDecoder = structure.init(&device);

        let t = &structure.pre_transformer;
        for meta in [
            t as &dyn Qwen3TtsCodecTransformerMeta,
            &decoder.pre_transformer,
        ] {
            assert_eq!([meta.latent_width(), meta.width()], [6, 8]);
        }
        let u = &structure.upsample[0];
        for meta in [u as &dyn Qwen3TtsCodecUpsampleMeta, &decoder.upsample[0]] {
            assert_eq!([meta.width(), meta.factor()], [6, 2]);
        }
        let v = &structure.decoder;
        for meta in [v as &dyn Qwen3TtsVocoderMeta, &decoder.decoder] {
            assert_eq!(
                (meta.latent_width(), meta.width(), meta.factors()),
                (6, 12, vec![2, 3])
            );
            assert_eq!(
                [
                    meta.out_width(),
                    meta.factor(),
                    meta.in_kernel(),
                    meta.out_kernel()
                ],
                [3, 6, 7, 7]
            );
        }
        let b = &structure.decoder.blocks[0];
        for meta in [
            b as &dyn Qwen3TtsVocoderBlockMeta,
            &decoder.decoder.blocks[0],
        ] {
            assert_eq!(
                (
                    meta.channels(),
                    meta.factor(),
                    meta.kernel(),
                    meta.dilations()
                ),
                ([12, 6], 2, 4, vec![1, 3, 9])
            );
        }
        let r = &structure.decoder.blocks[1].residuals[2];
        for meta in [
            r as &dyn Qwen3TtsVocoderResidualUnitMeta,
            &decoder.decoder.blocks[1].residuals[2],
        ] {
            assert_eq!(
                [
                    meta.channels(),
                    meta.dilation(),
                    meta.kernel(),
                    meta.pointwise_kernel()
                ],
                [3, 9, 7, 1]
            );
        }
    }

    /// The vocoder's taps and dilations are its parts' configs': the policy's
    /// dilations and kernel ratio, and each convolution's taps set in the
    /// structure, reach the modules.
    #[test]
    fn test_vocoder_fields_reach_the_parts() {
        let device = cpu_device();
        let mut structure = small()
            .with_vocoder_dilations(vec![2, 5])
            .with_vocoder_kernel_ratio(3)
            .to_structure();
        structure.decoder.in_kernel = 5;
        structure.decoder.out_kernel = 3;
        let unit = &mut structure.decoder.blocks[0].residuals[1];
        unit.kernel = 5;
        unit.pointwise_kernel = 3;
        let decoder: Qwen3TtsCodecDecoder = structure.init(&device);
        let v = &decoder.decoder;
        assert_eq!([v.in_kernel(), v.out_kernel()], [5, 3]);
        assert_eq!(
            (v.blocks[1].kernel(), v.blocks[1].dilations()),
            (9, vec![2, 5])
        );
        let u = &v.blocks[0].residuals[1];
        assert_eq!([u.kernel(), u.pointwise_kernel(), u.dilation()], [5, 3, 5]);
        assert_eq!(v.blocks[0].residuals[0].kernel(), 7);
    }

    /// The policy's pre-conv taps, upsampler hidden ratio and transformer
    /// conventions, and the structure's output clamp, reach the modules.
    #[test]
    fn test_codec_fields_reach_the_parts() {
        let device = cpu_device();
        let policy = small()
            .with_pre_conv_kernel(2)
            .with_upsample_hidden_ratio(3)
            .with_qk_norm(Some(BunsenNormalizationConfig::Rms(RmsNormConfig::new(0))))
            .with_softmax_widening(Widening::NONE)
            .with_rotary_direction(RotaryDirection::Reverse);
        let mut structure = policy.to_structure();
        assert_eq!(structure.output_clamp, Some(1.0));
        structure.output_clamp = Some(0.5);
        let TokenMixerConfig::Attention(attn) = &structure.pre_transformer.stack.layer.attn else {
            panic!("an attention layer");
        };
        assert!(attn.norm.is_some());
        assert_eq!(attn.softmax_widening, Widening::NONE);
        assert_eq!(
            structure.pre_transformer.stack.rotary.direction,
            RotaryDirection::Reverse
        );

        let decoder: Qwen3TtsCodecDecoder = structure.init(&device);
        assert_eq!(decoder.pre_conv.kernel_size(), 2);
        for meta in [
            &structure.upsample[0] as &dyn Qwen3TtsCodecUpsampleMeta,
            &decoder.upsample[0],
        ] {
            assert_eq!([meta.hidden(), meta.block_kernel()], [18, 7]);
        }
        assert_eq!(decoder.output_clamp, Some(0.5));
        let codes = Tensor::<3, Int>::zeros([1, 3, 2], &device);
        let peak = decoder.forward(codes).abs().max().into_scalar::<f32>();
        assert!(peak <= 0.5, "clamped to 0.5: {peak}");
    }

    /// A vocoder whose blocks do not chain, or a block whose residual unit
    /// is not over its output, is refused.
    #[test]
    fn test_vocoder_refusals() {
        let device = cpu_device();
        let vocoder = small().vocoder();
        let mut unchained = vocoder.clone();
        unchained.blocks[1].channels[0] = 5;
        let mut unit = vocoder;
        unit.blocks[0].residuals[0].channels = 5;
        for config in [unchained, unit] {
            assert_eq!(
                config.try_init(&device).unwrap_err().kind(),
                crate::errors::BunsenErrorKind::Illegal
            );
        }
    }
}
