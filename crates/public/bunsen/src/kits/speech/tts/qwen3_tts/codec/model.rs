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
            CausalConvTranspose1d,
            CausalConvTranspose1dConfig,
            CausalConvTranspose1dMeta,
        },
        quantization::{
            EuclideanCodebookMeta,
            SplitResidualVectorQuantizer,
            SplitResidualVectorQuantizerConfig,
            SplitResidualVectorQuantizerMeta,
        },
        transformers::decoders::{
            causal::CausalDecoderBlockMeta,
            stack::{
                DecoderStack,
                UniformStackConfig,
            },
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
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
}

/// Config for [`Qwen3TtsCodecUpsample`].
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecUpsampleConfig {
    /// The width, `L`.
    pub width: usize,

    /// The factor, `r`.
    pub factor: usize,
}

impl Qwen3TtsCodecUpsampleMeta for Qwen3TtsCodecUpsampleConfig {
    fn width(&self) -> usize {
        self.width
    }

    fn factor(&self) -> usize {
        self.factor
    }
}

impl ModuleInit<Qwen3TtsCodecUpsample> for Qwen3TtsCodecUpsampleConfig {
    /// # Errors
    /// As the convolutions' builds: a factor of 0 is refused.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsCodecUpsample> {
        let [w, r] = [self.width, self.factor];
        Ok(Qwen3TtsCodecUpsample {
            up: CausalConvTranspose1dConfig::new([w, w], r)
                .with_stride(r)
                .try_init(device)?,
            block: CausalConvNext1dConfig::new(w, 4 * w)
                .with_layer_scale(true)
                .try_init(device)?,
        })
    }
}

/// One upsampler: a causal transposed convolution (kernel and stride its
/// factor), then a causal `ConvNeXt` block with a layer scale.
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
}

/// Config for [`Qwen3TtsVocoderResidualUnit`].
#[derive(Config, Debug)]
pub struct Qwen3TtsVocoderResidualUnitConfig {
    /// The channels, `C`.
    pub channels: usize,

    /// The dilated convolution's dilation.
    pub dilation: usize,
}

impl Qwen3TtsVocoderResidualUnitMeta for Qwen3TtsVocoderResidualUnitConfig {
    fn channels(&self) -> usize {
        self.channels
    }

    fn dilation(&self) -> usize {
        self.dilation
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
            conv1: causal_conv([c, c], 7, self.dilation, device)?,
            act2: snake(c, device)?,
            conv2: causal_conv([c, c], 1, 1, device)?,
        })
    }
}

/// A vocoder residual unit: `SnakeBeta`, a dilated causal convolution of 7
/// taps, `SnakeBeta`, a pointwise convolution, and the input added.
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
}

/// Config for [`Qwen3TtsVocoderBlock`]; its residual units' dilations are
/// [`QWEN3_TTS_VOCODER_DILATIONS`].
#[derive(Config, Debug)]
pub struct Qwen3TtsVocoderBlockConfig {
    /// The channels, `[C_in, C_out]`.
    pub channels: [usize; 2],

    /// The factor, `r`.
    pub factor: usize,
}

impl Qwen3TtsVocoderBlockMeta for Qwen3TtsVocoderBlockConfig {
    fn channels(&self) -> [usize; 2] {
        self.channels
    }

    fn factor(&self) -> usize {
        self.factor
    }
}

impl ModuleInit<Qwen3TtsVocoderBlock> for Qwen3TtsVocoderBlockConfig {
    /// # Errors
    /// As the convolutions' builds: a factor of 0 is refused.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsVocoderBlock> {
        let [c_in, c_out] = self.channels;
        let r = self.factor;
        Ok(Qwen3TtsVocoderBlock {
            act: snake(c_in, device)?,
            upsample: CausalConvTranspose1dConfig::new([c_in, c_out], 2 * r)
                .with_stride(r)
                .try_init(device)?,
            residuals: QWEN3_TTS_VOCODER_DILATIONS
                .iter()
                .map(|&d| Qwen3TtsVocoderResidualUnitConfig::new(c_out, d).try_init(device))
                .collect::<BunsenResult<_>>()?,
        })
    }
}

/// A vocoder block (DAC's): `SnakeBeta`, a causal transposed convolution
/// upsampling by `r` (kernel `2 r`) and halving the width, then residual
/// units of dilation 1, 3 and 9.
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
}

impl Qwen3TtsVocoderBlock {
    /// `[B, C, T]` to `[B, C / 2, r T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let x = self.upsample.forward(self.act.forward(x));
        self.residuals.iter().fold(x, |x, unit| unit.forward(x))
    }
}

/// The dilations of a vocoder block's residual units.
pub const QWEN3_TTS_VOCODER_DILATIONS: [usize; 3] = [1, 3, 9];

/// Common meta for [`Qwen3TtsVocoder`] and [`Qwen3TtsVocoderConfig`].
pub trait Qwen3TtsVocoderMeta {
    /// The latent width, `L`: its input's.
    fn latent_width(&self) -> usize;

    /// Its width, `V`: its first block's input, halved by each block.
    fn width(&self) -> usize;

    /// Each block's factor.
    fn factors(&self) -> Vec<usize>;

    /// Its last width, its output convolution's input: `V` halved by each
    /// block.
    fn out_width(&self) -> usize {
        self.width() >> self.factors().len()
    }

    /// Output samples per input frame: the blocks' factors, multiplied.
    fn factor(&self) -> usize {
        self.factors().iter().product()
    }
}

/// Config for [`Qwen3TtsVocoder`].
#[derive(Config, Debug)]
pub struct Qwen3TtsVocoderConfig {
    /// The latent width, `L`.
    pub latent_width: usize,

    /// The width, `V`.
    pub width: usize,

    /// Each block's factor.
    pub factors: Vec<usize>,
}

impl Qwen3TtsVocoderMeta for Qwen3TtsVocoderConfig {
    fn latent_width(&self) -> usize {
        self.latent_width
    }

    fn width(&self) -> usize {
        self.width
    }

    fn factors(&self) -> Vec<usize> {
        self.factors.clone()
    }
}

impl ModuleInit<Qwen3TtsVocoder> for Qwen3TtsVocoderConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a width its
    /// blocks do not halve evenly; as the blocks' builds.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsVocoder> {
        let n = self.factors.len() as u32;
        if !self.width.is_multiple_of(1 << n) {
            return Err(BunsenError::illegal(format!(
                "a vocoder width of {} over {n} halving blocks",
                self.width
            )));
        }
        let v = self.width;
        let out = self.out_width();
        Ok(Qwen3TtsVocoder {
            input: causal_conv([self.latent_width, v], 7, 1, device)?,
            blocks: self
                .factors
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    Qwen3TtsVocoderBlockConfig::new([v >> i, v >> (i + 1)], r).try_init(device)
                })
                .collect::<BunsenResult<_>>()?,
            output_act: snake(out, device)?,
            output: causal_conv([out, 1], 7, 1, device)?,
        })
    }
}

/// The vocoder: a causal convolution into its width, its blocks, then
/// `SnakeBeta` and a causal convolution to one channel.
///
/// Built by [`Qwen3TtsVocoderConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsVocoder {
    /// `[L, V]`, 7 taps.
    pub input: CausalConv1d,

    /// The blocks, each upsampling and halving the width.
    pub blocks: Vec<Qwen3TtsVocoderBlock>,

    /// The activation before the output.
    pub output_act: SnakeActivation,

    /// To one channel, 7 taps.
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
}

/// Common meta for [`Qwen3TtsCodecDecoder`],
/// [`Qwen3TtsCodecDecoderStructureConfig`] and its policy,
/// [`Qwen3TtsCodecDecoderContractConfig`](super::Qwen3TtsCodecDecoderContractConfig).
pub trait Qwen3TtsCodecDecoderMeta {
    /// The codebooks, `K`.
    fn codebooks(&self) -> usize;

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
}

impl Qwen3TtsCodecDecoderMeta for Qwen3TtsCodecDecoderStructureConfig {
    fn codebooks(&self) -> usize {
        self.quantizer.codebooks
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
/// y = clamp(vocoder(z), -1, 1)                (DAC: SnakeBeta, transposed convs, residual units)
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
    /// The split quantizer: the first codebook apart from the rest.
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
}

impl Qwen3TtsCodecDecoderMeta for Qwen3TtsCodecDecoder {
    fn codebooks(&self) -> usize {
        self.quantizer.codebooks()
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
    /// `[B, 1, T · samples_per_frame]`, in `[-1, 1]`.
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
        let audio = y.clamp(-1.0, 1.0);
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
    /// `[B, 1, T · samples_per_frame]`, in `[-1, 1]`.
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
