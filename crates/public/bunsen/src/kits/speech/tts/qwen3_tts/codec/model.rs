//! The codec decoder's modules.

use alloc::vec::Vec;

use burn::{
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::*,
};

use super::Qwen3TtsCodecDecoderConfig;
use crate::{
    blocks::{
        activation::{
            SnakeActivation,
            SnakeActivationConfig,
        },
        conv::{
            CausalConv1d,
            CausalConv1dConfig,
            CausalConvNext1d,
            CausalConvNext1dConfig,
            CausalConvTranspose1d,
            CausalConvTranspose1dConfig,
        },
        quantization::{
            SplitResidualVectorQuantizer,
            SplitResidualVectorQuantizerConfig,
        },
        transformers::decoders::stack::DecoderStack,
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    contracts::unpack_shape_contract,
    errors::BunsenResult,
};

/// The transformer between the pre-conv and the upsamplers: a projection
/// in, a decoder stack, a projection out.
#[derive(Module, Debug)]
pub struct Qwen3TtsCodecTransformer {
    /// `[L, W]`, with a bias.
    pub input_proj: Linear,

    /// The layers and the final norm.
    pub stack: DecoderStack,

    /// `[W, L]`, with a bias.
    pub output_proj: Linear,
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

/// One upsampler: a causal transposed convolution (kernel and stride its
/// factor), then a causal `ConvNeXt` block with a layer scale.
#[derive(Module, Debug)]
pub struct Qwen3TtsCodecUpsample {
    /// The transposed convolution.
    pub up: CausalConvTranspose1d,

    /// The `ConvNeXt` block, channels last.
    pub block: CausalConvNext1d,
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

/// A vocoder residual unit: `SnakeBeta`, a dilated causal convolution of 7
/// taps, `SnakeBeta`, a pointwise convolution, and the input added.
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

/// A vocoder block (DAC's): `SnakeBeta`, a causal transposed convolution
/// upsampling by `r` (kernel `2 r`) and halving the width, then residual
/// units of dilation 1, 3 and 9.
#[derive(Module, Debug)]
pub struct Qwen3TtsVocoderBlock {
    /// The activation before the upsampler.
    pub act: SnakeActivation,

    /// The upsampler.
    pub upsample: CausalConvTranspose1d,

    /// The residual units.
    pub residuals: Vec<Qwen3TtsVocoderResidualUnit>,
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

/// The vocoder: a causal convolution into its width, its blocks, then
/// `SnakeBeta` and a causal convolution to one channel.
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
/// Built by [`Qwen3TtsCodecDecoderConfig`].
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

impl ModuleInit<Qwen3TtsCodecDecoder> for Qwen3TtsCodecDecoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsCodecDecoder> {
        let conv = |c_in, c_out, k, d| {
            CausalConv1dConfig::new([c_in, c_out], k)
                .with_dilation(d)
                .init(device)
        };
        let snake = |c| SnakeActivationConfig::snake_beta_log_scale(c).init(device);
        let latent = self.latent_width;
        let upsample = self
            .upsampling_ratios
            .iter()
            .map(|&r| {
                Ok(Qwen3TtsCodecUpsample {
                    up: CausalConvTranspose1dConfig::new([latent, latent], r)
                        .with_stride(r)
                        .init(device),
                    block: CausalConvNext1dConfig::new(latent, 4 * latent)
                        .with_layer_scale(true)
                        .try_init(device)?,
                })
            })
            .collect::<BunsenResult<_>>()?;
        let blocks = self
            .vocoder_widths()
            .into_iter()
            .zip(&self.upsample_rates)
            .map(|([c_in, c_out], &r)| Qwen3TtsVocoderBlock {
                act: snake(c_in),
                upsample: CausalConvTranspose1dConfig::new([c_in, c_out], 2 * r)
                    .with_stride(r)
                    .init(device),
                residuals: QWEN3_TTS_VOCODER_DILATIONS
                    .iter()
                    .map(|&d| Qwen3TtsVocoderResidualUnit {
                        act1: snake(c_out),
                        conv1: conv(c_out, c_out, 7, d),
                        act2: snake(c_out),
                        conv2: conv(c_out, c_out, 1, 1),
                    })
                    .collect(),
            })
            .collect();
        let out = self.vocoder_out_width();
        Ok(Qwen3TtsCodecDecoder {
            quantizer: SplitResidualVectorQuantizerConfig::new(
                self.codebooks,
                self.codebook_size,
                self.codebook_dim / 2,
                self.codebook_dim,
            )
            .init(device),
            pre_conv: conv(self.codebook_dim, latent, 3, 1),
            pre_transformer: Qwen3TtsCodecTransformer {
                input_proj: LinearConfig::new(latent, self.width).init(device),
                stack: self.stack().try_init(device)?,
                output_proj: LinearConfig::new(self.width, latent).init(device),
            },
            upsample,
            decoder: Qwen3TtsVocoder {
                input: conv(latent, self.vocoder_width, 7, 1),
                blocks,
                output_act: snake(out),
                output: conv(out, 1, 7, 1),
            },
            samples_per_frame: self.samples_per_frame(),
        })
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
