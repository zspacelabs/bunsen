//! The codec decoder.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    Tensor,
    module::{
        Module,
        Param,
    },
    nn::{
        Initializer,
        LayerNormConfig,
        Linear,
        LinearConfig,
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    prelude::{
        Int,
        s,
    },
    tensor::{
        DType,
        Device,
    },
};

use super::{
    Zonos3CodecConfig,
    Zonos3CodecStageConfig,
};
use crate::{
    blocks::{
        conv::{
            CausalConvNext1d,
            CausalConvNext1dConfig,
        },
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
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
    ops::signal::{
        InverseStft,
        InverseStftConfig,
    },
};

/// One codebook's decoder: each code's vector, projected.
#[derive(Module, Debug)]
pub struct Zonos3CodebookDecoder {
    /// `[S, d]`: the codebook.
    pub codebook: Param<Tensor<2>>,

    /// The projection, `[d, W]`, with a bias.
    pub out_proj: Linear,
}

impl Zonos3CodebookDecoder {
    /// `[B, T]` codes to `[B, T, W]`.
    pub fn forward(
        &self,
        codes: Tensor<2, Int>,
    ) -> Tensor<3> {
        let [b, t] = codes.dims();
        let d = self.codebook.dims()[1];
        let vectors = self
            .codebook
            .val()
            .select(0, codes.reshape([b * t]))
            .reshape([b, t, d]);
        self.out_proj.forward(vectors)
    }
}

/// One trunk stage: a projection in, a decoder stack, a projection out.
#[derive(Module, Debug)]
pub struct Zonos3CodecStage {
    /// `[C_in, W]`, without a bias.
    pub input_proj: Linear,

    /// The layers, with no final norm.
    pub stack: DecoderStack,

    /// `[W, C_out]`, without a bias; `None`, the stage writes its width.
    pub output_proj: Option<Linear>,
}

impl Zonos3CodecStage {
    /// `[B, T, C_in]` to `[B, T, C_out]`, positions from 0.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let t = x.dims()[1];
        let x = self.input_proj.forward(x);
        let x = self
            .stack
            .forward_hidden(x.clone(), &self.stack.rotary(t, &x.device()), None);
        match &self.output_proj {
            Some(proj) => proj.forward(x),
            None => x,
        }
    }
}

impl ModuleInit<Zonos3CodecStage> for Zonos3CodecStageConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Zonos3CodecStage> {
        let linear = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        Ok(Zonos3CodecStage {
            input_proj: linear(self.input_width, self.width),
            stack: self.stack().try_init(device)?,
            output_proj: self.output_width.map(|o| linear(self.width, o)),
        })
    }
}

impl ModuleInit<Zonos3Codec> for Zonos3CodecConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Zonos3Codec> {
        let codebook = |width| Zonos3CodebookDecoder {
            codebook: Initializer::Normal {
                mean: 0.0,
                std: 1.0,
            }
            .init([self.codebook_size, self.codebook_dim], device),
            out_proj: LinearConfig::new(self.codebook_dim, width).init(device),
        };
        let convnext = |width, hidden| CausalConvNext1dConfig::new(width, hidden).try_init(device);
        let head = self.head_width();
        Ok(Zonos3Codec {
            quantizers: core::iter::once(codebook(self.semantic_width))
                .chain((0..self.residual_codebooks).map(|_| codebook(self.latent_width)))
                .collect(),
            semantic_blocks: (0..self.semantic_depth)
                .map(|_| convnext(self.semantic_width, self.semantic_hidden))
                .collect::<BunsenResult<_>>()?,
            semantic_proj: LinearConfig::new(self.semantic_width, self.latent_width).init(device),
            stages: self
                .stages
                .iter()
                .map(|s| s.try_init(device))
                .collect::<BunsenResult<_>>()?,
            head_blocks: (0..self.head_depth)
                .map(|_| convnext(head, self.head_hidden))
                .collect::<BunsenResult<_>>()?,
            head_norm: BunsenNormalizationConfig::Layer(
                LayerNormConfig::new(head).with_epsilon(1e-6),
            )
            .init(device),
            head_outs: self
                .n_ffts
                .iter()
                .map(|&n| LinearConfig::new(head, n + 2).init(device))
                .collect(),
            istfts: self
                .n_ffts
                .iter()
                .map(|&n| InverseStftConfig::new(n, self.head_hop).try_init(device))
                .collect::<BunsenResult<_>>()?,
            mixer: Conv1dConfig::new(self.n_ffts.len(), 1, self.fir_kernel)
                .with_padding(PaddingConfig1d::Explicit(
                    self.fir_kernel / 2,
                    self.fir_kernel / 2,
                ))
                .with_bias(false)
                .init(device),
            resampler: Conv1dConfig::new(
                1,
                self.resample_new,
                2 * self.resample_width + self.resample_orig,
            )
            .with_stride(self.resample_orig)
            .with_padding(PaddingConfig1d::Valid)
            .with_bias(false)
            .init(device),
            head_hop: self.head_hop,
            resample_width: self.resample_width,
            hop: self.hop,
        })
    }
}

/// What [`Zonos3Codec::forward_traced`] computed on the way to the audio.
#[derive(Clone, Debug)]
pub struct Zonos3CodecTrace {
    /// `[B, T, L]`: the summed codebook latents, the trunk's input.
    pub latent: Tensor<3>,

    /// Each trunk step's output, `[B, T_i, C_i]`: each stage's, and each
    /// patch-up's between them.
    pub trunk: Vec<Tensor<3>>,

    /// Each iSTFT branch's waveform at the internal rate, `[B, N]`.
    pub branches: Vec<Tensor<2>>,

    /// `[B, N]`: the branches mixed.
    pub mixed: Tensor<2>,
}

/// ZONOS3-Mini's codec decoder (the `keep2_plain` decoder of its dual
/// codec): `[B, Q, T]` codes at 15 Hz to 44.1 kHz audio, `hop` samples per
/// frame.
///
/// ```text
/// z    = proj(convnext*(cb_0(codes_0))) + Σ_i cb_i(codes_i)   (the latent)
/// x    = stage_n(patch(… patch(stage_1(z))))                 (the trunk, 2x per patch-up)
/// y_b  = istft_b(exp(log_mag_b) · e^{i phase_b})              (per FFT size, from convnext*, norm, a linear)
/// out  = resample(mixer([y_1, y_2]))
/// ```
///
/// The trunk's stages are decoder stacks of pre-LayerNorm layers with
/// windowed causal attention and layer scales; the patch-up turns
/// `[T, 2 C]` into `[2 T, C]`, channel `2 c + j` to time `2 t + j`. The
/// head, the mixer and the resampler run in f32.
///
/// Built by [`Zonos3CodecConfig`].
#[derive(Module, Debug)]
pub struct Zonos3Codec {
    /// Each codebook's decoder: the first into the semantic head's width,
    /// the rest into the latent's.
    pub quantizers: Vec<Zonos3CodebookDecoder>,

    /// The semantic head's blocks, over the first codebook's projection.
    pub semantic_blocks: Vec<CausalConvNext1d>,

    /// The semantic head's projection to the latent, with a bias.
    pub semantic_proj: Linear,

    /// The trunk's stages.
    pub stages: Vec<Zonos3CodecStage>,

    /// The iSTFT head's blocks.
    pub head_blocks: Vec<CausalConvNext1d>,

    /// The iSTFT head's norm.
    pub head_norm: BunsenNormalization,

    /// Each branch's log-magnitude and phase, `[C, N_b + 2]`.
    pub head_outs: Vec<Linear>,

    /// Each branch's synthesis.
    pub istfts: Vec<InverseStft>,

    /// The branch mixer: a centred FIR over the branches, `[1, n_b, K]`.
    pub mixer: Conv1d,

    /// The polyphase resampler, `[new, 1, 2 w + orig]`, stride `orig`.
    pub resampler: Conv1d,

    /// The iSTFT hop.
    #[module(skip)]
    pub head_hop: usize,

    /// The resampler's half-width beyond its stride.
    #[module(skip)]
    pub resample_width: usize,

    /// Output samples per frame of codes.
    #[module(skip)]
    pub hop: usize,
}

/// `[B, T, 2 C]` to `[B, 2 T, C]`: channel `2 c + j` of time `t` to channel
/// `c` of time `2 t + j`.
pub fn patch_up(x: Tensor<3>) -> Tensor<3> {
    let [b, t, c] = x.dims();
    x.reshape([b, t, c / 2, 2])
        .swap_dims(2, 3)
        .reshape([b, 2 * t, c / 2])
}

impl Zonos3Codec {
    /// The audio of `codes`.
    ///
    /// # Arguments
    /// - `codes`: `[B, Q, T]`, the first codebook's codes, then the residual
    ///   codebooks'.
    ///
    /// # Returns
    /// - `[B, T hop]`.
    pub fn forward(
        &self,
        codes: Tensor<3, Int>,
    ) -> Tensor<2> {
        self.forward_traced(codes).0
    }

    /// [`forward`](Self::forward), with what it computed on the way.
    pub fn forward_traced(
        &self,
        codes: Tensor<3, Int>,
    ) -> (Tensor<2>, Zonos3CodecTrace) {
        let [b, t] = unpack_shape_contract!(
            ["B", "Q", "T"],
            &codes.dims(),
            &["B", "T"],
            &[("Q", self.quantizers.len())]
        );
        let codebook = |i: usize| codes.clone().slice(s![.., i..i + 1, ..]).reshape([b, t]);

        let semantic = self
            .semantic_blocks
            .iter()
            .fold(self.quantizers[0].forward(codebook(0)), |x, block| {
                block.forward(x)
            });
        let latent = self
            .quantizers
            .iter()
            .enumerate()
            .skip(1)
            .fold(self.semantic_proj.forward(semantic), |z, (i, q)| {
                z + q.forward(codebook(i))
            });

        let mut trunk = Vec::with_capacity(2 * self.stages.len());
        let mut x = latent.clone();
        for (i, stage) in self.stages.iter().enumerate() {
            if i > 0 {
                x = patch_up(x);
                trunk.push(x.clone());
            }
            x = stage.forward(x);
            trunk.push(x.clone());
        }

        let frames = x.dims()[1];
        let h = self.head_blocks.iter().fold(x, |h, block| block.forward(h));
        let h = self.head_norm.forward(h).cast(DType::F32);
        let branches: Vec<Tensor<2>> = self
            .head_outs
            .iter()
            .zip(&self.istfts)
            .map(|(out, istft)| {
                let o = out.forward(h.clone()).cast(DType::F32);
                let bins = o.dims()[2] / 2;
                let magnitude = o
                    .clone()
                    .slice(s![.., .., ..bins])
                    .clamp_max(100f64.ln())
                    .exp();
                let phase = o.slice(s![.., .., bins..]);
                istft.istft(
                    (magnitude.clone() * phase.clone().cos()).swap_dims(1, 2),
                    (magnitude * phase.sin()).swap_dims(1, 2),
                    Some(frames * self.head_hop),
                )
            })
            .collect();
        let mixed = self
            .mixer
            .forward(Tensor::stack(branches.clone(), 1))
            .squeeze_dim::<2>(1);
        let audio = self.resample(mixed.clone());
        let audio = audio.slice(s![.., ..t * self.hop]);
        (
            audio,
            Zonos3CodecTrace {
                latent,
                trunk,
                branches,
                mixed,
            },
        )
    }

    /// The resampler: `[B, N]` padded by `w` before and `w + orig` after,
    /// strided through the polyphase kernel, its phases interleaved, cut to
    /// `ceil(new N / orig)`.
    pub fn resample(
        &self,
        x: Tensor<2>,
    ) -> Tensor<2> {
        let [b, n] = x.dims();
        let [new, _, k] = self.resampler.weight.dims();
        let w = self.resample_width;
        let orig = k - 2 * w;
        let (device, dtype) = (x.device(), x.dtype());
        let zeros = |len| Tensor::<2>::zeros([b, len], (&device, dtype));
        let padded = Tensor::cat(vec![zeros(w), x, zeros(w + orig)], 1);
        let y = self
            .resampler
            .forward(padded.unsqueeze_dim(1))
            .swap_dims(1, 2);
        let steps = y.dims()[1];
        y.reshape([b, steps * new])
            .slice(s![.., ..(new * n).div_ceil(orig)])
    }
}

impl CanonicalKeys for Zonos3Codec {
    fn canonical_rekey(&self) -> Rekey {
        self.stages
            .iter()
            .enumerate()
            .fold(Rekey::keep_rest(), |rekey, (i, stage)| {
                rekey.mount_canonical(format!("stages.{i}.stack."), &stage.stack)
            })
    }
}
