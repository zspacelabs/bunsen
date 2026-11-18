//! The codec decoder's config.

use alloc::vec::Vec;

use burn::{
    config::Config,
    nn::LayerNormConfig,
};
use serde::Deserialize;

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        norm::BunsenNormalizationConfig,
        transformers::{
            attention::csa::CausalSelfAttentionConfig,
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::DecoderStackStructureConfig,
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
                RotaryPairLayout,
            },
            mlp::TransformerMlpConfig,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// One trunk stage of `config.json`'s `codec` section, under upstream's
/// names.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Zonos3CodecStageJson {
    /// The causal sliding window, in frames, the frame's own included.
    pub context_frames: usize,
    /// The stage's width.
    pub d_model: usize,
    /// Its feed-forward's hidden width.
    pub dim_feedforward: usize,
    /// The width it reads.
    pub input_dimension: usize,
    /// Its attention heads.
    pub num_heads: usize,
    /// Its layers.
    pub num_layers: usize,
    /// The width it writes; `None`, its own (no output projection).
    pub output_dimension: Option<usize>,
}

/// The `codec` section of `Zyphra/ZONOS3-Mini-hf`'s `config.json` (the
/// `keep2_plain` decoder), under upstream's names.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Zonos3CodecJson {
    /// Each codebook vector's width.
    pub codebook_dim: usize,
    /// Codes per codebook.
    pub codebook_size: usize,
    /// The branch mixer's taps.
    pub fir_kernel: usize,
    /// The iSTFT head's `ConvNeXt` blocks.
    pub head_depth: usize,
    /// Their hidden width.
    pub head_hidden: usize,
    /// The iSTFT hop, at the internal rate.
    pub head_hop: usize,
    /// Output samples per frame of codes.
    pub hop: usize,
    /// The latent's width.
    pub latent_dim: usize,
    /// The iSTFT branches' FFT sizes.
    pub n_ffts: Vec<usize>,
    /// The resampler's output phases.
    pub resample_new: usize,
    /// Its input stride.
    pub resample_orig: usize,
    /// Its kernel's half-width beyond the stride.
    pub resample_width: usize,
    /// The codebooks after the first.
    pub residual_codebooks: usize,
    /// The semantic head's `ConvNeXt` blocks.
    pub semantic_depth: usize,
    /// Their hidden width.
    pub semantic_hidden: usize,
    /// The first codebook's projection width, the semantic head's.
    pub semantic_width: usize,
    /// The trunk's stages.
    pub stages: Vec<Zonos3CodecStageJson>,
}

/// Config of one [`Zonos3Codec`](super::Zonos3Codec) trunk stage.
#[derive(Config, Debug)]
pub struct Zonos3CodecStageConfig {
    /// The width it reads.
    pub input_width: usize,

    /// Its width.
    pub width: usize,

    /// Its layers.
    pub n_layers: usize,

    /// Its attention heads.
    pub n_heads: usize,

    /// Its feed-forward's expansion over its width.
    pub expansion: usize,

    /// The causal sliding window, in frames, the frame's own included.
    pub window: usize,

    /// The width it writes; `None`, its own (no output projection).
    #[config(default = "None")]
    pub output_width: Option<usize>,
}

impl Zonos3CodecStageConfig {
    /// The width it writes.
    pub fn out_width(&self) -> usize {
        self.output_width.unwrap_or(self.width)
    }

    /// Its layers, as a decoder stack without a final norm: pre-LayerNorm,
    /// multi-head attention over a causal sliding window with interleaved
    /// rotary and no QK-norm, a GELU feed-forward, no biases on the
    /// projections, and a layer scale on each sublayer.
    pub fn stack(&self) -> DecoderStackStructureConfig {
        let layer = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_heads, self.n_heads, self.width)
                .with_norm(None)
                .with_sliding_window(Some(self.window))
                .into(),
            TransformerMlpConfig::new(self.width)
                .with_expansion_factor(self.expansion)
                .with_activation(BunsenActivationConfig::Gelu)
                .into(),
        )
        .with_norm(BunsenNormalizationConfig::Layer(
            LayerNormConfig::new(0).with_epsilon(1e-5),
        ))
        .with_layer_scale(true);
        let head_dim = self.width / self.n_heads;
        DecoderStackStructureConfig::new(
            vec![layer; self.n_layers],
            BunsenNormalizationConfig::Identity,
            RotaryEmbeddingConfig::new(0, head_dim)
                .with_direction(RotaryDirection::Forward)
                .with_pair_layout(RotaryPairLayout::Interleaved),
        )
    }
}

/// Config for [`Zonos3Codec`](super::Zonos3Codec): ZONOS3-Mini's codec
/// decoder, codes in, audio out.
#[derive(Config, Debug)]
pub struct Zonos3CodecConfig {
    /// Codes per codebook.
    pub codebook_size: usize,

    /// Each codebook vector's width.
    pub codebook_dim: usize,

    /// The codebooks after the first.
    pub residual_codebooks: usize,

    /// The first codebook's projection width, the semantic head's.
    pub semantic_width: usize,

    /// The semantic head's `ConvNeXt` blocks.
    pub semantic_depth: usize,

    /// Their hidden width.
    pub semantic_hidden: usize,

    /// The latent's width, the trunk's input.
    pub latent_width: usize,

    /// The trunk's stages; a 2x patch-up between each two.
    pub stages: Vec<Zonos3CodecStageConfig>,

    /// The iSTFT head's `ConvNeXt` blocks.
    pub head_depth: usize,

    /// Their hidden width.
    pub head_hidden: usize,

    /// The iSTFT branches' FFT sizes.
    pub n_ffts: Vec<usize>,

    /// The iSTFT hop, at the internal rate.
    pub head_hop: usize,

    /// The branch mixer's taps (odd: centred).
    pub fir_kernel: usize,

    /// The resampler's input stride.
    pub resample_orig: usize,

    /// Its output phases.
    pub resample_new: usize,

    /// Its kernel's half-width beyond the stride.
    pub resample_width: usize,

    /// Output samples per frame of codes.
    pub hop: usize,
}

impl Zonos3CodecConfig {
    /// The config of a `codec` section.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// stage whose feed-forward is not a whole multiple of its width, whose
    /// width is not a whole number of heads, or whose input is not its
    /// predecessor's output halved by the patch-up; or an even mixer.
    pub fn from_json(json: &Zonos3CodecJson) -> BunsenResult<Self> {
        let mut stages = Vec::with_capacity(json.stages.len());
        let mut reads = json.latent_dim;
        for (i, s) in json.stages.iter().enumerate() {
            if !s.dim_feedforward.is_multiple_of(s.d_model)
                || !s.d_model.is_multiple_of(s.num_heads)
            {
                return Err(BunsenError::unsupported(format!(
                    "codec stage {i}: d_model {}, {} heads, dim_feedforward {}",
                    s.d_model, s.num_heads, s.dim_feedforward
                )));
            }
            if s.input_dimension != reads {
                return Err(BunsenError::unsupported(format!(
                    "codec stage {i} reads {}, after {reads}",
                    s.input_dimension
                )));
            }
            let stage = Zonos3CodecStageConfig::new(
                s.input_dimension,
                s.d_model,
                s.num_layers,
                s.num_heads,
                s.dim_feedforward / s.d_model,
                s.context_frames,
            )
            .with_output_width(s.output_dimension);
            reads = stage.out_width() / 2;
            stages.push(stage);
        }
        if json.fir_kernel.is_multiple_of(2) {
            return Err(BunsenError::unsupported(format!(
                "an even mixer of {} taps",
                json.fir_kernel
            )));
        }
        Ok(Self::new(
            json.codebook_size,
            json.codebook_dim,
            json.residual_codebooks,
            json.semantic_width,
            json.semantic_depth,
            json.semantic_hidden,
            json.latent_dim,
            stages,
            json.head_depth,
            json.head_hidden,
            json.n_ffts.clone(),
            json.head_hop,
            json.fir_kernel,
            json.resample_orig,
            json.resample_new,
            json.resample_width,
            json.hop,
        ))
    }

    /// The trunk's last width: the iSTFT head's.
    pub fn head_width(&self) -> usize {
        self.stages
            .last()
            .map_or(self.latent_width, |s| s.out_width())
    }
}
