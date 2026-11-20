//! The codec decoder's config.

use alloc::vec::Vec;

use burn::config::Config;

use super::Zonos3CodecStageMeta;
use crate::blocks::{
    activation::BunsenActivationConfig,
    norm::{
        BunsenLayerNormConfig,
        BunsenNormalizationConfig,
    },
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
};

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
            BunsenLayerNormConfig::new(0).with_epsilon(1e-5),
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
    /// The trunk's last width: the iSTFT head's.
    pub fn head_width(&self) -> usize {
        self.stages
            .last()
            .map_or(self.latent_width, |s| s.out_width())
    }
}
