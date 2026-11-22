//! [`Zonos3CodecStageConfig`]: one trunk stage's config.

use burn::config::Config;

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
            projected::{
                ProjectedDecoderStackConfig,
                ProjectedDecoderStackMeta,
            },
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

/// Config of one [`Zonos3Codec`](super::Zonos3Codec) trunk stage, the
/// release's `stages` entry; it lowers to a
/// [`ProjectedDecoderStackConfig`] ([`projected`](Self::projected)).
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

impl Zonos3CodecStageConfig {
    /// The stage as a [`ProjectedDecoderStackConfig`]: [`stack`](Self::stack)
    /// between projections without a bias, the output projection only when
    /// it writes another width.
    pub fn projected(&self) -> ProjectedDecoderStackConfig {
        ProjectedDecoderStackConfig::new(self.input_width, self.stack())
            .with_output_width(self.output_width)
    }
}

impl ProjectedDecoderStackMeta for Zonos3CodecStageConfig {
    fn input_width(&self) -> usize {
        self.input_width
    }

    fn d_model(&self) -> usize {
        self.width
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn output_width(&self) -> usize {
        self.output_width.unwrap_or(self.width)
    }
}
