//! [`Zonos3CodecStage`]: one trunk stage of the codec decoder, and its config.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    tensor::Device,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalizationConfig,
        },
        transformers::{
            attention::csa::CausalSelfAttentionConfig,
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::{
                    DecoderStack,
                    DecoderStackMeta,
                    DecoderStackStructureConfig,
                },
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
                RotaryPairLayout,
            },
            mlp::TransformerMlpConfig,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
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

/// A [`Zonos3CodecStage`]'s geometry, which its config and the stage answer
/// alike.
pub trait Zonos3CodecStageMeta {
    /// The width it reads, `C_in`.
    fn input_width(&self) -> usize;

    /// Its width, `W`.
    fn width(&self) -> usize;

    /// Its layers.
    fn n_layers(&self) -> usize;

    /// The width it writes, `C_out`.
    fn out_width(&self) -> usize;
}

impl Zonos3CodecStageMeta for Zonos3CodecStageConfig {
    fn input_width(&self) -> usize {
        self.input_width
    }

    fn width(&self) -> usize {
        self.width
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn out_width(&self) -> usize {
        self.output_width.unwrap_or(self.width)
    }
}

impl Zonos3CodecStageMeta for Zonos3CodecStage {
    fn input_width(&self) -> usize {
        self.input_proj.weight.dims()[0]
    }

    fn width(&self) -> usize {
        self.input_proj.weight.dims()[1]
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers()
    }

    fn out_width(&self) -> usize {
        self.output_proj
            .as_ref()
            .map_or_else(|| self.width(), |proj| proj.weight.dims()[1])
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
