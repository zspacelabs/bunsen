//! [`Qwen3TtsCodecTransformer`]: the codec decoder's transformer, and its config.

use burn::{
    config::Config,
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::*,
};

use crate::{
    blocks::transformers::decoders::{
        causal::CausalDecoderBlockMeta,
        stack::{
            DecoderStack,
            UniformStackConfig,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

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
