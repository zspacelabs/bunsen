//! The talker's modules.

use burn::{
    nn::{
        Embedding,
        EmbeddingConfig,
        Linear,
        LinearConfig,
    },
    prelude::*,
    tensor::activation::silu,
};

use super::Qwen3TtsTalkerConfig;
use crate::{
    blocks::transformers::{
        embedding::RotarySet,
        models::language_decoder::{
            LmDecoderMeta,
            LmDecoderModel,
            LmKvForwardHidden,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::language::qwen3::Qwen3Model,
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
};

/// The text projection: `linear_fc2(silu(linear_fc1(x)))`, both with biases,
/// from the text table's width to the talker's.
#[derive(Module, Debug)]
pub struct Qwen3TtsResizeMlp {
    /// `[T_w, T_w]`.
    pub linear_fc1: Linear,

    /// `[T_w, W]`.
    pub linear_fc2: Linear,
}

impl Qwen3TtsResizeMlp {
    /// `[..., T_w]` to `[..., W]`.
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D> {
        self.linear_fc2.forward(silu(self.linear_fc1.forward(x)))
    }
}

/// Qwen3-TTS's talker (`Qwen3TTSTalkerForConditionalGeneration` without its
/// code predictor): a Qwen3 decoder whose inputs are sums of embeddings and
/// whose head reads the first codebook of the next frame.
///
/// Its embedding is the codec table (`codec_embedding`: the codes, then the
/// control rows: pad, BOS, EOS, the think tags, the languages and the
/// speakers), its head a projection of its own (`codec_head`). Beside it
/// sit the text table and its projection to the talker's width
/// ([`text_embed`](Self::text_embed)), which a prompt's text rows and each
/// step's trailing text read. A prompt row is a text embedding plus a codec
/// embedding; a generated frame's row, its codebooks' embeddings summed plus
/// the next trailing text.
///
/// Built by [`Qwen3TtsTalkerConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsTalker {
    /// The decoder over the codec table.
    pub lm: LmDecoderModel,

    /// The text table, `[V_t, T_w]`.
    pub text_embedding: Embedding,

    /// The text table's projection to the talker's width.
    pub text_projection: Qwen3TtsResizeMlp,
}

impl ModuleInit<Qwen3TtsTalker> for Qwen3TtsTalkerConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsTalker> {
        let Qwen3Model { lm } = self.lm.try_init(device)?;
        let linear = |i, o| LinearConfig::new(i, o).init(device);
        Ok(Qwen3TtsTalker {
            lm,
            text_embedding: EmbeddingConfig::new(self.text_vocab, self.text_width).init(device),
            text_projection: Qwen3TtsResizeMlp {
                linear_fc1: linear(self.text_width, self.text_width),
                linear_fc2: linear(self.text_width, self.lm.d_model),
            },
        })
    }
}

impl Qwen3TtsTalker {
    /// The talker's width.
    pub fn d_model(&self) -> usize {
        self.lm.d_model()
    }

    /// Codec rows: `[B, T]` ids of the codec table to `[B, T, W]`.
    pub fn codec_embed(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.lm.embed(ids)
    }

    /// Text rows: `[B, T]` text ids to `[B, T, W]`, through the table and
    /// the projection.
    pub fn text_embed(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.text_projection
            .forward(self.text_embedding.forward(ids))
    }

    /// The rotary tables for positions `0..positions`.
    pub fn rotary(
        &self,
        positions: usize,
        device: &Device,
    ) -> RotarySet {
        self.lm.rotary(positions, device)
    }

    /// The decoder over rows at positions `start..start + T`.
    ///
    /// # Arguments
    /// - `embeds`: `[B, T, W]`, summed embeddings.
    /// - `r_emb`: rotary tables covering `start + T` positions.
    /// - `kv_cache`: the cache, at position `start`; `None`, `start` is 0.
    ///
    /// # Returns
    /// `[B, T, W]`, normalized: the head's input, and the code predictor's.
    pub fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.lm.forward_hidden(embeds, r_emb, kv_cache)
    }

    /// The first codebook's logits over the codec table: `[B, T, W]` to
    /// `[B, T, V]`, in f32.
    pub fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.lm.logits(hidden)
    }
}

impl KVCacheModel for Qwen3TtsTalker {
    fn kv_cache_config(&self) -> KVCacheConfig {
        self.lm.kv_cache_config()
    }
}
