//! The talker's modules.

use burn::{
    config::Config,
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

/// Common meta for [`Qwen3TtsResizeMlp`] and [`Qwen3TtsResizeMlpConfig`].
pub trait Qwen3TtsResizeMlpMeta {
    /// The input width.
    fn d_input(&self) -> usize;

    /// The hidden width.
    fn d_hidden(&self) -> usize;

    /// The output width.
    fn d_output(&self) -> usize;
}

/// Config for [`Qwen3TtsResizeMlp`].
#[derive(Config, Debug)]
pub struct Qwen3TtsResizeMlpConfig {
    /// The input width.
    pub d_input: usize,

    /// The hidden width.
    pub d_hidden: usize,

    /// The output width.
    pub d_output: usize,
}

impl Qwen3TtsResizeMlpMeta for Qwen3TtsResizeMlpConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    fn d_output(&self) -> usize {
        self.d_output
    }
}

impl ModuleInit<Qwen3TtsResizeMlp> for Qwen3TtsResizeMlpConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsResizeMlp> {
        Ok(Qwen3TtsResizeMlp {
            linear_fc1: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            linear_fc2: LinearConfig::new(self.d_hidden, self.d_output).init(device),
        })
    }
}

/// The text projection: `linear_fc2(silu(linear_fc1(x)))`, both with biases,
/// from the text table's width to the talker's.
///
/// Built by [`Qwen3TtsResizeMlpConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsResizeMlp {
    /// `[D_in, D_h]`.
    pub linear_fc1: Linear,

    /// `[D_h, D_out]`.
    pub linear_fc2: Linear,
}

impl Qwen3TtsResizeMlpMeta for Qwen3TtsResizeMlp {
    fn d_input(&self) -> usize {
        self.linear_fc1.weight.dims()[0]
    }

    fn d_hidden(&self) -> usize {
        self.linear_fc1.weight.dims()[1]
    }

    fn d_output(&self) -> usize {
        self.linear_fc2.weight.dims()[1]
    }
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

/// A [`Qwen3TtsTalker`]'s geometry, which its config and the talker answer
/// alike.
pub trait Qwen3TtsTalkerMeta {
    /// The talker's width, `W`.
    fn d_model(&self) -> usize;

    /// The codec table's rows, `V`: the codes and the control rows.
    fn codec_vocab(&self) -> usize;

    /// The text table's rows, `V_t`.
    fn text_vocab(&self) -> usize;

    /// The text table's width, `T_w`.
    fn text_width(&self) -> usize;
}

impl Qwen3TtsTalkerMeta for Qwen3TtsTalkerConfig {
    fn d_model(&self) -> usize {
        self.lm.d_model
    }

    fn codec_vocab(&self) -> usize {
        self.lm.n_vocab
    }

    fn text_vocab(&self) -> usize {
        self.text_vocab
    }

    fn text_width(&self) -> usize {
        self.text_width
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
        Ok(Qwen3TtsTalker {
            lm,
            text_embedding: EmbeddingConfig::new(self.text_vocab, self.text_width).init(device),
            text_projection: Qwen3TtsResizeMlpConfig::new(
                self.text_width,
                self.text_width,
                self.lm.d_model,
            )
            .try_init(device)?,
        })
    }
}

impl Qwen3TtsTalkerMeta for Qwen3TtsTalker {
    fn d_model(&self) -> usize {
        self.lm.d_model()
    }

    fn codec_vocab(&self) -> usize {
        self.lm.n_vocab()
    }

    fn text_vocab(&self) -> usize {
        self.text_embedding.weight.dims()[0]
    }

    fn text_width(&self) -> usize {
        self.text_embedding.weight.dims()[1]
    }
}

impl Qwen3TtsTalker {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kits::language::qwen3::Qwen3Config,
        support::testing::cpu_device,
    };

    /// The config and the talker built from it answer the meta alike.
    #[test]
    fn test_talker_meta() {
        let config = Qwen3TtsTalkerConfig::new(Qwen3Config::new(11, 8, 1, 2, 1, 4, 12), 13, 6, 4);
        let talker: Qwen3TtsTalker = config.init(&cpu_device());
        for meta in [&config as &dyn Qwen3TtsTalkerMeta, &talker] {
            assert_eq!(
                [
                    meta.d_model(),
                    meta.codec_vocab(),
                    meta.text_vocab(),
                    meta.text_width()
                ],
                [8, 11, 13, 6]
            );
        }
    }

    /// The config and the projection built from it answer the meta alike.
    #[test]
    fn test_resize_mlp_meta() {
        let config = Qwen3TtsResizeMlpConfig::new(3, 5, 7);
        let mlp: Qwen3TtsResizeMlp = config.init(&cpu_device());
        for meta in [&config as &dyn Qwen3TtsResizeMlpMeta, &mlp] {
            assert_eq!(
                [meta.d_input(), meta.d_hidden(), meta.d_output()],
                [3, 5, 7]
            );
        }
    }
}
