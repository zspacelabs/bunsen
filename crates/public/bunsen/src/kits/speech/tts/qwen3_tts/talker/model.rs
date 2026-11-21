//! The talker's modules.

use burn::{
    config::Config,
    nn::{
        Embedding,
        EmbeddingConfig,
    },
    prelude::*,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        mlp::{
            BunsenMlp,
            BunsenMlpConfig,
            BunsenMlpMeta,
        },
        transformers::{
            embedding::RotarySet,
            models::language_decoder::{
                LmDecoderMeta,
                LmDecoderModel,
                LmKvForwardHidden,
            },
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::language::generation::qwen3::{
        Qwen3Config,
        Qwen3Model,
    },
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
};

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

    /// The text projection's hidden width (the release's `T_w`).
    fn text_hidden(&self) -> usize;
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

    fn text_hidden(&self) -> usize {
        self.text_projection.d_hidden
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

    /// The text table's projection to the talker's width,
    /// `linear2(act(linear1(x)))` (the release's
    /// [`release_text_projection`]).
    pub text_projection: BunsenMlp,
}

impl ModuleInit<Qwen3TtsTalker> for Qwen3TtsTalkerConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a text
    /// projection that does not read `T_w` and write `W`; as the parts'
    /// builds.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsTalker> {
        let p = &self.text_projection;
        if [p.d_input, p.d_output] != [self.text_width, self.lm.d_model] {
            return Err(BunsenError::illegal(format!(
                "a text projection {} -> {}; the talker's is {} -> {}",
                p.d_input, p.d_output, self.text_width, self.lm.d_model
            )));
        }
        let Qwen3Model { lm } = self.lm.try_init(device)?;
        Ok(Qwen3TtsTalker {
            lm,
            text_embedding: EmbeddingConfig::new(self.text_vocab, self.text_width).init(device),
            text_projection: p.try_init(device)?,
        })
    }
}

/// The release's text projection (upstream's `Qwen3TTSTalkerResizeMLP`):
/// `text_width -> text_width -> d_model`, `SiLU`, both projections with
/// biases.
pub fn release_text_projection(
    text_width: usize,
    d_model: usize,
) -> BunsenMlpConfig {
    BunsenMlpConfig::new(text_width, text_width, d_model)
        .with_activation(BunsenActivationConfig::Silu)
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

    fn text_hidden(&self) -> usize {
        self.text_projection.d_hidden()
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

/// Config for [`Qwen3TtsTalker`](super::Qwen3TtsTalker): the language model
/// over the first codebook, and the text table its prompt reads.
#[derive(Config, Debug)]
pub struct Qwen3TtsTalkerConfig {
    /// The decoder: a Qwen3 stack over the codec table (`n_vocab`, the
    /// codes and the control rows), its head a projection of its own.
    pub lm: Qwen3Config,

    /// The text table's rows.
    pub text_vocab: usize,

    /// The text table's width, the text projection's input.
    pub text_width: usize,

    /// The text projection, `T_w` to the decoder's width.
    pub text_projection: BunsenMlpConfig,

    /// The codebooks a frame holds: the talker's first and the code
    /// predictor's rest.
    pub code_groups: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kits::language::generation::qwen3::Qwen3Config,
        support::testing::cpu_device,
    };

    /// The config and the talker built from it answer the meta alike, a
    /// text projection's hidden width other than the release's included.
    #[test]
    fn test_talker_meta() {
        let config = Qwen3TtsTalkerConfig::new(
            Qwen3Config::new(11, 8, 1, 2, 1, 4, 12),
            13,
            6,
            BunsenMlpConfig::new(6, 5, 8).with_activation(BunsenActivationConfig::Gelu),
            4,
        );
        let talker: Qwen3TtsTalker = config.init(&cpu_device());
        for meta in [&config as &dyn Qwen3TtsTalkerMeta, &talker] {
            assert_eq!(
                [
                    meta.d_model(),
                    meta.codec_vocab(),
                    meta.text_vocab(),
                    meta.text_width(),
                    meta.text_hidden()
                ],
                [8, 11, 13, 6, 5]
            );
        }
    }
}
