//! The Qwen3 decoder.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
        norm::{
            NormalizationConfig,
            RmsNorm,
            RmsNormConfig,
        },
    },
    prelude::Int,
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        attention::csa::{
            CausalSelfAttention,
            CausalSelfAttentionConfig,
            QkNormPlacement,
        },
        embedding::{
            RotaryDirection,
            RotaryEmbedding,
            RotaryEmbeddingConfig,
        },
        mlp::{
            GatedMlp,
            GatedMlpConfig,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::{
        embedding::unembed,
        transformers::attention::{
            KVCache,
            KVCacheConfig,
        },
    },
};

/// Common meta for [`Qwen3Model`] and [`Qwen3Config`].
pub trait Qwen3Meta {
    /// The token count.
    fn n_vocab(&self) -> usize;

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The key/value head count.
    fn n_kv_heads(&self) -> usize;

    /// The head width.
    fn head_dim(&self) -> usize;
}

/// Config for [`Qwen3Model`]: `transformers`' `Qwen3Config`, as far as a
/// decode needs it.
///
/// Implements [`Qwen3Meta`].
#[derive(Config, Debug)]
pub struct Qwen3Config {
    /// The token count.
    pub n_vocab: usize,

    /// The model width.
    pub d_model: usize,

    /// The layer count.
    pub n_layers: usize,

    /// The query head count.
    pub n_heads: usize,

    /// The key/value head count.
    pub n_kv_heads: usize,

    /// The head width.
    pub head_dim: usize,

    /// The feed-forward hidden width.
    pub d_ff: usize,

    /// The RMS normalization epsilon.
    #[config(default = "1e-6")]
    pub rms_eps: f64,

    /// The rotary embedding's base.
    #[config(default = "1_000_000")]
    pub rope_theta: usize,
}

impl Qwen3Config {
    /// The text model of `Qwen3-ASR-0.6B` (`Qwen3-0.6B`'s geometry): 151936
    /// tokens, 28 layers of width 1024, 16 query and 8 key/value heads of
    /// width 128, a 3072-wide feed-forward, and a rotary base of 10^6.
    pub fn qwen3_0_6b() -> Self {
        Self::new(151_936, 1024, 28, 16, 8, 128, 3072)
    }

    /// The rotary embedding for `seq_len` positions.
    pub fn rotary(
        &self,
        seq_len: usize,
    ) -> RotaryEmbeddingConfig {
        RotaryEmbeddingConfig::new(seq_len, self.head_dim)
            .with_base(self.rope_theta)
            .with_direction(RotaryDirection::Forward)
    }
}

impl Qwen3Meta for Qwen3Config {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl ModuleInit<Qwen3Model> for Qwen3Config {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3Model> {
        let norm = |n| RmsNormConfig::new(n).with_epsilon(self.rms_eps);
        let layers = (0..self.n_layers)
            .map(|i| {
                Ok(Qwen3Layer {
                    input_layernorm: norm(self.d_model).init(device),
                    self_attn: CausalSelfAttentionConfig::new(
                        self.n_heads,
                        self.n_kv_heads,
                        self.d_model,
                    )
                    .with_head_dim(Some(self.head_dim))
                    .with_norm(NormalizationConfig::Rms(norm(self.head_dim)))
                    .with_qk_norm(QkNormPlacement::BeforeRope)
                    .try_init(i, device)?,
                    post_attention_layernorm: norm(self.d_model).init(device),
                    mlp: GatedMlpConfig::new(self.d_model, self.d_ff).try_init(device)?,
                })
            })
            .collect::<BunsenResult<_>>()?;
        Ok(Qwen3Model {
            embed_tokens: EmbeddingConfig::new(self.n_vocab, self.d_model).init(device),
            layers,
            norm: norm(self.d_model).init(device),
            rope_theta: self.rope_theta,
        })
    }
}

/// One Qwen3 decoder layer: pre-normalized causal self-attention, then a
/// pre-normalized `SwiGLU` feed-forward, each residual.
#[derive(Module, Debug)]
pub struct Qwen3Layer {
    /// The attention's normalization.
    pub input_layernorm: RmsNorm,

    /// Causal self-attention: grouped-query, its queries and keys
    /// RMS-normalized per head before the rotary embedding.
    pub self_attn: CausalSelfAttention,

    /// The feed-forward's normalization.
    pub post_attention_layernorm: RmsNorm,

    /// The `SwiGLU` feed-forward.
    pub mlp: GatedMlp,
}

impl Qwen3Layer {
    /// Applies the layer.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq, d_model]`.
    /// * `r_emb` - the rotary embedding of the `seq` positions.
    /// * `kv_cache` - the decode's cache, when there is one.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        r_emb: &RotaryEmbedding,
        kv_cache: &mut Option<&mut KVCache>,
    ) -> Tensor<3> {
        let x = x.clone()
            + self
                .self_attn
                .forward(self.input_layernorm.forward(x), r_emb, kv_cache);
        x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x))
    }
}

/// The Qwen3 decoder-only language model (Qwen team 2025), with its head
/// tied to its embedding: the text model of Qwen3-ASR.
///
/// The fields carry the names `transformers`' `Qwen3Model` gives them.
///
/// Built by [`Qwen3Config`].
/// Implements [`Qwen3Meta`].
#[derive(Module, Debug)]
pub struct Qwen3Model {
    /// The token embedding, and the tied output head.
    pub embed_tokens: Embedding,

    /// The decoder layers.
    pub layers: Vec<Qwen3Layer>,

    /// The final normalization.
    pub norm: RmsNorm,

    /// The rotary embedding's base.
    pub rope_theta: usize,
}

impl Qwen3Meta for Qwen3Model {
    fn n_vocab(&self) -> usize {
        self.embed_tokens.weight.dims()[0]
    }

    fn d_model(&self) -> usize {
        self.embed_tokens.weight.dims()[1]
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_heads(&self) -> usize {
        use crate::blocks::transformers::attention::csa::CausalSelfAttentionMeta;
        self.layers[0].self_attn.n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.layers[0].self_attn.head_dim
    }
}

impl Qwen3Model {
    /// The rotary embedding for positions `0..seq_len`.
    pub fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotaryEmbedding {
        RotaryEmbeddingConfig::new(seq_len, self.head_dim())
            .with_base(self.rope_theta)
            .with_direction(RotaryDirection::Forward)
            .init(device)
    }

    /// A cache for `batch` decodes of up to `seq_len` positions; it grows past
    /// that.
    pub fn new_cache(
        &self,
        batch: usize,
        seq_len: usize,
    ) -> KVCache {
        KVCacheConfig::new(
            batch,
            self.n_kv_heads(),
            seq_len,
            self.head_dim(),
            self.n_layers(),
        )
        .init()
    }

    /// Embeds tokens.
    ///
    /// # Arguments
    /// * `tokens` - `[batch, seq]`.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`.
    pub fn embed(
        &self,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.embed_tokens.forward(tokens)
    }

    /// Runs the decoder over embeddings at positions `start..start + seq`,
    /// through the cache when there is one.
    ///
    /// # Arguments
    /// * `embeds` - `[batch, seq, d_model]`.
    /// * `r_emb` - a rotary embedding covering `start + seq` positions.
    /// * `kv_cache` - the cache, at position `start`; `None` means `start` is 0
    ///   and nothing is kept.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, normalized: the head's input.
    pub fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotaryEmbedding,
        mut kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        let [_, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &embeds.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())]
        );
        let start = kv_cache.as_ref().map(|c| c.pos()).unwrap_or(0);
        let r_emb = r_emb.clip_range(start..start + seq);
        let mut x = embeds;
        for layer in &self.layers {
            x = layer.forward(x, &r_emb, &mut kv_cache);
        }
        self.norm.forward(x)
    }

    /// The tied head: logits over the vocabulary.
    ///
    /// # Arguments
    /// * `hidden` - `[batch, seq, d_model]`, from
    ///   [`forward_hidden`](Self::forward_hidden).
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]`.
    pub fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        unembed(&self.embed_tokens, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// A config non-default in every field, and the model built from it,
    /// answer the meta alike; a cached decode's hidden states are the
    /// uncached forward's.
    #[test]
    fn test_meta_and_cache() {
        let config = Qwen3Config::new(40, 12, 2, 3, 1, 6, 20)
            .with_rms_eps(1e-5)
            .with_rope_theta(500);
        let device = cpu_device();
        let model: Qwen3Model = config.init(&device);
        assert_eq!(model.n_vocab(), config.n_vocab());
        assert_eq!(model.d_model(), config.d_model());
        assert_eq!(model.n_layers(), config.n_layers());
        assert_eq!(model.n_kv_heads(), config.n_kv_heads());
        assert_eq!(model.head_dim(), config.head_dim());
        assert_eq!(model.rope_theta, config.rope_theta);
        assert_eq!(model.layers[0].self_attn.c_q.weight.dims(), [12, 18]);

        let tokens = Tensor::<2, Int>::from_data([[1, 5, 7, 2, 9]], &device);
        let embeds = model.embed(tokens);
        let r_emb = model.rotary(8, &device);
        let whole = model.forward_hidden(embeds.clone(), &r_emb, None);
        let mut cache = model.new_cache(1, 8);
        let head =
            model.forward_hidden(embeds.clone().slice_dim(1, 0..3), &r_emb, Some(&mut cache));
        let tail = model.forward_hidden(embeds.slice_dim(1, 3..5), &r_emb, Some(&mut cache));
        let stepped = Tensor::cat(vec![head, tail], 1);
        stepped.to_data_as::<f32>().assert_approx_eq::<f32>(
            &whole.to_data_as::<f32>(),
            burn::tensor::Tolerance::rel_abs(1e-4, 1e-5),
        );
    }
}
