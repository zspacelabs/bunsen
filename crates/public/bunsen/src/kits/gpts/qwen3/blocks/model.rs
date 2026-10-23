//! The Qwen3 decoder.
use std::ops::Range;

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
    prelude::{
        Int,
        s,
    },
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        attention::csa::{
            CausalSelfAttentionConfig,
            QkNormPlacement,
        },
        decoders::causal::{
            CausalDecoderBlock,
            CausalDecoderBlockConfig,
        },
        embedding::{
            RotaryDirection,
            RotaryEmbedding,
            RotaryEmbeddingConfig,
        },
        mlp::GatedMlpConfig,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::{
        embedding::unembed,
        transformers::attention::{
            KVCache,
            KVCacheConfig,
            KVCacheModel,
        },
    },
};

/// The positions a Qwen3 cache allocates up front by default: one growth
/// chunk. Qwen3 builds its rotary tables to order, so it has no positional cap
/// of its own; a caller that knows a decode's length sizes (and caps) the
/// cache to it.
pub const QWEN3_KV_CACHE_PRE_SIZE: usize = 1024;

/// The [`KVCacheConfig`] a Qwen3 model of `meta`'s geometry decodes with: its
/// KV heads, head width and layer count, [`QWEN3_KV_CACHE_PRE_SIZE`]
/// positions up front, uncapped.
fn kv_cache_config(meta: &impl Qwen3Meta) -> KVCacheConfig {
    KVCacheConfig::new(
        1,
        meta.n_kv_heads(),
        QWEN3_KV_CACHE_PRE_SIZE,
        meta.head_dim(),
        meta.n_layers(),
    )
}

impl KVCacheModel for Qwen3Config {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

impl KVCacheModel for Qwen3Model {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

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

    /// `Qwen3-1.7B`'s geometry: the text model of `Qwen3-ASR-1.7B`.
    pub fn qwen3_1_7b() -> Self {
        Self::new(151_936, 2048, 28, 16, 8, 128, 6144)
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
                CausalDecoderBlockConfig::new(
                    CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, self.d_model)
                        .with_head_dim(Some(self.head_dim))
                        .with_norm(NormalizationConfig::Rms(norm(self.head_dim)))
                        .with_qk_norm(QkNormPlacement::BeforeRope),
                    GatedMlpConfig::new(self.d_model, self.d_ff).into(),
                )
                .with_norm(NormalizationConfig::Rms(norm(0)))
                .try_init(i, device)
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

/// The Qwen3 decoder-only language model (Qwen team 2025), with its head
/// tied to its embedding: the text model of Qwen3-ASR.
///
/// Its own fields carry the names `transformers`' `Qwen3Model` gives them;
/// its layers are the generic
/// [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)
/// (RMS norms, a `SwiGLU`, QK-norm before the rotary embedding), onto which
/// Qwen3-ASR's loader maps the `transformers` names.
///
/// Built by [`Qwen3Config`].
/// Implements [`Qwen3Meta`].
#[derive(Module, Debug)]
pub struct Qwen3Model {
    /// The token embedding, and the tied output head.
    pub embed_tokens: Embedding,

    /// The decoder layers.
    pub layers: Vec<CausalDecoderBlock>,

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
        self.layers[0].attn.n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.layers[0].attn.head_dim
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
        #[cfg(any(debug_assertions, test))]
        let [batch, seq] =
            crate::contracts::unpack_shape_contract!(["batch", "seq"], &tokens.dims());

        let y = self.embed_tokens.forward(tokens);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("d_model", self.d_model())],
        );

        y
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
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
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
        let y = self.norm.forward(x);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("d_model", self.d_model())],
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;

        y
    }

    /// The attention weights of an uncached pass over `embeds`, at `layers`,
    /// each cut to the queries `rows` and the keys `cols`.
    ///
    /// # Arguments
    /// * `embeds` - `[batch, seq, d_model]`, at positions `0..seq`.
    /// * `r_emb` - a rotary embedding covering `seq` positions.
    /// * `layers` - the layers, by index, in the order returned.
    /// * `rows`, `cols` - the query and key positions kept.
    ///
    /// # Returns
    /// For each of `layers`, `[batch, n_heads, rows.len(), cols.len()]`.
    ///
    /// # Panics
    /// If a layer index is past the last layer.
    pub fn attention_weights(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotaryEmbedding,
        layers: &[usize],
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<Tensor<4>> {
        let [_, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &embeds.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())]
        );
        assert!(
            layers.iter().all(|&l| l < self.layers.len()),
            "layers {layers:?} of {}",
            self.layers.len()
        );
        let r_emb = r_emb.clip_range(0..seq);
        let deepest = layers.iter().copied().max().map_or(0, |l| l + 1);
        let mut weights: Vec<Option<Tensor<4>>> = vec![None; layers.len()];
        let mut x = embeds;
        for (index, layer) in self.layers.iter().enumerate().take(deepest) {
            for (slot, _) in layers.iter().enumerate().filter(|&(_, &l)| l == index) {
                let w = layer
                    .attn
                    .attention_weights(layer.attn_norm.forward(x.clone()), &r_emb);
                weights[slot] = Some(w.slice(s![.., .., rows.clone(), cols.clone()]));
            }
            x = layer.forward(x, &r_emb, &mut None);
        }
        weights
            .into_iter()
            .map(|w| w.expect("every layer visited"))
            .collect()
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
        #[cfg(any(debug_assertions, test))]
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &hidden.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())],
        );

        let y = unembed(&self.embed_tokens, hidden);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "n_vocab"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("n_vocab", self.n_vocab())],
        );

        y
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
        assert_eq!(model.layers[0].attn.c_q.weight.dims(), [12, 18]);

        let tokens = Tensor::<2, Int>::from_data([[1, 5, 7, 2, 9]], &device);
        let embeds = model.embed(tokens);
        let r_emb = model.rotary(8, &device);
        let whole = model.forward_hidden(embeds.clone(), &r_emb, None);
        let mut cache = model.kv_cache_config().with_seq_len(8).init();
        let head =
            model.forward_hidden(embeds.clone().slice_dim(1, 0..3), &r_emb, Some(&mut cache));
        let tail = model.forward_hidden(embeds.slice_dim(1, 3..5), &r_emb, Some(&mut cache));
        let stepped = Tensor::cat(vec![head, tail], 1);
        stepped.to_data_as::<f32>().assert_approx_eq::<f32>(
            &whole.to_data_as::<f32>(),
            burn::tensor::Tolerance::rel_abs(1e-4, 1e-5),
        );
    }

    /// The weights at a layer are its attention's over that layer's input,
    /// cut to the rows and columns asked for, in the order asked for.
    #[test]
    fn test_attention_weights() {
        let device = cpu_device();
        let model: Qwen3Model = Qwen3Config::new(40, 12, 3, 3, 1, 6, 20).init(&device);
        let embeds = model.embed(Tensor::<2, Int>::from_data([[1, 5, 7, 2, 9]], &device));
        let r_emb = model.rotary(5, &device);

        let weights = model.attention_weights(embeds.clone(), &r_emb, &[1, 0], 2..5, 0..3);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].dims(), [1, 3, 3, 3]);

        let first = &model.layers[0];
        let x0 = embeds.clone();
        let expected0 = first
            .attn
            .attention_weights(first.attn_norm.forward(x0.clone()), &r_emb)
            .slice(s![.., .., 2..5, 0..3]);
        let second = &model.layers[1];
        let x1 = first.forward(x0, &r_emb, &mut None);
        let expected1 = second
            .attn
            .attention_weights(second.attn_norm.forward(x1), &r_emb)
            .slice(s![.., .., 2..5, 0..3]);
        crate::support::testing::assert_tensors_close(
            &weights[0],
            &expected1,
            burn::tensor::Tolerance::default(),
        );
        crate::support::testing::assert_tensors_close(
            &weights[1],
            &expected0,
            burn::tensor::Tolerance::default(),
        );
    }
}
