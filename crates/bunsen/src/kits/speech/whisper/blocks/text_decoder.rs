use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        Embedding,
        EmbeddingConfig,
        LayerNorm,
        LayerNormConfig,
        attention::MhaCache,
    },
    prelude::{
        Backend,
        s,
    },
    tensor::{
        Bool,
        Distribution,
        Int,
    },
};

use super::WHISPER_DEFAULT_D_MODEL;
use crate::{
    blocks::transformers::attention::causal_mask,
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::speech::whisper::blocks::{
        ResidualDecoderAttentionBlock,
        ResidualDecoderAttentionBlockConfig,
        ResidualDecoderAttentionBlockMeta,
    },
    ops::embedding::unembed,
};

/// Common meta for [`TextDecoder`] and [`TextDecoderConfig`].
pub trait TextDecoderMeta {
    /// Returns the size of the vocabulary.
    fn vocab_size(&self) -> usize;

    /// The embedding size of the model.
    fn d_model(&self) -> usize;

    /// Returns the max context size.
    fn max_context(&self) -> usize;

    /// Returns the number of heads.
    fn n_heads(&self) -> usize;

    /// Returns the number of layers.
    fn n_layers(&self) -> usize;
}

/// Config for [`TextDecoder`].
#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,

    /// The embedding size of the model.
    pub d_model: usize,

    /// Maximum text context size.
    pub max_context: usize,

    /// The number of layers.
    pub n_layers: usize,

    /// Head Dimensionality.
    #[config(defaul_value = "WHISPER_DEFAULT_D_MODEL")]
    pub d_head: usize,

    /// Dropout.
    #[config(default = "0.0")]
    pub block_dropout: f64,
}

impl TextDecoderMeta for TextDecoderConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn n_heads(&self) -> usize {
        self.d_model / self.d_head
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }
}

/// Builds attention mask for decoder.
pub fn attn_decoder_mask<B: Backend>(
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length], device);

    for i in 0..(seq_length - 1) {
        let values =
            Tensor::<B, 2>::zeros([1, seq_length - (i + 1)], device).add_scalar(f64::NEG_INFINITY);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }

    mask
}

impl<B: Backend> ModuleInit<B, TextDecoder<B>> for TextDecoderConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<TextDecoder<B>> {
        Ok(TextDecoder {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),

            positional_embedding: Param::from_tensor(Tensor::<B, 2>::random(
                [self.max_context, self.d_model],
                Distribution::Normal(0.0, 1.0),
                device,
            )),

            blocks: (0..self.n_layers)
                .map(|_| {
                    ResidualDecoderAttentionBlockConfig::new(self.d_model)
                        .with_d_head(self.d_head)
                        .with_dropout(self.block_dropout)
                        .try_init(device)
                })
                .collect::<BunsenResult<Vec<ResidualDecoderAttentionBlock<B>>>>()?,

            ln: LayerNormConfig::new(self.d_model).init(device),
            // mask: Param::from_tensor(attn_decoder_mask(self.max_text_context, device)),
        })
    }
}

/// Text decoder module for Whisper speech recognition model.
///
/// Autoregressive token decoder: token plus positional embeddings, a stack of
/// [`ResidualDecoderAttentionBlock`] layers (causally-masked self-attention and
/// cross-attention over the audio encoder output), a final layer norm, and an
/// unembedding into vocabulary logits.
///
/// Built by [`TextDecoderConfig`].
#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    /// The token embedding.
    pub token_embedding: Embedding<B>,

    /// The positional embedding.
    pub positional_embedding: Param<Tensor<B, 2>>,

    /// The decoder blocks.
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,

    /// The output layer norm.
    pub ln: LayerNorm<B>,
    // mask: Param<Tensor<B, 2>>,
}

impl<B: Backend> TextDecoderMeta for TextDecoder<B> {
    fn vocab_size(&self) -> usize {
        self.token_embedding.weight.val().dims()[0]
    }

    fn d_model(&self) -> usize {
        self.blocks[0].d_model()
    }

    fn max_context(&self) -> usize {
        self.positional_embedding.val().dims()[0]
    }

    fn n_heads(&self) -> usize {
        self.blocks[0].n_heads()
    }

    fn n_layers(&self) -> usize {
        self.blocks.len()
    }
}

impl<B: Backend> TextDecoder<B> {
    /// Runs the decoder.
    ///
    /// # Arguments
    /// * `x`: `[batch, seq]`.
    /// * `xa`: `[batch, seq, d_model]`.
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]`.
    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        xa: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [_batch, seq_len] = x.dims();
        assert!(
            seq_len <= self.max_context(),
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.max_context()
        );

        let x = self.embed(x);

        //let mask = attn_decoder_mask(seq_len);

        let mask: Option<Tensor<B, 3, Bool>> = causal_mask(seq_len, 0, &x.device()).into();

        let mut x = x;
        for b in self.blocks.iter() {
            x = b.forward(x, xa.clone(), mask.clone()).output;
        }

        let x = self.ln.forward(x);

        unembed(&self.token_embedding, x)

        // denorm [batch, seq_len, n_vocab]
        // Needs softmax / beamsearch.
    }

    /// Opens an incremental decode cache over this decoder.
    pub fn new_cache(&self) -> TextDecoderCache<B> {
        TextDecoderCache {
            layers: self
                .blocks
                .iter()
                .map(|_| {
                    (
                        MhaCache::autoregressive(),
                        MhaCache::autoregressive_cross_attention(),
                    )
                })
                .collect(),
            pos: 0,
        }
    }

    /// Forward pass against an incremental decode cache.
    ///
    /// # This is burn's cache model, not a KV cache
    ///
    /// Pass the **whole token sequence so far**, not just the new token. The
    /// cache reuses the projections it already computed for the prefix, so the
    /// per-step cost is the projection of the new tail plus the attention.
    /// [`MhaCache`] is built that way, and burn's own
    /// `TransformerDecoder::forward_autoregressive_inference` uses it the same
    /// way.
    ///
    /// What that buys for Whisper is mostly the cross-attention: `xa` is fixed
    /// for a whole decode, so its keys and values over all 1500 encoder frames
    /// are projected **once** rather than once per token, in every layer.
    /// Self-attention still recomputes its scores over the full prefix — a
    /// true KV cache would not, and that is a separate change.
    ///
    /// # Arguments
    /// * `x`: `[batch, seq]` — every token so far, including the new one.
    /// * `xa`: `[batch, cross_len, d_model]` encoder output. Must be the same
    ///   tensor for the whole decode; the cache cannot detect otherwise.
    /// * `cache`: from [`new_cache`](Self::new_cache).
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]` logits, as [`forward`](Self::forward) would
    /// give for the same sequence. Autoregressive callers want the last
    /// position.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 2, Int>,
        xa: Tensor<B, 3>,
        cache: &mut TextDecoderCache<B>,
    ) -> Tensor<B, 3> {
        assert_eq!(
            cache.n_layers(),
            self.blocks.len(),
            "cache was built for a different decoder",
        );

        let seq_len = x.dims()[1];
        assert!(
            seq_len <= self.max_context(),
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.max_context(),
        );
        assert!(
            seq_len >= cache.pos,
            "sequence shrank from {} to {seq_len}; a cache only ever grows.              Call `reset` to start a new stream.",
            cache.pos,
        );

        let mut h = self.embed(x);
        let mask: Option<Tensor<B, 3, Bool>> = causal_mask(seq_len, 0, &h.device()).into();

        for (block, (self_cache, cross_cache)) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            h = block
                .forward_cached(h, xa.clone(), mask.clone(), self_cache, cross_cache)
                .output;
        }

        cache.pos = seq_len;

        unembed(&self.token_embedding, self.ln.forward(h))
    }

    fn embed(
        &self,
        x: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let seq_len = x.dims()[1];
        self.token_embedding.forward(x)
            + self
                .positional_embedding
                .val()
                .slice(s![0..seq_len])
                .unsqueeze::<3>()
    }
}

/// Per-layer attention caches for incremental decoding.
///
/// Built by [`TextDecoder::new_cache`]. Holds a self-attention and a
/// cross-attention [`MhaCache`] per block.
///
/// Deliberately **not** a `Module`: this is per-stream decode state, it holds
/// no parameters, and `MhaCache` is not a `Module` either.
pub struct TextDecoderCache<B: Backend> {
    /// `(self-attention, cross-attention)` per block.
    layers: Vec<(MhaCache<B>, MhaCache<B>)>,

    /// Length of the longest sequence seen, which a cache may only grow.
    pos: usize,
}

impl<B: Backend> TextDecoderCache<B> {
    /// The length of the longest sequence this cache has seen.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// The number of blocks this cache covers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Drops the cached history and rewinds to the start of a stream.
    pub fn reset(&mut self) {
        for (self_cache, cross_cache) in &mut self.layers {
            *self_cache = MhaCache::autoregressive();
            *cross_cache = MhaCache::autoregressive_cross_attention();
        }
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::PerformanceBackend,
    };

    #[test]
    #[serial]
    fn test_text_decoder_forward() {
        type B = PerformanceBackend;
        let device = Default::default();

        let d_model = 128;
        let vocab_size = 64;
        let max_context = 128;
        let n_layers = 2;

        let config = TextDecoderConfig::new(vocab_size, d_model, max_context, n_layers);

        assert_eq!(config.vocab_size(), vocab_size);
        assert_eq!(config.d_model(), d_model);
        assert_eq!(config.max_context(), max_context);

        let decoder: TextDecoder<B> = config.init(&device);

        assert_eq!(decoder.vocab_size(), vocab_size);
        assert_eq!(decoder.d_model(), d_model);
        assert_eq!(decoder.max_context(), max_context);

        let batch = 2;
        let seq_len = max_context / 2;

        let x: Tensor<B, 2, Int> = Tensor::zeros([batch, seq_len], &device);
        let xa: Tensor<B, 3> =
            Tensor::random([batch, seq_len, d_model], Default::default(), &device);

        let output = decoder.forward(x.clone(), xa.clone());

        assert_shape_contract!(
            ["batch", "seq", "n_vocab"],
            &output,
            &[("batch", batch), ("seq", seq_len), ("n_vocab", vocab_size),],
        );
    }
    /// **The cache contract.** Decoding with a growing prefix must give the
    /// same logits as decoding each prefix from scratch.
    ///
    /// This is the property the cache exists to preserve, and the one a stale
    /// projection or a mis-shaped causal mask breaks — neither of which a
    /// shape check would notice.
    #[test]
    #[serial]
    fn test_forward_cached_matches_forward() {
        use burn::{
            prelude::TensorData,
            tensor::{
                Distribution,
                Tolerance,
                backend::BackendTypes,
            },
        };

        use crate::burner::tensor::TensorElemOpExt;

        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();

        // `n_heads` is `d_model / d_head` with `d_head` defaulting to 64, so
        // a smaller `d_model` would give the attention zero heads.
        let (vocab, d_model, max_ctx, layers) = (64, 128, 16, 2);
        let (batch, cross_len, seq) = (2, 5, 6);

        let decoder: TextDecoder<B> =
            TextDecoderConfig::new(vocab, d_model, max_ctx, layers).init(&device);

        let tokens: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(
                (0..batch * seq)
                    .map(|k| (k % vocab) as i64)
                    .collect::<Vec<_>>(),
                [batch, seq],
            ),
            &device,
        );
        let xa: Tensor<B, 3> =
            Tensor::random([batch, cross_len, d_model], Distribution::Default, &device);

        // Grow the prefix a token at a time, as an autoregressive loop would.
        let mut cache = decoder.new_cache();
        for t in 1..=seq {
            let prefix = tokens.clone().slice_dim(1, 0..t as isize);

            let cached = decoder.forward_cached(prefix.clone(), xa.clone(), &mut cache);
            let fresh = decoder.forward(prefix, xa.clone());

            assert_eq!(cached.dims(), fresh.dims());
            assert_eq!(cache.pos(), t);

            cached
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&fresh.to_data_as::<F>(), Tolerance::permissive());
        }
    }

    /// A cache is reusable across streams once reset.
    #[test]
    #[serial]
    fn test_cache_reset_restarts_the_stream() {
        use burn::{
            prelude::TensorData,
            tensor::{
                Distribution,
                Tolerance,
                backend::BackendTypes,
            },
        };

        use crate::burner::tensor::TensorElemOpExt;

        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();
        let (vocab, d_model, max_ctx, layers) = (32, 128, 8, 1);

        let decoder: TextDecoder<B> =
            TextDecoderConfig::new(vocab, d_model, max_ctx, layers).init(&device);

        let tokens: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(vec![3i64, 5], [1, 2]), &device);
        let xa: Tensor<B, 3> = Tensor::random([1, 4, d_model], Distribution::Default, &device);

        let mut cache = decoder.new_cache();
        let first = decoder.forward_cached(tokens.clone(), xa.clone(), &mut cache);
        assert_eq!(cache.pos(), 2);

        cache.reset();
        assert_eq!(cache.pos(), 0);

        let second = decoder.forward_cached(tokens, xa, &mut cache);
        first
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&second.to_data_as::<F>(), Tolerance::permissive());
    }
}
