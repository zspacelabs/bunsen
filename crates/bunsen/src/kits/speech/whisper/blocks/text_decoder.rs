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
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::speech::whisper::blocks::{
        ResidualDecoderAttentionBlock,
        ResidualDecoderAttentionBlockConfig,
        ResidualDecoderAttentionBlockMeta,
    },
    ops::{
        embedding::unembed,
        transformers::attention::{
            AttnKvPair,
            causal_mask,
        },
    },
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

    /// Opens an incremental decode cache against a fixed encoder output.
    ///
    /// Projects the cross-attention keys and values for every block up front;
    /// they are reused unchanged for the whole decode.
    ///
    /// # Arguments
    /// * `xa`: `[batch, cross_len, d_model]` encoder output.
    pub fn new_cache(
        &self,
        xa: Tensor<B, 3>,
    ) -> TextDecoderCache<B> {
        TextDecoderCache {
            self_kv: self.blocks.iter().map(|_| None).collect(),
            cross_kv: self
                .blocks
                .iter()
                .map(|block| block.build_cross_kv(xa.clone()))
                .collect(),
            pos: 0,
        }
    }

    /// Forward pass over the next token(s), against a key/value cache.
    ///
    /// Pass only the **new** tokens: the cache holds everything before them.
    /// Decoding a sequence a token at a time through this gives the same
    /// logits as one [`forward`](Self::forward) over the whole sequence, at a
    /// fraction of the work — the encoder-derived cross-attention keys and
    /// values are projected once by [`new_cache`](Self::new_cache), and
    /// self-attention scores only the new queries.
    ///
    /// The positional embedding is taken from
    /// [`cache.pos()`](TextDecoderCache::pos) rather than from zero, and the
    /// causal mask spans `[seq_new, pos + seq_new]`. A single new token needs
    /// no mask at all, since it may attend to all of its history.
    ///
    /// # Arguments
    /// * `x`: `[batch, seq_new]` the next token(s).
    /// * `cache`: from [`new_cache`](Self::new_cache). Carries the encoder
    ///   output, so it is not passed again.
    ///
    /// # Returns
    /// `[batch, seq_new, n_vocab]` — logits for the new token(s) only.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 2, Int>,
        cache: &mut TextDecoderCache<B>,
    ) -> Tensor<B, 3> {
        assert_eq!(
            cache.n_layers(),
            self.blocks.len(),
            "cache was built for a different decoder",
        );

        let seq_new = x.dims()[1];
        let past = cache.pos;
        assert!(
            past + seq_new <= self.max_context(),
            "Token sequence length {} must not exceed {}.",
            past + seq_new,
            self.max_context(),
        );

        let mut h = self.embed_at(x, past);

        // One query attends to everything cached, so a mask would be all-false
        // and is skipped.
        let mask: Option<Tensor<B, 3, Bool>> = if seq_new > 1 {
            Some(causal_mask(seq_new, past, &h.device()))
        } else {
            None
        };

        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward_kv(h, mask.clone(), &mut cache.self_kv[i], &cache.cross_kv[i]);
        }

        cache.pos += seq_new;

        unembed(&self.token_embedding, self.ln.forward(h))
    }

    fn embed(
        &self,
        x: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        self.embed_at(x, 0)
    }

    /// Embeds `x`, taking positions from `offset` rather than from zero.
    fn embed_at(
        &self,
        x: Tensor<B, 2, Int>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let seq_len = x.dims()[1];
        self.token_embedding.forward(x)
            + self
                .positional_embedding
                .val()
                .slice(s![offset..offset + seq_len])
                .unsqueeze::<3>()
    }
}

/// Per-layer key/value caches for incremental decoding.
///
/// Built by [`TextDecoder::new_cache`] against a fixed encoder output. Holds,
/// per block, the growing self-attention cache and the fixed cross-attention
/// one, plus how many tokens have been consumed — which is what offsets the
/// positional embedding.
///
/// Deliberately **not** a `Module`: this is per-stream decode state and holds
/// no parameters.
pub struct TextDecoderCache<B: Backend> {
    /// Self-attention keys and values, grown one step at a time.
    self_kv: Vec<Option<AttnKvPair<B>>>,

    /// Cross-attention keys and values, projected once from the encoder
    /// output. This is the bulk of what the cache saves: over 1500 encoder
    /// frames, per layer, per token.
    cross_kv: Vec<AttnKvPair<B>>,

    /// Tokens consumed so far.
    pos: usize,
}

impl<B: Backend> TextDecoderCache<B> {
    /// The number of tokens consumed so far.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// The number of blocks this cache covers.
    pub fn n_layers(&self) -> usize {
        self.cross_kv.len()
    }

    /// Drops the decoded history, keeping the cross-attention projections.
    ///
    /// Use this to decode a second sequence against the same audio without
    /// re-projecting the encoder output.
    pub fn reset(&mut self) {
        for kv in &mut self.self_kv {
            *kv = None;
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
    /// **The cache contract.** Decoding a token at a time must give the same
    /// logits as one uncached pass over the whole sequence.
    ///
    /// This is what the cache exists to preserve, and what a wrong positional
    /// offset, a mis-shaped causal mask, or a stale key/value append would
    /// break — none of which a shape check would catch.
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

        let whole = decoder.forward(tokens.clone(), xa.clone());

        // One new token per step, as an autoregressive loop would.
        let mut cache = decoder.new_cache(xa);
        let mut steps = Vec::with_capacity(seq);
        for t in 0..seq {
            let step = tokens.clone().slice_dim(1, t as isize..(t + 1) as isize);
            steps.push(decoder.forward_cached(step, &mut cache));
        }
        assert_eq!(cache.pos(), seq);

        let stepped: Tensor<B, 3> = Tensor::cat(steps, 1);
        assert_eq!(stepped.dims(), whole.dims());

        stepped
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&whole.to_data_as::<F>(), Tolerance::permissive());
    }

    /// Prefilling a prompt then stepping must match stepping throughout —
    /// the shape a real decode has.
    #[test]
    #[serial]
    fn test_cached_prefill_then_step() {
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
        let (vocab, d_model, max_ctx, layers) = (64, 128, 16, 1);
        let (cross_len, seq, prompt) = (4, 5, 3);

        let decoder: TextDecoder<B> =
            TextDecoderConfig::new(vocab, d_model, max_ctx, layers).init(&device);

        let tokens: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new((0..seq).map(|k| k as i64).collect::<Vec<_>>(), [1, seq]),
            &device,
        );
        let xa: Tensor<B, 3> =
            Tensor::random([1, cross_len, d_model], Distribution::Default, &device);

        let whole = decoder.forward(tokens.clone(), xa.clone());

        let mut cache = decoder.new_cache(xa);
        let mut parts = vec![
            decoder.forward_cached(tokens.clone().slice_dim(1, 0..prompt as isize), &mut cache),
        ];
        for t in prompt..seq {
            let step = tokens.clone().slice_dim(1, t as isize..(t + 1) as isize);
            parts.push(decoder.forward_cached(step, &mut cache));
        }

        let mixed: Tensor<B, 3> = Tensor::cat(parts, 1);
        mixed
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&whole.to_data_as::<F>(), Tolerance::permissive());
    }

    /// `reset` rewinds the decode while keeping the encoder projections, so a
    /// second sequence over the same audio does not re-project them.
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

        let mut cache = decoder.new_cache(xa);
        let first = decoder.forward_cached(tokens.clone(), &mut cache);
        assert_eq!(cache.pos(), 2);

        cache.reset();
        assert_eq!(cache.pos(), 0);

        let second = decoder.forward_cached(tokens, &mut cache);
        first
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&second.to_data_as::<F>(), Tolerance::permissive());
    }
}
