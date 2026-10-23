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
    prelude::s,
    tensor::{
        Bool,
        DType,
        Device,
        Distribution,
        Int,
    },
};

use super::WHISPER_DEFAULT_D_MODEL;
use crate::{
    blocks::transformers::decoders::cross::{
        CrossDecoderBlock,
        CrossDecoderBlockConfig,
        CrossDecoderBlockMeta,
    },
    burner::{
        module::{
            HasDType,
            ModuleInit,
        },
        tensor::device_float_dtype,
    },
    errors::BunsenResult,
    ops::{
        embedding::unembed,
        transformers::attention::{
            KVCache,
            KVCacheConfig,
            KVCacheModel,
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

/// The [`KVCacheConfig`] a Whisper text decoder of `meta`'s geometry decodes
/// with: its heads, head width and layer count, pre-sized to the whole text
/// context and capped there, since no decode may run past it.
fn kv_cache_config(meta: &impl TextDecoderMeta) -> KVCacheConfig {
    KVCacheConfig::new(
        1,
        meta.n_heads(),
        meta.max_context(),
        meta.d_model() / meta.n_heads(),
        meta.n_layers(),
    )
    .with_max_seq_len(Some(meta.max_context()))
}

impl KVCacheModel for TextDecoderConfig {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

/// [`new_cache`](TextDecoder::new_cache) builds the decode's cache with the
/// encoder's cross-attention keys and values set; this gives its geometry and
/// sizing.
impl KVCacheModel for TextDecoder {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
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

impl ModuleInit<TextDecoder> for TextDecoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<TextDecoder> {
        Ok(TextDecoder {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),

            positional_embedding: Param::from_tensor(Tensor::<2>::random(
                [self.max_context, self.d_model],
                Distribution::Normal(0.0, 1.0),
                device,
            )),

            blocks: (0..self.n_layers)
                .map(|_| {
                    CrossDecoderBlockConfig::new(self.d_model)
                        .with_d_head(self.d_head)
                        .with_dropout(self.block_dropout)
                        .try_init(device)
                })
                .collect::<BunsenResult<Vec<CrossDecoderBlock>>>()?,

            ln: LayerNormConfig::new(self.d_model).init(device),
            // mask: Param::from_tensor(attn_decoder_mask(self.max_text_context, device)),
        })
    }
}

/// Text decoder module for Whisper speech recognition model.
///
/// Autoregressive token decoder: token plus positional embeddings, a stack of
/// [`CrossDecoderBlock`] layers (causally-masked self-attention and
/// cross-attention over the audio encoder output), a final layer norm, and an
/// unembedding into vocabulary logits.
///
/// Built by [`TextDecoderConfig`].
#[derive(Module, Debug)]
pub struct TextDecoder {
    /// The token embedding.
    pub token_embedding: Embedding,

    /// The positional embedding.
    pub positional_embedding: Param<Tensor<2>>,

    /// The decoder blocks.
    pub blocks: Vec<CrossDecoderBlock>,

    /// The output layer norm.
    pub ln: LayerNorm,
    // mask: Param<Tensor<2>>,
}

impl TextDecoderMeta for TextDecoder {
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

impl HasDType for TextDecoder {
    fn dtype(&self) -> DType {
        self.token_embedding.weight.dtype()
    }
}

impl TextDecoder {
    /// Runs the decoder.
    ///
    /// # Arguments
    /// * `x`: `[batch, seq]` token ids.
    /// * `xa`: `[batch, seq, d_model]` encoder features, in any float dtype;
    ///   cast to [`dtype`](Self::dtype) here.
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]` logits, in the backend's default float
    /// ([`device_float_dtype`]) whatever the decoder's own precision. This
    /// is the model's outward edge: everything past it — the log-softmax,
    /// the samplers, the [`LogitFilter`]s, the host reads — works in the
    /// backend's float, and upstream likewise takes its logits to `float()`
    /// before the search sees them.
    ///
    /// [`LogitFilter`]: crate::kits::speech::whisper::logit_filters::LogitFilter
    pub fn forward(
        &self,
        x: Tensor<2, Int>,
        xa: Tensor<3>,
    ) -> Tensor<3> {
        let xa = xa.cast(self.dtype());

        let [_batch, seq_len] = x.dims();
        assert!(
            seq_len <= self.max_context(),
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.max_context()
        );

        let x = self.embed(x);

        //let mask = attn_decoder_mask(seq_len);

        let mask: Option<Tensor<3, Bool>> = causal_mask(seq_len, 0, &x.device()).into();

        let mut x = x;
        for b in self.blocks.iter() {
            x = b.forward(x, xa.clone(), mask.clone()).output;
        }

        let x = self.ln.forward(x);

        // denorm [batch, seq_len, n_vocab]
        // Needs softmax / beamsearch.
        {
            let dtype = device_float_dtype(&x.device());
            unembed(&self.token_embedding, x).cast(dtype)
        }
    }

    /// Opens an incremental decode's [`KVCache`] against a fixed encoder
    /// output.
    ///
    /// Projects the cross-attention keys and values for every block up front
    /// ([`set_cross_kv`](KVCache::set_cross_kv)); they are reused unchanged
    /// for the whole decode. The cache is sized as
    /// [`kv_cache_config`](KVCacheModel::kv_cache_config) says, at a row per
    /// row of `xa`.
    ///
    /// # Arguments
    /// * `xa`: `[batch, cross_len, d_model]` encoder output, in any float
    ///   dtype; the cache is built in [`dtype`](Self::dtype).
    pub fn new_cache(
        &self,
        xa: Tensor<3>,
    ) -> KVCache {
        self.new_cache_grouped(xa, 1)
    }

    /// [`new_cache`](Self::new_cache) shared by `group` decode rows per
    /// audio: the cross-attention projections are built once per row of
    /// `xa`, and a forward over `xa.batch * group` token rows, laid out
    /// `row = audio * group + member`, attends each member to its audio's
    /// projection ([`KVCache::group`]). The self-attention history is at the
    /// full row count, and [`KVCache::reorder`] picks its rows as ever.
    ///
    /// This is what a beam search wants: its beams share their audio's
    /// cross-attention, which is the bulk of the cache, and the copy per
    /// beam that repeating `xa` would cost is never made.
    pub fn new_cache_grouped(
        &self,
        xa: Tensor<3>,
        group: usize,
    ) -> KVCache {
        assert!(group >= 1, "at least one row per audio");

        // The cross-attention keys and values are the bulk of the decode's
        // state and are scored against this decoder's weights on every step,
        // so they are built in the decoder's precision whatever the caller
        // handed in.
        let xa = xa.cast(self.dtype());

        let mut cache = self
            .kv_cache_config()
            .with_batch_size(xa.dims()[0] * group)
            .init();
        for (layer, block) in self.blocks.iter().enumerate() {
            let (k, v) = block.cross_kv(xa.clone());
            cache.set_cross_kv(layer, k, v);
        }
        cache
    }

    /// Forward pass over the next token(s), against a decode's [`KVCache`].
    ///
    /// Pass only the **new** tokens: the cache holds everything before them.
    /// Decoding a sequence a token at a time through this gives the same
    /// logits as one [`forward`](Self::forward) over the whole sequence, at a
    /// fraction of the work: the encoder-derived cross-attention keys and
    /// values are projected once by [`new_cache`](Self::new_cache), and
    /// self-attention scores only the new queries.
    ///
    /// The positional embedding is taken from [`cache.pos()`](KVCache::pos)
    /// rather than from zero, and the causal mask spans
    /// `[seq_new, pos + seq_new]`. A single new token needs no mask at all,
    /// since it may attend to all of its history.
    ///
    /// # Arguments
    /// * `x`: `[batch, seq_new]` the next token(s).
    /// * `cache`: from [`new_cache`](Self::new_cache). Carries the encoder
    ///   output, so it is not passed again.
    ///
    /// # Returns
    /// `[batch, seq_new, n_vocab]`: logits for the new token(s) only, in
    /// the backend's default float, as [`forward`](Self::forward).
    pub fn forward_cached(
        &self,
        x: Tensor<2, Int>,
        cache: &mut KVCache,
    ) -> Tensor<3> {
        use crate::ops::transformers::attention::KVCacheMeta;
        assert_eq!(
            cache.num_layers(),
            self.blocks.len(),
            "cache was built for a different decoder",
        );

        let seq_new = x.dims()[1];
        let past = cache.pos();
        assert!(
            past + seq_new <= self.max_context(),
            "Token sequence length {} must not exceed {}.",
            past + seq_new,
            self.max_context(),
        );

        let mut h = self.embed_at(x, past);

        // One query attends to everything cached, so a mask would be all-false
        // and is skipped.
        let mask: Option<Tensor<3, Bool>> = if seq_new > 1 {
            Some(causal_mask(seq_new, past, &h.device()))
        } else {
            None
        };

        // The cache's position advances once the last block has written.
        for (layer, block) in self.blocks.iter().enumerate() {
            h = block.forward_w_kv_cache(h, mask.clone(), cache, layer);
        }

        {
            let dtype = device_float_dtype(&h.device());
            unembed(&self.token_embedding, self.ln.forward(h)).cast(dtype)
        }
    }

    fn embed(
        &self,
        x: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.embed_at(x, 0)
    }

    /// Embeds `x`, taking positions from `offset` rather than from zero.
    fn embed_at(
        &self,
        x: Tensor<2, Int>,
        offset: usize,
    ) -> Tensor<3> {
        let seq_len = x.dims()[1];
        self.token_embedding.forward(x)
            + self
                .positional_embedding
                .val()
                .slice(s![offset..offset + seq_len])
                .unsqueeze::<3>()
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::{
            DeviceMemoryGuard,
            cpu_device,
            performance_device,
        },
    };

    #[test]
    #[serial]
    fn test_text_decoder_forward() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let d_model = 128;
        let vocab_size = 64;
        let max_context = 128;
        let n_layers = 2;

        let config = TextDecoderConfig::new(vocab_size, d_model, max_context, n_layers);

        assert_eq!(config.vocab_size(), vocab_size);
        assert_eq!(config.d_model(), d_model);
        assert_eq!(config.max_context(), max_context);

        let decoder: TextDecoder = config.init(&device);

        assert_eq!(decoder.vocab_size(), vocab_size);
        assert_eq!(decoder.d_model(), d_model);
        assert_eq!(decoder.max_context(), max_context);

        let batch = 2;
        let seq_len = max_context / 2;

        let x: Tensor<2, Int> = Tensor::zeros([batch, seq_len], &device);
        let xa: Tensor<3> = Tensor::random([batch, seq_len, d_model], Default::default(), &device);

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
            },
        };

        type F = f32;

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        // `n_heads` is `d_model / d_head` with `d_head` defaulting to 64, so
        // a smaller `d_model` would give the attention zero heads.
        let (vocab, d_model, max_ctx, layers) = (64, 128, 16, 2);
        let (batch, cross_len, seq) = (2, 5, 6);

        let decoder: TextDecoder =
            TextDecoderConfig::new(vocab, d_model, max_ctx, layers).init(&device);

        let tokens: Tensor<2, Int> = Tensor::from_data(
            TensorData::new(
                (0..batch * seq)
                    .map(|k| (k % vocab) as i64)
                    .collect::<Vec<_>>(),
                [batch, seq],
            ),
            &device,
        );
        let xa: Tensor<3> =
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

        let stepped: Tensor<3> = Tensor::cat(steps, 1);
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
            },
        };

        type F = f32;

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let (vocab, d_model, max_ctx, layers) = (64, 128, 16, 1);
        let (cross_len, seq, prompt) = (4, 5, 3);

        let decoder: TextDecoder =
            TextDecoderConfig::new(vocab, d_model, max_ctx, layers).init(&device);

        let tokens: Tensor<2, Int> = Tensor::from_data(
            TensorData::new((0..seq).map(|k| k as i64).collect::<Vec<_>>(), [1, seq]),
            &device,
        );
        let xa: Tensor<3> = Tensor::random([1, cross_len, d_model], Distribution::Default, &device);

        let whole = decoder.forward(tokens.clone(), xa.clone());

        let mut cache = decoder.new_cache(xa);
        let mut parts = vec![
            decoder.forward_cached(tokens.clone().slice_dim(1, 0..prompt as isize), &mut cache),
        ];
        for t in prompt..seq {
            let step = tokens.clone().slice_dim(1, t as isize..(t + 1) as isize);
            parts.push(decoder.forward_cached(step, &mut cache));
        }

        let mixed: Tensor<3> = Tensor::cat(parts, 1);
        mixed
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&whole.to_data_as::<F>(), Tolerance::permissive());
    }

    /// **Both caches are the decoder's precision**, whatever the encoder
    /// features were handed over in.
    ///
    /// The cross-attention cache is the bulk of a decode's state and the
    /// self-attention cache is scored against it every step; a cache at the
    /// caller's precision rather than the model's would be re-cast on every
    /// use at best, and silently wrong at worst.
    #[test]
    #[serial]
    fn test_caches_take_the_decoders_dtype() {
        use burn::{
            module::Module,
            tensor::Distribution,
        };

        use crate::burner::{
            module::DTypeMapper,
            tensor::device_float_dtype,
        };

        let device = cpu_device();

        let float = device_float_dtype(&device);
        let decoder: TextDecoder = TextDecoderConfig::new(32, 128, 8, 1).init(&device);
        let decoder = decoder.map(&mut DTypeMapper::new(DType::F16));
        assert_eq!(decoder.dtype(), DType::F16);

        // Features at the *caller's* precision, as a caller that widened
        // them, or built them by hand, would have.
        let xa: Tensor<3> = Tensor::random([1, 4, 128], Distribution::Default, &device);
        assert_eq!(xa.dtype(), float);

        let mut cache = decoder.new_cache(xa);
        assert_eq!(cache.dtype(), Some(DType::F16), "cross-attention cache");
        for layer in 0..decoder.n_layers() {
            let (k, v) = cache.cross_kv(layer).expect("set");
            assert_eq!((k.dtype(), v.dtype()), (DType::F16, DType::F16));
        }

        let tokens: Tensor<2, Int> =
            Tensor::from_data(TensorData::new(vec![3i64, 5], [1, 2]), &device);
        let logits = decoder.forward_cached(tokens, &mut cache);

        assert_eq!(logits.dtype(), float, "logits are the interface's");
        assert_eq!(cache.pos(), 2);
        assert_eq!(cache.dtype(), Some(DType::F16), "self-attention cache");
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
            },
        };

        type F = f32;

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let (vocab, d_model, max_ctx, layers) = (32, 128, 8, 1);

        let decoder: TextDecoder =
            TextDecoderConfig::new(vocab, d_model, max_ctx, layers).init(&device);

        let tokens: Tensor<2, Int> =
            Tensor::from_data(TensorData::new(vec![3i64, 5], [1, 2]), &device);
        let xa: Tensor<3> = Tensor::random([1, 4, d_model], Distribution::Default, &device);

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
