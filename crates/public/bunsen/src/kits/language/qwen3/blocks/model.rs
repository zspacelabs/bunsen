//! The Qwen3 decoder.
use burn::{
    config::Config,
    module::Module,
    nn::norm::RmsNormConfig,
    tensor::{
        Device,
        FloatDType,
    },
};

use crate::{
    blocks::{
        norm::BunsenNormalizationConfig,
        transformers::{
            attention::csa::{
                CausalSelfAttentionConfig,
                QkNormPlacement,
            },
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::UniformStackConfig,
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
                TokenEmbeddingConfig,
            },
            head::LmHeadConfig,
            mlp::TransformerGatedMlpConfig,
            models::language_decoder::{
                LmAttentionOnlyKind,
                LmDecoderMeta,
                LmDecoderModel,
                LmDecoderVariantConfig,
                LmDecoderWrapper,
            },
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::transformers::attention::{
        KVCacheConfig,
        KVCacheModel,
    },
};

/// The positions a Qwen3 cache allocates up front by default: one growth
/// chunk. Qwen3 builds its rotary tables to order, so it has no positional cap
/// of its own; a caller that knows a decode's length sizes (and caps) the
/// cache to it.
pub const QWEN3_KV_CACHE_PRE_SIZE: usize = 1024;

/// The cache a Qwen3 of this geometry decodes with, before it is built: its
/// KV heads, head width and layer count, [`QWEN3_KV_CACHE_PRE_SIZE`]
/// positions up front, uncapped; the built model's
/// ([`LmDecoderModel`]'s) is the same.
impl KVCacheModel for Qwen3Config {
    fn kv_cache_config(&self) -> KVCacheConfig {
        KVCacheConfig::for_decode(
            self.n_kv_heads(),
            QWEN3_KV_CACHE_PRE_SIZE,
            self.head_dim(),
            self.n_layers(),
        )
    }
}

/// Config for [`Qwen3Model`]: `transformers`' `Qwen3Config`, as far as a
/// decode needs it.
///
/// Implements [`LmDecoderMeta`].
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

    /// Whether the output head is the embedding (`transformers`'
    /// `tie_word_embeddings`); otherwise it is a linear map of its own.
    #[config(default = "true")]
    pub tied_head: bool,

    /// The dtype attention's softmax runs in. `transformers` runs it in f32
    /// for a half model (`softmax(..., dtype=torch.float32)`), which this
    /// keeps; in an f32 model it changes nothing.
    #[config(default = "Some(FloatDType::F32)")]
    pub softmax_dtype: Option<FloatDType>,

    /// The dtype the logits leave the model in: f32, as `transformers`
    /// generates from (`logits.float()`), whatever the model computes in.
    /// `None` leaves them in the model's dtype.
    #[config(default = "Some(FloatDType::F32)")]
    pub logits_dtype: Option<FloatDType>,
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

    /// `Qwen3-4B`'s geometry: 36 layers of width 2560, 32 query and 8
    /// key/value heads of width 128, and a 9728-wide feed-forward.
    pub fn qwen3_4b() -> Self {
        Self::new(151_936, 2560, 36, 32, 8, 128, 9728)
    }

    /// The geometry of `Qwen3-4B-Instruct-2507` and `Qwen3-4B-Thinking-2507`:
    /// [`qwen3_4b`](Self::qwen3_4b)'s, with a rotary base of 5 * 10^6.
    pub fn qwen3_4b_2507() -> Self {
        Self::qwen3_4b().with_rope_theta(5_000_000)
    }

    /// `Qwen3-8B`'s geometry: 36 layers of width 4096, 32 query and 8
    /// key/value heads of width 128, a 12288-wide feed-forward, and an output
    /// head of its own.
    pub fn qwen3_8b() -> Self {
        Self::new(151_936, 4096, 36, 32, 8, 128, 12288).with_tied_head(false)
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

    /// The output head: tied to the embedding or a projection of its own
    /// (`tied_head`), with no bias, its logits in `logits_dtype`.
    pub fn head(&self) -> LmHeadConfig {
        LmHeadConfig::new(self.d_model, self.n_vocab)
            .with_tied(self.tied_head)
            .with_logits_dtype(self.logits_dtype)
    }

    /// The decoder stack: `n_layers` alike, each RMS norms, a `SwiGLU`, and
    /// grouped-query attention with QK-norm before the rotary embedding.
    pub fn stack(&self) -> UniformStackConfig {
        let norm =
            |n| BunsenNormalizationConfig::Rms(RmsNormConfig::new(n).with_epsilon(self.rms_eps));
        UniformStackConfig::new(
            self.n_layers,
            CausalDecoderBlockConfig::new(
                CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, self.d_model)
                    .with_head_dim(Some(self.head_dim))
                    .with_norm(Some(norm(self.head_dim)))
                    .with_qk_norm(QkNormPlacement::BeforeRope)
                    .with_softmax_dtype(self.softmax_dtype)
                    .into(),
                TransformerGatedMlpConfig::new(self.d_model, self.d_ff).into(),
            )
            .with_norm(norm(0)),
            norm(0),
            self.rotary(0),
        )
    }

    /// The token embedding: a plain table, unscaled and unnormalized.
    pub fn embedding(&self) -> TokenEmbeddingConfig {
        TokenEmbeddingConfig::new(self.n_vocab, self.d_model)
    }

    /// The language model: [`embedding`](Self::embedding),
    /// [`head`](Self::head) and [`stack`](Self::stack), its KV cache
    /// [`QWEN3_KV_CACHE_PRE_SIZE`] positions up front.
    pub fn lm(&self) -> LmDecoderVariantConfig {
        LmDecoderVariantConfig::new(self.embedding(), self.head(), self.stack().into())
            .with_kv_cache_pre_size(QWEN3_KV_CACHE_PRE_SIZE)
    }
}

impl LmDecoderMeta for Qwen3Config {
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
        Ok(Qwen3Model {
            lm: self.lm().try_init(device)?,
        })
    }
}

/// The Qwen3 decoder-only language model (Qwen team 2025): the dense Qwen3
/// checkpoints, their head tied to the embedding (up to 4B) or a linear map
/// of its own (8B), and the text model of Qwen3-ASR.
///
/// An [`LmDecoderModel`] whose stack ([`Qwen3Config::stack`]) is RMS norms, a
/// `SwiGLU`, and QK-norm before the rotary embedding;
/// [`transformers_rekey`](crate::kits::language::qwen3::weights::transformers_rekey)
/// maps `transformers`' `Qwen3ForCausalLM` names onto the
/// [`LmDecoderModel`]'s, and the load is into [`lm`](Self::lm).
///
/// An [`LmAttentionOnlyKind`] [`LmDecoderWrapper`]: it decodes through its
/// stack's
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache)
/// ([`StackedLm`](crate::kits::support::decoding::StackedLm)), or a flat
/// [`KVCache`](crate::ops::transformers::attention::KVCache)
/// ([`LmKvForwardHidden`](crate::blocks::transformers::models::language_decoder::LmKvForwardHidden),
/// [`KVCacheModel`]).
///
/// Built by [`Qwen3Config`].
/// Implements [`LmDecoderMeta`].
#[derive(Module, Debug)]
pub struct Qwen3Model {
    /// The language model.
    pub lm: LmDecoderModel,
}

impl LmDecoderWrapper for Qwen3Model {
    type Kind = LmAttentionOnlyKind;

    fn lm(&self) -> &LmDecoderModel {
        &self.lm
    }

    fn lm_mut(&mut self) -> &mut LmDecoderModel {
        &mut self.lm
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        prelude::Int,
    };

    use super::*;
    use crate::{
        blocks::transformers::{
            head::LmHeadMeta,
            models::language_decoder::{
                LmDecoderOps,
                LmKvForwardHidden,
            },
        },
        kits::support::decoding::{
            SearchConfig,
            StackedLm,
            decode_cached,
        },
        ops::{
            embedding::unembed,
            transformers::attention::KVCacheStorage,
        },
        support::testing::cpu_device,
    };

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
        assert_eq!(model.lm.stack.rotary.base, config.rope_theta);
        assert_eq!(
            model.lm.stack.layers[0]
                .attn
                .branch
                .attention()
                .c_q
                .weight
                .dims(),
            [12, 18]
        );
        assert!(model.lm.head.is_tied(), "tied by default");

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

    /// An untied head is a linear map of its own, which the logits are:
    /// not the embedding's.
    #[test]
    fn test_untied_head() {
        let device = cpu_device();
        let model: Qwen3Model = Qwen3Config::new(40, 12, 1, 3, 1, 6, 20)
            .with_tied_head(false)
            .init(&device);
        let head = model.lm.head.proj.as_ref().expect("an untied head");
        assert_eq!(head.weight.dims(), [12, 40]);
        assert!(head.bias.is_none());

        let tokens = Tensor::<2, Int>::from_data([[1, 5, 7]], &device);
        let hidden = model.forward_hidden(model.embed(tokens), &model.rotary(3, &device), None);
        let logits = model.logits(hidden.clone());
        crate::support::testing::assert_tensors_close(
            &logits,
            &head.forward(hidden.clone()),
            burn::tensor::Tolerance::default(),
        );
        let tied = unembed(model.lm.embedding.weight.val(), hidden);
        assert!(
            !logits.all_close(tied, None, None),
            "the logits are not the embedding's"
        );
    }

    /// A greedy decode through the loop is the argmax of an uncached
    /// forward over the prompt and every token before it; a beam of three
    /// over three copies of the prompt finishes each of its candidates
    /// within the budget.
    #[test]
    fn test_decode_cached() {
        let device = cpu_device();
        let model: Qwen3Model = Qwen3Config::new(40, 12, 2, 3, 1, 6, 20).init(&device);
        let prompt = vec![1i64, 5, 7];
        let ids = |ids: &[i64], rows: usize| {
            let data: Vec<i64> = (0..rows).flat_map(|_| ids.iter().copied()).collect();
            Tensor::<2, Int>::from_data(
                burn::prelude::TensorData::new(data, [rows, ids.len()]),
                &device,
            )
        };

        let budget = 5;
        let mut state = model.decode_state(1, prompt.len() + budget, KVCacheStorage::Auto, &device);
        let first = model.prefill(model.embed(ids(&prompt, 1)), &mut state);
        let mut search = SearchConfig::new().init_search(&[39], 0);
        let out = decode_cached(
            &model,
            &mut state,
            vec![prompt.clone()],
            first,
            budget,
            search.as_mut(),
            &[],
        );
        let generated = out[0][0].tokens_with_stop();
        assert!(!generated.is_empty() && generated.len() <= budget);

        let mut seen = prompt.clone();
        for &token in &generated {
            let r_emb = model.rotary(seen.len(), &device);
            let hidden = model.forward_hidden(model.embed(ids(&seen, 1)), &r_emb, None);
            let n = hidden.dims()[1];
            let expected = model
                .logits(hidden.slice_dim(1, n - 1..n))
                .argmax(2)
                .into_scalar::<i64>();
            assert_eq!(token, expected, "after {seen:?}");
            seen.push(token);
        }

        let mut state = model.decode_state(3, prompt.len() + budget, KVCacheStorage::Auto, &device);
        let first = model.prefill(model.embed(ids(&prompt, 3)), &mut state);
        let mut beam = SearchConfig::new().with_beam_size(3).init_search(&[39], 0);
        let out = decode_cached(
            &model,
            &mut state,
            vec![prompt.clone(); 3],
            first,
            budget,
            beam.as_mut(),
            &[],
        );
        assert_eq!(out.len(), 1, "one group");
        assert_eq!(out[0].len(), 3);
        assert!(out[0].iter().all(|c| c.tokens.len() <= budget));
    }
}
