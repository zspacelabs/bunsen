//! # Causal Self-Attention

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
        norm::{
            Normalization,
            NormalizationConfig,
            RmsNormConfig,
        },
    },
    prelude::{
        Bool,
        Int,
        s,
    },
    tensor::Device,
};

use crate::{
    blocks::transformers::embedding::RotaryEmbedding,
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
    errors::{
        BunsenResult,
        WithOkOrPanic,
    },
    ops::transformers::attention::{
        KVCache,
        ScaledDotProductAttentionConfig,
        scaled_dot_product_attention,
        sdpa_attn_weight,
    },
};

/// Common meta for [`CausalSelfAttention`] and [`CausalSelfAttentionConfig`].
pub trait CausalSelfAttentionMeta {
    /// Returns the size of the input and output.
    fn n_embed(&self) -> usize;

    /// Returns the number of heads.
    fn n_head(&self) -> usize;

    /// Returns the number of KV heads.
    fn n_kv_head(&self) -> usize;

    /// Returns the size of each head.
    fn head_dim(&self) -> usize {
        self.n_embed() / self.n_head()
    }

    /// Group Query Attention.
    fn gqa_enabled(&self) -> bool {
        self.n_head() != self.n_kv_head()
    }
}

/// Config for [`CausalSelfAttention`].
#[derive(Config, Debug)]
pub struct CausalSelfAttentionConfig {
    /// Number of Heads.
    pub n_head: usize,

    /// Number of KV Heads.
    pub n_kv_head: usize,

    /// Embedding Size.
    pub n_embed: usize,

    /// Attention Normalization.
    /// This normalization will be adapted to the appropriate feature count.
    #[config(default = "NormalizationConfig::Rms(RmsNormConfig::new(0))")]
    pub norm: NormalizationConfig,

    /// The width of each head; `None` is `n_embed / n_head`.
    ///
    /// A model whose heads are wider or narrower than that (Qwen3's are 128
    /// over 16 heads of a 1024-wide model) names it here; the query and
    /// output projections then map `n_embed` to and from `n_head * head_dim`.
    #[config(default = "None")]
    pub head_dim: Option<usize>,

    /// Whether the query/key normalization applies before or after the
    /// rotary embedding.
    #[config(default = "QkNormPlacement::AfterRope")]
    pub qk_norm: QkNormPlacement,
}

/// Where [`CausalSelfAttention`] normalizes its queries and keys, relative to
/// the rotary embedding.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum QkNormPlacement {
    /// Rotate, then normalize: `nanochat`'s order.
    AfterRope,

    /// Normalize, then rotate: Qwen3's order.
    BeforeRope,
}

impl QkNormPlacement {
    /// Normalizes and rotates `x`, a `[B, T, H, D]` query or key, in this
    /// order.
    pub fn apply(
        self,
        x: Tensor<4>,
        norm: &Normalization,
        r_emb: &RotaryEmbedding,
    ) -> Tensor<4> {
        match self {
            Self::AfterRope => norm.forward(r_emb.apply(x)),
            Self::BeforeRope => r_emb.apply(norm.forward(x)),
        }
    }
}

impl CausalSelfAttentionMeta for CausalSelfAttentionConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn n_head(&self) -> usize {
        self.n_head
    }

    fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.n_embed / self.n_head.max(1))
    }
}

impl CausalSelfAttentionConfig {
    /// Validate the config.
    pub fn validate(&self) {
        assert!(
            self.n_head > 0,
            "n_head must be > 0; got n_head={}",
            self.n_head
        );
        assert!(
            self.head_dim.is_some() || self.n_embed.is_multiple_of(self.n_head),
            "n_embed ({}) must be multiple of n_head ({})",
            self.n_embed,
            self.n_head
        );
        assert!(
            self.head_dim.is_some() || self.n_embed.is_multiple_of(self.n_kv_head),
            "n_embed ({}) must be multiple of n_kv_head ({})",
            self.n_embed,
            self.n_kv_head
        );
        assert!(
            self.head_dim() > 0,
            "head_dim must be > 0; got n_embed={}/n_head={}",
            self.n_embed,
            self.n_head
        );
        assert!(
            self.n_kv_head <= self.n_head,
            "n_kv_head must be <= n_head; got n_kv_head={}, n_head={}",
            self.n_kv_head,
            self.n_head
        );
        assert!(
            self.n_head.is_multiple_of(self.n_kv_head),
            "n_head must be divisible by n_kv_head; got n_head={}, n_kv_head={}",
            self.n_head,
            self.n_kv_head
        );
    }
}

impl CausalSelfAttentionConfig {
    /// Initializes the module.
    pub fn try_init(
        &self,
        layer_index: usize,
        device: &Device,
    ) -> BunsenResult<CausalSelfAttention> {
        self.validate();
        let head_dim = self.head_dim();

        Ok(CausalSelfAttention {
            layer_index,
            head_dim,
            qk_norm: self.qk_norm,
            c_q: LinearConfig::new(self.n_embed, self.n_head * head_dim)
                .with_bias(false)
                .init(device),
            q_norm: self
                .norm
                .clone()
                .with_num_features(self.head_dim())
                .init(device),
            c_k: LinearConfig::new(self.n_embed, self.n_kv_head * head_dim)
                .with_bias(false)
                .init(device),
            k_norm: self
                .norm
                .clone()
                .with_num_features(self.head_dim())
                .init(device),
            c_v: LinearConfig::new(self.n_embed, self.n_kv_head * head_dim)
                .with_bias(false)
                .init(device),
            c_proj: LinearConfig::new(self.n_head * head_dim, self.n_embed)
                .with_bias(false)
                .init(device),
        })
    }

    /// Initializes the module, or panic.
    pub fn init(
        &self,
        layer_index: usize,
        device: &Device,
    ) -> CausalSelfAttention {
        self.try_init(layer_index, device).ok_or_panic()
    }
}

/// Causal Self-Attention Module
///
/// Multi-head causal self-attention with rotary positional embeddings,
/// optional Group Query Attention (fewer KV heads than query heads), and
/// per-head query/key normalization. Construct via
/// [`CausalSelfAttentionConfig`] and `.init(layer_index, device)`, then call
/// [`forward`](CausalSelfAttention::forward) with an input sequence, a
/// [`RotaryEmbedding`], and an optional [`KVCache`].
///
/// Built by [`CausalSelfAttentionConfig`].
#[derive(Module, Debug)]
pub struct CausalSelfAttention {
    /// Layer Index.
    pub layer_index: usize,

    /// Head Dimension.
    pub head_dim: usize,

    /// Where the query/key normalization applies.
    #[module(skip)]
    pub qk_norm: QkNormPlacement,

    /// Query Linear.
    pub c_q: Linear,

    /// Query Normalization.
    pub q_norm: Normalization,

    /// Key Linear.
    pub c_k: Linear,

    /// Key Normalization.
    pub k_norm: Normalization,

    /// Value Linear.
    pub c_v: Linear,

    /// Output Projection Linear.
    pub c_proj: Linear,
}

impl CausalSelfAttentionMeta for CausalSelfAttention {
    fn n_embed(&self) -> usize {
        self.c_q.weight.dims()[0]
    }

    fn n_head(&self) -> usize {
        self.c_q.weight.dims()[1] / self.head_dim()
    }

    fn n_kv_head(&self) -> usize {
        self.c_k.weight.dims()[1] / self.head_dim()
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl CausalSelfAttention {
    /// The queries and keys of `input`: projected, normalized and rotated.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, D]` sequence.
    /// - `r_emb`: a rotary embedding with len ``T``.
    ///
    /// # Returns
    /// - `[B, H_q, T, head_dim]` queries and `[B, H_kv, T, head_dim]` keys.
    fn queries_keys(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
    ) -> (Tensor<4>, Tensor<4>) {
        let [b, t] = unpack_shape_contract!(["B", "T", "D"], &input.dims(), &["B", "T"]);
        let q = self
            .c_q
            .forward(input.clone())
            .reshape([b, t, self.n_head(), self.head_dim()]);
        let q = self.qk_norm.apply(q, &self.q_norm, r_emb).swap_dims(1, 2);
        let k = self
            .c_k
            .forward(input)
            .reshape([b, t, self.n_kv_head(), self.head_dim()]);
        let k = self.qk_norm.apply(k, &self.k_norm, r_emb).swap_dims(1, 2);
        (q, k)
    }

    /// The attention weights an uncached [`forward`](Self::forward) of
    /// `input` applies: per query head, `softmax(q k^T / sqrt(head_dim))`
    /// over the keys at and before each query.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, D]` sequence.
    /// - `r_emb`: a rotary embedding with len ``T``.
    ///
    /// # Returns
    /// - `[B, H_q, T, T]`, each row a distribution over the keys; zero above
    ///   the diagonal.
    pub fn attention_weights(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
    ) -> Tensor<4> {
        let [b, t] = unpack_shape_contract!(
            ["B", "T", "D"],
            &input.dims(),
            &["B", "T"],
            &[("D", self.n_embed())]
        );
        let (q, k) = self.queries_keys(input, r_emb);
        let weights = sdpa_attn_weight(
            q,
            k,
            None,
            None,
            ScaledDotProductAttentionConfig::new()
                .with_is_causal(true)
                .with_enable_gqa(self.gqa_enabled()),
        );
        assert_shape_contract_periodically!(
            ["B", "H", "T", "T"],
            &weights.dims(),
            &[("B", b), ("H", self.n_head()), ("T", t)]
        );
        weights
    }

    /// Forward Pass.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, D]` sequence.
    /// - `r_emb`: a rotary embedding with len ``T``.
    /// - `kv_cache`: optional KV cache.
    ///
    /// # Returns
    /// - `[B, T, D]` attention.
    pub fn forward(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
        kv_cache: &mut Option<&mut KVCache>,
    ) -> Tensor<3> {
        let [b, t_q] = unpack_shape_contract!(
            ["B", "T", "D"],
            &input.dims(),
            &["B", "T"],
            &[("D", self.n_embed())]
        );

        let (q, k) = self.queries_keys(input.clone(), r_emb);

        let v = self
            .c_v
            .forward(input)
            .reshape([b, t_q, self.n_kv_head(), self.head_dim()])
            .swap_dims(1, 2);

        // B, H_?, T, D

        let mut k = k;
        let mut v = v;

        if let Some(kvc) = kv_cache {
            (k, v) = (*kvc).insert_kv(self.layer_index, k.clone(), v.clone());
        }
        let t_kv = k.dims()[2];

        let attn_mask: Option<Tensor<2, Bool>>;
        let is_causal: bool;
        if kv_cache.is_none() || t_q == t_kv {
            attn_mask = None;
            is_causal = true;
        } else if t_q == 1 {
            attn_mask = None;
            is_causal = false;
        } else {
            let device = q.device();
            let mut mask = Tensor::<2, Bool>::empty([t_q, t_kv], &device);
            let prefix_len = t_kv - t_q;
            if prefix_len > 0 {
                mask = mask.slice_fill(s![.., ..prefix_len], true);
            }
            // Causal within the chunk: each new token sees itself and the
            // new tokens before it.
            let fill = Tensor::<2, Int>::ones([t_q, t_q], &device).tril(0).bool();
            mask = mask.slice_assign(s![.., prefix_len..], fill);

            attn_mask = Some(mask);
            is_causal = false;
        };

        let cfg = ScaledDotProductAttentionConfig::new()
            .with_is_causal(is_causal)
            .with_enable_gqa(self.gqa_enabled());

        let y = scaled_dot_product_attention(q, k, v, None, attn_mask, cfg)
            .swap_dims(1, 2)
            .reshape([
                b as i32,
                t_q as i32,
                (self.n_head() * self.head_dim()) as i32,
            ]);

        let y = self.c_proj.forward(y);

        assert_shape_contract_periodically!(
            ["B", "T", "D"],
            &y.dims(),
            &[("B", b), ("T", t_q), ("D", self.n_embed())]
        );

        y
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        blocks::transformers::embedding::{
            RotaryEmbedding,
            RotaryEmbeddingConfig,
        },
        burner::module::ModuleInit,
        contracts::assert_shape_contract,
        ops::transformers::attention::KVCacheConfig,
        support::testing::{
            DeviceMemoryGuard,
            assert_tensors_close,
            cpu_device,
            performance_device,
            seeded_tensor,
        },
    };

    #[test]
    fn test_csa_config() {
        let cfg = CausalSelfAttentionConfig::new(3, 2, 4);
        assert_eq!(cfg.n_embed, 4);
        assert_eq!(cfg.n_head, 3);
        assert_eq!(cfg.n_kv_head, 2);
        assert_eq!(cfg.n_embed(), 4);
        assert_eq!(cfg.n_head(), 3);
        assert_eq!(cfg.n_kv_head(), 2);
        assert_eq!(cfg.head_dim(), 1);
    }

    #[test]
    #[allow(unused)]
    fn test_csa_forward() {
        let device = cpu_device();

        let batch = 1;
        let seq_len = 10;

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;
        let layer_index = 12;

        let cfg = CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed);
        let csa: CausalSelfAttention = cfg.init(layer_index, &device);

        let head_dim = csa.head_dim();

        let re_cfg = RotaryEmbeddingConfig::new(seq_len, csa.head_dim());
        let r_emb: RotaryEmbedding = re_cfg.init(&device);

        let input: Tensor<3> =
            Tensor::random([batch, seq_len, n_embed], Distribution::Default, &device);

        let mut kv_cache: Option<&mut KVCache> = None;

        let output = csa.forward(input.clone(), &r_emb, &mut kv_cache);
        assert_shape_contract!(
            ["B", "T", "D"],
            &output.dims(),
            &[("B", batch), ("T", seq_len), ("D", n_embed)]
        );
    }

    /// The weights are what the forward pass applies: each row a causal
    /// distribution, and the values they weigh, projected, the output.
    #[test]
    fn test_attention_weights_are_the_forwards() {
        let device = cpu_device();
        let (batch, seq_len, n_embed, n_head, n_kv_head) = (2, 6, 32, 4, 2);
        let csa: CausalSelfAttention =
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed).init(0, &device);
        let head_dim = csa.head_dim();
        let r_emb: RotaryEmbedding = RotaryEmbeddingConfig::new(seq_len, head_dim).init(&device);
        let input: Tensor<3> =
            Tensor::random([batch, seq_len, n_embed], Distribution::Default, &device);

        let weights = csa.attention_weights(input.clone(), &r_emb);
        assert_eq!(weights.dims(), [batch, n_head, seq_len, seq_len]);
        let sums = weights.clone().sum_dim(3);
        assert_tensors_close(&sums, &Tensor::ones_like(&sums), Tolerance::default());
        let upper = weights.clone().triu(1).abs().max().into_scalar::<f32>();
        assert_eq!(upper, 0.0);

        let v = csa
            .c_v
            .forward(input.clone())
            .reshape([batch, seq_len, n_kv_head, head_dim])
            .swap_dims(1, 2);
        let v = crate::ops::repeat::repeat_interleave::<4, 5, _>(v, n_head / n_kv_head, 1);
        let applied = csa
            .c_proj
            .forward(weights.matmul(v).swap_dims(1, 2).reshape([
                batch,
                seq_len,
                n_head * head_dim,
            ]));
        let forward = csa.forward(input, &r_emb, &mut None);
        assert_tensors_close(&applied, &forward, Tolerance::default());
    }

    /// A chunk of several new tokens after a cached prefix attends as the
    /// uncached pass does: each new token sees the prefix, the earlier new
    /// tokens, and itself.
    #[test]
    #[serial]
    fn test_csa_cached_chunk_matches_full_pass() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let [batch, seq_len, n_embed] = [2, 6, 16];
        let prefix = 3;

        let csa: CausalSelfAttention =
            CausalSelfAttentionConfig::new(4, 2, n_embed).init(0, &device);
        let r_emb: RotaryEmbedding =
            RotaryEmbeddingConfig::new(seq_len, csa.head_dim()).init(&device);
        let input =
            seeded_tensor::<3>(1, [batch, seq_len, n_embed], Distribution::Default, &device);

        let full = csa.forward(input.clone(), &r_emb, &mut None);

        let mut cache: KVCache =
            KVCacheConfig::new(batch, csa.n_kv_head(), seq_len, csa.head_dim(), 1).init();
        let head = csa.forward(
            input.clone().slice(s![.., ..prefix]),
            &r_emb.clip_range(0..prefix),
            &mut Some(&mut cache),
        );
        let chunk = csa.forward(
            input.slice(s![.., prefix..]),
            &r_emb.clip_range(prefix..seq_len),
            &mut Some(&mut cache),
        );

        assert_tensors_close(
            &head,
            &full.clone().slice(s![.., ..prefix]),
            Tolerance::default(),
        );
        assert_tensors_close(&chunk, &full.slice(s![.., prefix..]), Tolerance::default());
    }

    /// Heads wider than `n_embed / n_head` project to and from
    /// `n_head * head_dim`; either norm placement keeps the shape.
    #[test]
    fn test_explicit_head_dim() {
        use crate::{
            blocks::transformers::embedding::RotaryEmbeddingConfig,
            burner::module::ModuleInit,
            support::testing::cpu_device,
        };
        let device = cpu_device();
        for placement in [QkNormPlacement::AfterRope, QkNormPlacement::BeforeRope] {
            let cfg = CausalSelfAttentionConfig::new(2, 1, 12)
                .with_head_dim(Some(8))
                .with_qk_norm(placement);
            assert_eq!(cfg.head_dim(), 8);
            let attn = cfg.init(0, &device);
            assert_eq!(attn.n_head(), 2);
            assert_eq!(attn.n_kv_head(), 1);
            assert_eq!(attn.c_proj.weight.dims(), [16, 12]);
            let r_emb = RotaryEmbeddingConfig::new(5, 8).init(&device);
            let y = attn.forward(Tensor::zeros([2, 5, 12], &device), &r_emb, &mut None);
            assert_eq!(y.dims(), [2, 5, 12]);
        }
    }
}
