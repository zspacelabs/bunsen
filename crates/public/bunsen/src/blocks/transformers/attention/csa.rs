//! # Causal Self-Attention

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Initializer,
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
        causal_window_visible,
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

    /// The query/key normalization (QK-norm), sized to `head_dim` whatever
    /// feature count it names; `None` normalizes nothing, and builds no norm.
    #[config(default = "Some(NormalizationConfig::Rms(RmsNormConfig::new(0)))")]
    pub norm: Option<NormalizationConfig>,

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

    /// Whether the query, key and value projections carry a bias.
    #[config(default = "false")]
    pub qkv_bias: bool,

    /// Whether the key projection carries a bias; `None` follows `qkv_bias`.
    ///
    /// Whisper-family stacks bias their queries and values but not their
    /// keys (`Some(false)` under a `qkv_bias`).
    #[config(default = "None")]
    pub key_bias: Option<bool>,

    /// Whether the output projection carries a bias.
    #[config(default = "false")]
    pub output_bias: bool,

    /// The sliding attention window, in positions: a query attends to itself
    /// and to the `sliding_window - 1` positions before it, never further
    /// back; `None` attends to every earlier position.
    #[config(default = "None")]
    pub sliding_window: Option<usize>,

    /// A gain on the queries and keys after their normalization and
    /// rotation: both are multiplied by it, sharpening every score by its
    /// square. `None` multiplies by nothing.
    ///
    /// nanochat's 2026 layers use 1.2.
    #[config(default = "None")]
    pub qk_gain: Option<f64>,

    /// A value residual: an embedding of the input tokens, gated per KV head
    /// from the attention input, added to the values
    /// ([`forward_with_value_embed`](CausalSelfAttention::forward_with_value_embed)).
    /// `None` adds nothing, and builds no gate.
    #[config(default = "None")]
    pub value_residual: Option<ValueResidualConfig>,
}

/// Config of a [`CausalSelfAttention`]'s value residual, `ResFormer`'s in
/// nanochat's 2026 form:
///
/// ```text
/// v += gate_scale * sigmoid(gate(x[.., :gate_channels])) * ve
/// ```
///
/// The gate is one scalar per KV head, read from the attention input's first
/// `gate_channels` channels; `ve` is a `[B, T, n_kv_head * head_dim]` value
/// embedding the caller supplies.
#[derive(Config, Debug)]
pub struct ValueResidualConfig {
    /// How many of the attention input's leading channels the gate reads.
    #[config(default = "12")]
    pub gate_channels: usize,

    /// The gate's range: `(0, gate_scale)`.
    #[config(default = "3.0")]
    pub gate_scale: f64,
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
    /// order; with no norm, only rotates it.
    pub fn apply(
        self,
        x: Tensor<4>,
        norm: Option<&Normalization>,
        r_emb: &RotaryEmbedding,
    ) -> Tensor<4> {
        let Some(norm) = norm else {
            return r_emb.apply(x);
        };
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
    /// Whether the key projection carries a bias: `key_bias`, else
    /// `qkv_bias`.
    pub fn key_bias(&self) -> bool {
        self.key_bias.unwrap_or(self.qkv_bias)
    }

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
        assert!(
            self.sliding_window != Some(0),
            "sliding_window must be > 0 when set"
        );
        if let Some(vr) = &self.value_residual {
            assert!(
                vr.gate_channels > 0 && vr.gate_channels <= self.n_embed,
                "value_residual.gate_channels must be in 1..=n_embed ({}); got {}",
                self.n_embed,
                vr.gate_channels
            );
        }
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
        let norm = || {
            self.norm
                .clone()
                .map(|n| n.with_num_features(head_dim).init(device))
        };
        let linear =
            |d_in, d_out, bias| LinearConfig::new(d_in, d_out).with_bias(bias).init(device);

        Ok(CausalSelfAttention {
            layer_index,
            head_dim,
            qk_norm: self.qk_norm,
            sliding_window: self.sliding_window,
            c_q: linear(self.n_embed, self.n_head * head_dim, self.qkv_bias),
            q_norm: norm(),
            c_k: linear(self.n_embed, self.n_kv_head * head_dim, self.key_bias()),
            k_norm: norm(),
            c_v: linear(self.n_embed, self.n_kv_head * head_dim, self.qkv_bias),
            c_proj: linear(self.n_head * head_dim, self.n_embed, self.output_bias),
            qk_gain: self.qk_gain,
            value_gate: self.value_residual.as_ref().map(|vr| {
                // As upstream's init: a small positive gate, near half open.
                LinearConfig::new(vr.gate_channels, self.n_kv_head)
                    .with_bias(false)
                    .with_initializer(Initializer::Uniform {
                        min: 0.0,
                        max: 0.02,
                    })
                    .init(device)
            }),
            value_gate_scale: self.value_residual.as_ref().map(|vr| vr.gate_scale),
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
/// optional Group Query Attention (fewer KV heads than query heads), optional
/// per-head query/key normalization, optional projection biases, an
/// optional sliding window, an optional query/key gain, and an optional
/// gated value residual. Construct via
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

    /// The sliding attention window, in positions, if any.
    #[module(skip)]
    pub sliding_window: Option<usize>,

    /// Query Linear.
    pub c_q: Linear,

    /// Query Normalization; `None` without QK-norm.
    pub q_norm: Option<Normalization>,

    /// Key Linear.
    pub c_k: Linear,

    /// Key Normalization; `None` without QK-norm.
    pub k_norm: Option<Normalization>,

    /// Value Linear.
    pub c_v: Linear,

    /// Output Projection Linear.
    pub c_proj: Linear,

    /// The gain on the normalized, rotated queries and keys, if any.
    #[module(skip)]
    pub qk_gain: Option<f64>,

    /// The value residual's gate: the attention input's first channels to
    /// one scalar per KV head. `None` without a value residual.
    pub value_gate: Option<Linear>,

    /// The value residual's gate range, `(0, value_gate_scale)`.
    #[module(skip)]
    pub value_gate_scale: Option<f64>,
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
    /// The queries and keys of `input`: projected, normalized and rotated,
    /// and scaled by the gain when there is one.
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
        let q = self
            .gained(self.qk_norm.apply(q, self.q_norm.as_ref(), r_emb))
            .swap_dims(1, 2);
        let k = self
            .c_k
            .forward(input)
            .reshape([b, t, self.n_kv_head(), self.head_dim()]);
        let k = self
            .gained(self.qk_norm.apply(k, self.k_norm.as_ref(), r_emb))
            .swap_dims(1, 2);
        (q, k)
    }

    /// `x` times the query/key gain, or `x` without one.
    fn gained(
        &self,
        x: Tensor<4>,
    ) -> Tensor<4> {
        match self.qk_gain {
            Some(gain) => x.mul_scalar(gain),
            None => x,
        }
    }

    /// The values of `input`, `[B, H_kv, T, head_dim]`, with the value
    /// residual added when there is one.
    ///
    /// # Panics
    /// If `value_embed` is given to an attention without a value residual,
    /// or withheld from one with a value residual.
    fn values(
        &self,
        input: Tensor<3>,
        value_embed: Option<Tensor<3>>,
    ) -> Tensor<4> {
        let [b, t] = unpack_shape_contract!(["B", "T", "D"], &input.dims(), &["B", "T"]);
        let shape = [b, t, self.n_kv_head(), self.head_dim()];
        let v = self.c_v.forward(input.clone()).reshape(shape);
        let v = match (&self.value_gate, value_embed) {
            (None, None) => v,
            (Some(gate), Some(ve)) => {
                assert_shape_contract_periodically!(
                    ["B", "T", "V"],
                    &ve.dims(),
                    &[
                        ("B", b),
                        ("T", t),
                        ("V", self.n_kv_head() * self.head_dim())
                    ]
                );
                let channels = gate.weight.dims()[0];
                let scale = self.value_gate_scale.unwrap_or(1.0);
                // [B, T, H_kv, 1]: one gate per KV head.
                let g = burn::tensor::activation::sigmoid(gate.forward(input.slice(s![
                    ..,
                    ..,
                    ..channels
                ])))
                .mul_scalar(scale)
                .unsqueeze_dim::<4>(3);
                v + g * ve.reshape(shape)
            }
            (None, Some(_)) => {
                panic!("a value embedding for an attention without a value residual")
            }
            (Some(_), None) => {
                panic!("an attention with a value residual needs its value embedding")
            }
        };
        v.swap_dims(1, 2)
    }

    /// The attention weights an uncached [`forward`](Self::forward) of
    /// `input` applies: per query head, `softmax(q k^T / sqrt(head_dim))`
    /// over the keys at and before each query, within the sliding window when
    /// there is one.
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
        let device = q.device();
        let window = self.sliding_window.filter(|&w| w < t);
        let weights = sdpa_attn_weight(
            q,
            k,
            None,
            window.map(|w| causal_window_visible(t, t, Some(w), &device)),
            ScaledDotProductAttentionConfig::new()
                .with_is_causal(window.is_none())
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
    ///
    /// # Panics
    /// If the attention has a value residual: that needs
    /// [`forward_with_value_embed`](Self::forward_with_value_embed).
    pub fn forward(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
        kv_cache: &mut Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.forward_with_value_embed(input, r_emb, kv_cache, None)
    }

    /// Forward Pass, with the value embedding an attention with a value
    /// residual adds to its values.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, D]` sequence.
    /// - `r_emb`: a rotary embedding with len ``T``.
    /// - `kv_cache`: optional KV cache.
    /// - `value_embed`: the `[B, T, H_kv * head_dim]` value embedding of the
    ///   input's tokens; `None` for an attention without a value residual.
    ///
    /// # Returns
    /// - `[B, T, D]` attention.
    ///
    /// # Panics
    /// If `value_embed` is given to an attention without a value residual,
    /// or withheld from one with a value residual.
    pub fn forward_with_value_embed(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
        kv_cache: &mut Option<&mut KVCache>,
        value_embed: Option<Tensor<3>>,
    ) -> Tensor<3> {
        let [b, t_q] = unpack_shape_contract!(
            ["B", "T", "D"],
            &input.dims(),
            &["B", "T"],
            &[("D", self.n_embed())]
        );

        let (q, k) = self.queries_keys(input.clone(), r_emb);

        let v = self.values(input, value_embed);

        // B, H_?, T, D

        let mut k = k;
        let mut v = v;

        if let Some(kvc) = kv_cache {
            (k, v) = (*kvc).insert_kv(self.layer_index, k.clone(), v.clone());
        }
        let mut t_kv = k.dims()[2];

        // The window binds only once there are more keys than it spans. One
        // new token sees the window's last keys, which a slice keeps; anything
        // else is masked.
        let window = self.sliding_window.filter(|&w| w < t_kv);
        if let (Some(w), 1) = (window, t_q) {
            k = k.slice(s![.., .., t_kv - w..]);
            v = v.slice(s![.., .., t_kv - w..]);
            t_kv = w;
        }

        let attn_mask: Option<Tensor<2, Bool>>;
        let is_causal: bool;
        if let (Some(w), true) = (window, t_q > 1) {
            attn_mask = Some(causal_window_visible(t_q, t_kv, Some(w), &q.device()));
            is_causal = false;
        } else if kv_cache.is_none() || t_q == t_kv {
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

    /// A config non-default in every field, and the layer built from it,
    /// answer the meta alike, and the layer builds what the config names.
    #[test]
    fn test_meta_and_options() {
        let device = cpu_device();
        let cfg = CausalSelfAttentionConfig::new(4, 2, 12)
            .with_head_dim(Some(5))
            .with_norm(None)
            .with_qk_norm(QkNormPlacement::BeforeRope)
            .with_qkv_bias(true)
            .with_key_bias(Some(false))
            .with_output_bias(true)
            .with_sliding_window(Some(3));
        let csa: CausalSelfAttention = cfg.init(1, &device);
        assert_eq!(csa.n_embed(), cfg.n_embed());
        assert_eq!(csa.n_head(), cfg.n_head());
        assert_eq!(csa.n_kv_head(), cfg.n_kv_head());
        assert_eq!(csa.head_dim(), cfg.head_dim());
        assert_eq!(csa.qk_norm, cfg.qk_norm);
        assert_eq!(csa.sliding_window, cfg.sliding_window);
        assert!(csa.q_norm.is_none() && csa.k_norm.is_none());
        assert!(csa.c_q.bias.is_some());
        assert!(!cfg.key_bias());
        assert!(csa.c_k.bias.is_none());
        assert!(csa.c_v.bias.is_some());
        assert!(csa.c_proj.bias.is_some());

        let defaults: CausalSelfAttention =
            CausalSelfAttentionConfig::new(4, 2, 12).init(0, &device);
        assert!(defaults.q_norm.is_some() && defaults.k_norm.is_some());
        for linear in [
            &defaults.c_q,
            &defaults.c_k,
            &defaults.c_v,
            &defaults.c_proj,
        ] {
            assert!(linear.bias.is_none());
        }
        assert_eq!(defaults.sliding_window, None);
    }

    fn windowed(window: usize) -> (CausalSelfAttention, RotaryEmbedding, Tensor<3>) {
        let device = cpu_device();
        let csa: CausalSelfAttention = CausalSelfAttentionConfig::new(4, 2, 16)
            .with_norm(None)
            .with_qkv_bias(true)
            .with_key_bias(Some(false))
            .with_output_bias(true)
            .with_sliding_window(Some(window))
            .init(0, &device);
        let r_emb = RotaryEmbeddingConfig::new(9, csa.head_dim()).init(&device);
        let input = seeded_tensor::<3>(7, [2, 9, 16], Distribution::Default, &device);
        (csa, r_emb, input)
    }

    /// Under a sliding window, each query's output is the output an
    /// unwindowed pass computes over only the window ending at it: the
    /// window masks exactly the keys it should.
    #[test]
    fn test_sliding_window_is_the_window_alone() {
        let window = 3;
        let (csa, r_emb, input) = windowed(window);
        let full = csa.forward(input.clone(), &r_emb, &mut None);

        let mut unwindowed = csa.clone();
        unwindowed.sliding_window = None;
        for i in 0..9 {
            let start = (i + 1usize).saturating_sub(window);
            let alone = unwindowed.forward(
                input.clone().slice(s![.., start..=i]),
                &r_emb.clip_range(start..i + 1),
                &mut None,
            );
            assert_tensors_close(
                &full.clone().slice(s![.., i..=i]),
                &alone.slice(s![.., i - start..=i - start]),
                Tolerance::default(),
            );
        }

        // The weights agree: zero outside the window, each row a
        // distribution.
        let weights = csa.attention_weights(input, &r_emb);
        let sums = weights.clone().sum_dim(3);
        assert_tensors_close(&sums, &Tensor::ones_like(&sums), Tolerance::default());
        let outside = weights.clone().triu(1).abs().max().into_scalar::<f32>()
            + weights
                .tril(-(window as i64))
                .abs()
                .max()
                .into_scalar::<f32>();
        assert_eq!(outside, 0.0);
    }

    /// A windowed decode through the cache, one token at a time or a chunk
    /// after a prefix, is the windowed uncached pass.
    #[test]
    fn test_sliding_window_through_the_cache() {
        let (csa, r_emb, input) = windowed(3);
        let full = csa.forward(input.clone(), &r_emb, &mut None);

        let mut cache: KVCache = KVCacheConfig::new(2, 2, 9, csa.head_dim(), 1).init();
        let steps: Vec<Tensor<3>> = (0..9)
            .map(|i| {
                csa.forward(
                    input.clone().slice(s![.., i..=i]),
                    &r_emb.clip_range(i..i + 1),
                    &mut Some(&mut cache),
                )
            })
            .collect();
        assert_tensors_close(&Tensor::cat(steps, 1), &full, Tolerance::default());

        let mut cache: KVCache = KVCacheConfig::new(2, 2, 9, csa.head_dim(), 1).init();
        let head = csa.forward(
            input.clone().slice(s![.., ..4]),
            &r_emb.clip_range(0..4),
            &mut Some(&mut cache),
        );
        let chunk = csa.forward(
            input.slice(s![.., 4..]),
            &r_emb.clip_range(4..9),
            &mut Some(&mut cache),
        );
        assert_tensors_close(
            &Tensor::cat(vec![head, chunk], 1),
            &full,
            Tolerance::default(),
        );
    }
}
