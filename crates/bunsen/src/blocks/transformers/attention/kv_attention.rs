//! # Key/value cached attention over `MultiHeadAttention` weights.
//!
//! A genuine KV cache: keys and values are kept **projected and head-split**,
//! so a decode step attends over the whole history while projecting only the
//! new tokens, and the attention itself only scores the new queries.
//!
//! This reuses [`MultiHeadAttention`]'s own `Linear` layers rather than
//! replacing the module, so weights, loading and any cross-checks stay exactly
//! as they are — only the forward path differs.
//!
//! ## Why not `MhaCache`
//!
//! [`MhaCache`](burn::nn::attention::MhaCache) caches the *projections* and
//! expects the whole sequence back on every call, so its attention still scores
//! the full prefix each step. That is a real saving for cross-attention, whose
//! keys and values never change, but it leaves self-attention quadratic. These
//! functions take only the new tokens.
//!
//! ## Why not `KVCache`
//!
//! [`KVCache`](super::KVCache) is a preallocated *multi-layer* cache: a single
//! `Tensor<B, 6>` over `[layers, kv, batch, heads, seq, d_k]`, grown in chunks,
//! and a `Module` in its own right. It wants the geometry — layer count, head
//! count, head dimension, batch size, and a sequence bound — declared up front,
//! and it serves bunsen's own attention stack
//! ([`CausalSelfAttention`](super::CausalSelfAttention) and the nanochat kit).
//!
//! [`AttnKv`] is a plain per-layer value over burn's [`MultiHeadAttention`]
//! weights. It declares no geometry, and it also covers a case `KVCache` does
//! not model: cross-attention keys and values, projected once from the encoder
//! output and then reused unchanged rather than grown.
//!
//! Reach for `KVCache` when the geometry is known up front and the allocation
//! matters; reach for these when the attention is burn's own and the cache is
//! per-layer.
//!
//! ## Matching the uncached path
//!
//! The arithmetic mirrors `MultiHeadAttention::forward` exactly — the same
//! `1/sqrt(d_k)` scaling, the same `min_float` mask fill, the same
//! `quiet_softmax` switch — so a cached decode reproduces an uncached one.
//! `layer_norm_self_attn_kv`'s contract test is what holds that true.
//!
//! Note this path deliberately omits the attention-score dropout that
//! `MultiHeadAttention::attn_scores` applies: caching is an inference concern,
//! and burn's `Dropout` is a no-op outside training anyway.

use burn::{
    Tensor,
    nn::{
        LayerNorm,
        attention::MultiHeadAttention,
    },
    prelude::{
        Backend,
        Bool,
    },
    tensor::activation::{
        quiet_softmax,
        softmax,
    },
};

/// Projected, head-split keys and values: `[batch, heads, seq, d_k]`.
///
/// Built by [`project_kv`]. Self-attention grows one of these a step at a
/// time; cross-attention builds one per layer and reuses it untouched.
#[derive(Clone, Debug)]
pub struct AttnKv<B: Backend> {
    /// `[batch, heads, seq, d_k]` keys.
    pub key: Tensor<B, 4>,

    /// `[batch, heads, seq, d_k]` values.
    pub value: Tensor<B, 4>,
}

impl<B: Backend> AttnKv<B> {
    /// The number of cached positions.
    pub fn seq_len(&self) -> usize {
        self.key.dims()[2]
    }

    /// The batch size.
    pub fn batch_size(&self) -> usize {
        self.key.dims()[0]
    }

    /// Appends `next` along the sequence axis.
    ///
    /// This reallocates: every call copies the whole cache, so growing a
    /// self-attention cache one token at a time costs O(T^2) copying across a
    /// decode of length T. That is not worth avoiding at a few hundred
    /// positions — Whisper's text window is 448 — but a long-context decode
    /// wants a preallocated cache instead; see [`KVCache`](super::KVCache).
    pub fn concat(
        self,
        next: Self,
    ) -> Self {
        Self {
            key: Tensor::cat(vec![self.key, next.key], 2),
            value: Tensor::cat(vec![self.value, next.value], 2),
        }
    }
}

/// Projects `[batch, seq, d_model]` into head-split keys and values.
///
/// For cross-attention this is the whole win: call it once per layer against
/// the encoder output and the result serves every decode step.
pub fn project_kv<B: Backend>(
    mha: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
) -> AttnKv<B> {
    AttnKv {
        key: split_heads(mha, mha.key.forward(x.clone())),
        value: split_heads(mha, mha.value.forward(x)),
    }
}

/// `[batch, seq, d_model]` -> `[batch, heads, seq, d_k]`.
///
/// Mirrors `MultiHeadAttention::attention_linear`, minus the projection.
fn split_heads<B: Backend>(
    mha: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, seq, _] = x.dims();
    x.reshape([batch, seq, mha.n_heads, mha.d_k])
        .swap_dims(1, 2)
}

/// Scaled dot-product attention over head-split tensors, then the output
/// projection.
///
/// # Arguments
/// * `q`: `[batch, heads, seq_new, d_k]`.
/// * `kv`: keys and values over `[batch, heads, seq_total, d_k]`.
/// * `mask`: optional `[batch, seq_new, seq_total]`.
///
/// # Returns
/// `[batch, seq_new, d_model]`.
fn attend<B: Backend>(
    mha: &MultiHeadAttention<B>,
    q: Tensor<B, 4>,
    kv: &AttnKv<B>,
    mask: Option<Tensor<B, 3, Bool>>,
) -> Tensor<B, 3> {
    let [batch, _, seq_new, _] = q.dims();

    let scores = q
        .matmul(kv.key.clone().transpose())
        .div_scalar((mha.d_k as f32).sqrt());

    let scores = match mask {
        Some(mask) => {
            let [mask_batch, seq_q, seq_k] = mask.dims();
            scores.mask_fill(mask.reshape([mask_batch, 1, seq_q, seq_k]), mha.min_float)
        }
        None => scores,
    };

    let weights = if mha.quiet_softmax {
        quiet_softmax(scores, 3)
    } else {
        softmax(scores, 3)
    };

    let context =
        weights
            .matmul(kv.value.clone())
            .swap_dims(1, 2)
            .reshape([batch, seq_new, mha.d_model]);

    mha.output.forward(context)
}

/// Layer-normed self-attention over a growing key/value cache.
///
/// Appends this step's keys and values to `cache` and attends over everything
/// cached. Unlike the uncached path, `x` is only the **new** tokens.
///
/// # Arguments
/// * `x`: `[batch, seq_new, d_model]` — the new token(s) only.
/// * `mask`: optional `[batch, seq_new, seq_past + seq_new]`. Build it with
///   [`causal_mask(seq_new, seq_past, device)`](super::causal_mask). A single
///   new token needs no mask: it may attend to everything already cached.
/// * `cache`: `None` at the start of a stream; grown in place thereafter.
///
/// # Returns
/// `[batch, seq_new, d_model]`.
pub fn layer_norm_self_attn_kv<B: Backend>(
    layer_norm: &LayerNorm<B>,
    mha: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
    mask: Option<Tensor<B, 3, Bool>>,
    cache: &mut Option<AttnKv<B>>,
) -> Tensor<B, 3> {
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract!(
        ["batch", "seq_new", "d_model"],
        &x,
        &[("d_model", mha.d_model)],
    );

    let normed = layer_norm.forward(x);

    let q = split_heads(mha, mha.query.forward(normed.clone()));
    let fresh = project_kv(mha, normed);

    let grown = match cache.take() {
        Some(past) => past.concat(fresh),
        None => fresh,
    };

    let out = attend(mha, q, &grown, mask);
    *cache = Some(grown);

    out
}

/// Layer-normed cross-attention against a fixed key/value cache.
///
/// The keys and values come from the encoder output and never change, so
/// [`project_kv`] runs once per layer per decode rather than once per token.
/// No mask: every query may attend to the whole encoded sequence.
///
/// # Arguments
/// * `x`: `[batch, seq_new, d_model]` — the new token(s) only.
/// * `kv`: from [`project_kv`] over the encoder output.
///
/// # Returns
/// `[batch, seq_new, d_model]`.
pub fn layer_norm_cross_attn_kv<B: Backend>(
    layer_norm: &LayerNorm<B>,
    mha: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
    kv: &AttnKv<B>,
) -> Tensor<B, 3> {
    #[cfg(any(debug_assertions, test))]
    {
        crate::contracts::assert_shape_contract!(
            ["batch", "seq_new", "d_model"],
            &x,
            &[("d_model", mha.d_model)],
        );
        assert_eq!(
            kv.batch_size(),
            x.dims()[0],
            "cross-attention cache batch ({}) != query batch ({})",
            kv.batch_size(),
            x.dims()[0],
        );
    }

    let q = split_heads(mha, mha.query.forward(layer_norm.forward(x)));
    attend(mha, q, kv, None)
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::{
            LayerNormConfig,
            attention::{
                MhaInput,
                MultiHeadAttentionConfig,
            },
        },
        tensor::{
            Distribution,
            Tolerance,
            backend::BackendTypes,
        },
    };

    use super::*;
    use crate::{
        blocks::transformers::attention::causal_mask,
        burner::tensor::TensorElemOpExt,
        support::testing::CpuBackend,
    };

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    /// **The contract.** Stepping a sequence through the cache one token at a
    /// time must reproduce a single uncached pass over the whole thing.
    ///
    /// This is what pins the arithmetic to `MultiHeadAttention::forward`: a
    /// wrong scale, a missed `min_float`, or an off-by-one in the causal mask
    /// all break it, and none would show up in a shape check.
    #[test]
    fn test_cached_self_attn_matches_uncached() {
        let device = Default::default();
        let (batch, d_model, n_heads, seq) = (2, 32, 4, 5);

        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init::<B>(&device);
        let ln = LayerNormConfig::new(d_model).init::<B>(&device);

        let x = Tensor::<B, 3>::random([batch, seq, d_model], Distribution::Default, &device);

        // Uncached: one pass, causally masked.
        let mask = causal_mask::<B>(seq, 0, &device);
        let whole = mha
            .forward(MhaInput::self_attn(ln.forward(x.clone())).mask_attn(mask))
            .context;

        // Cached: one token at a time. A lone query needs no mask.
        let mut cache: Option<AttnKv<B>> = None;
        let mut steps = Vec::with_capacity(seq);
        for t in 0..seq {
            let step = x.clone().slice_dim(1, t as isize..(t + 1) as isize);
            steps.push(layer_norm_self_attn_kv(&ln, &mha, step, None, &mut cache));
        }

        assert_eq!(cache.as_ref().unwrap().seq_len(), seq);

        let stepped: Tensor<B, 3> = Tensor::cat(steps, 1);
        stepped
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&whole.to_data_as::<F>(), Tolerance::permissive());
    }

    /// Feeding several tokens at once must agree with feeding them singly,
    /// which is what a prompt prefill does.
    #[test]
    fn test_cached_self_attn_prefill_matches_stepping() {
        let device = Default::default();
        let (batch, d_model, n_heads, seq) = (1, 32, 4, 6);

        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init::<B>(&device);
        let ln = LayerNormConfig::new(d_model).init::<B>(&device);
        let x = Tensor::<B, 3>::random([batch, seq, d_model], Distribution::Default, &device);

        // Prefill 4, then step the last 2.
        let split = 4;
        let mut cache: Option<AttnKv<B>> = None;
        let prefill = layer_norm_self_attn_kv(
            &ln,
            &mha,
            x.clone().slice_dim(1, 0..split as isize),
            Some(causal_mask::<B>(split, 0, &device)),
            &mut cache,
        );
        let mut parts = vec![prefill];
        for t in split..seq {
            let step = x.clone().slice_dim(1, t as isize..(t + 1) as isize);
            parts.push(layer_norm_self_attn_kv(&ln, &mha, step, None, &mut cache));
        }
        let mixed: Tensor<B, 3> = Tensor::cat(parts, 1);

        // All in one go.
        let mut cache2: Option<AttnKv<B>> = None;
        let at_once = layer_norm_self_attn_kv(
            &ln,
            &mha,
            x,
            Some(causal_mask::<B>(seq, 0, &device)),
            &mut cache2,
        );

        mixed
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&at_once.to_data_as::<F>(), Tolerance::permissive());
    }

    /// Cross-attention against a precomputed cache must equal the uncached
    /// path, and must accept a cross length unrelated to the query length.
    #[test]
    fn test_cached_cross_attn_matches_uncached() {
        let device = Default::default();
        let (batch, d_model, n_heads) = (2, 32, 4);
        let (seq, cross_len) = (3, 17);

        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init::<B>(&device);
        let ln = LayerNormConfig::new(d_model).init::<B>(&device);

        let x = Tensor::<B, 3>::random([batch, seq, d_model], Distribution::Default, &device);
        let xa =
            Tensor::<B, 3>::random([batch, cross_len, d_model], Distribution::Default, &device);

        let uncached = mha
            .forward(MhaInput::new(ln.forward(x.clone()), xa.clone(), xa.clone()))
            .context;

        let kv = project_kv(&mha, xa);
        assert_eq!(kv.seq_len(), cross_len);

        let cached = layer_norm_cross_attn_kv(&ln, &mha, x, &kv);

        cached
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&uncached.to_data_as::<F>(), Tolerance::permissive());
    }
}
