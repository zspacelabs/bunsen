//! Multihead Layer Normalized Attention Utilities.

use burn::{
    Tensor,
    nn::{
        LayerNorm,
        attention::{
            MhaInput,
            MhaOutput,
            MultiHeadAttention,
        },
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

use super::AttnKvPair;

/// Projects `[batch, seq, d_model]` into head-split keys and values.
///
/// For cross-attention this is the whole win: call it once per layer against
/// the encoder output and the result serves every decode step.
pub fn project_kv_pair<B: Backend>(
    mha: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
) -> AttnKvPair<B> {
    AttnKvPair {
        key: split_heads(mha, mha.key.forward(x.clone())),
        value: split_heads(mha, mha.value.forward(x)),
    }
}

/// `[batch, seq, d_model]` -> `[batch, heads, seq, d_k]`.
///
/// Mirrors `MultiHeadAttention::attention_linear`, minus the projection.
pub fn split_heads<B: Backend>(
    mha: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, seq, _] = x.dims();
    x.reshape([batch, seq, mha.n_heads, mha.d_k])
        .swap_dims(1, 2)
}
/// Computes layer normalized self-attn.
///
/// # Arguments
/// * `layer_norm` - `LayerNorm`.
/// * `mh_attn` - `MultiHeadAttention`.
/// * `x` - `[batch, seq_len, d_model]` input.
/// * `mask` - Optional `[batch, seq_len, seq_len]` attention mask.
///
/// # Returns
/// `RdabForwardRecord` - forward record.
/// * `fr.output` : `[batch, seq_len, d_model]`.
/// * `fr.ca_weights` : `[batch, n_heads, seq_len, seq_len]`.
pub fn layer_norm_self_attn<B: Backend>(
    layer_norm: &LayerNorm<B>,
    mh_attn: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
    mask: Option<Tensor<B, 3, Bool>>,
) -> MhaOutput<B> {
    #[cfg(any(debug_assertions, test))]
    {
        use crate::contracts::*;
        let d_model = mh_attn.d_model;
        assert_eq!(
            d_model,
            layer_norm.gamma.dims()[0],
            "layer_norm dims ({}) != d_model ({d_model})",
            layer_norm.gamma.dims()[0],
        );

        let [batch, seq_len] = unpack_shape_contract!(
            ["batch", "seq_len", "d_model"],
            &x,
            &["batch", "seq_len"],
            &[("d_model", d_model)]
        );

        if let Some(mask) = &mask {
            let [mask_batch] = unpack_shape_contract!(
                ["mask_batch", "seq_len", "seq_len"],
                mask,
                &["mask_batch"],
                &[("seq_len", seq_len)]
            );
            if mask_batch != 1 {
                assert_eq!(
                    mask_batch, batch,
                    "batch sizes not broadcastable {batch} vs {mask_batch}"
                );
            }
        }
    }

    let input = MhaInput::self_attn(layer_norm.forward(x));
    let input = match mask {
        Some(mask) => input.mask_attn(mask),
        None => input,
    };
    mh_attn.forward(input)
}

/// Computes layer normalized cross-attn.
///
/// # Arguments
/// * `layer_norm` - `LayerNorm`.
/// * `mh_attn` - `MultiHeadAttention`.
/// * `x` - `[batch, seq_len, d_model]` input.
/// * `xa` - `[batch, cross_len, d_model]` cross-attention input.
///
/// `cross_len` is independent of `seq_len`: cross-attention exists to attend a
/// query sequence over a *different* one.
///
/// # Returns
/// `RdabForwardRecord` - forward record.
/// * `fr.output` : `[batch, seq_len, d_model]`.
/// * `fr.ca_weights` : `[batch, n_heads, seq_len, cross_len]`.
pub fn layer_norm_cross_attn<B: Backend>(
    layer_norm: &LayerNorm<B>,
    mh_attn: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
    xa: Tensor<B, 3>,
) -> MhaOutput<B> {
    #[cfg(any(debug_assertions, test))]
    {
        crate::contracts::define_shape_contract!(CONTRACT, ["batch", "seq_len", "d_model"]);
        let d_model = mh_attn.d_model;
        assert_eq!(
            d_model,
            layer_norm.gamma.dims()[0],
            "layer_norm dims ({}) != d_model ({d_model})",
            layer_norm.gamma.dims()[0],
        );

        let [batch, _seq_len] =
            CONTRACT.unpack_shape(&x, &["batch", "seq_len"], &[("d_model", d_model)]);

        // `xa` shares the batch and the model width, but NOT the sequence
        // length — binding `seq_len` here would reject every real cross
        // attention.
        CONTRACT.assert_shape(&xa, &[("batch", batch), ("d_model", d_model)]);
    };

    mh_attn.forward(MhaInput::new(layer_norm.forward(x.clone()), xa.clone(), xa))
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
pub fn attend_q_kv_mask<B: Backend>(
    mha: &MultiHeadAttention<B>,
    q: Tensor<B, 4>,
    kv: &AttnKvPair<B>,
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
///   [`causal_mask(seq_new, seq_past, device)`](`super::causal_mask`). A single
///   new token needs no mask: it may attend to everything already cached.
/// * `cache`: `None` at the start of a stream; grown in place thereafter.
///
/// # Returns
/// `[batch, seq_new, d_model]`.
pub fn layer_norm_self_attn_w_kv_cache<B: Backend>(
    layer_norm: &LayerNorm<B>,
    mha: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
    mask: Option<Tensor<B, 3, Bool>>,
    cache: &mut Option<AttnKvPair<B>>,
) -> Tensor<B, 3> {
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract!(
        ["batch", "seq_new", "d_model"],
        &x,
        &[("d_model", mha.d_model)],
    );

    let normed = layer_norm.forward(x);

    let q = split_heads(mha, mha.query.forward(normed.clone()));
    let fresh = project_kv_pair(mha, normed);

    let grown = match cache.take() {
        Some(past) => past.concat(fresh),
        None => fresh,
    };

    let out = attend_q_kv_mask(mha, q, &grown, mask);
    *cache = Some(grown);

    out
}

/// Layer-normed cross-attention against a fixed key/value cache.
///
/// The keys and values come from the encoder output and never change, so
/// [`project_kv_pair`] runs once per layer per decode rather
/// than once per token. No mask: every query may attend to the whole encoded
/// sequence.
///
/// # Arguments
/// * `x`: `[batch, seq_new, d_model]` — the new token(s) only.
/// * `kv`: from [`project_kv_pair`] over the encoder output.
///
/// # Returns
/// `[batch, seq_new, d_model]`.
pub fn layer_norm_cross_attn_w_kv_cache<B: Backend>(
    layer_norm: &LayerNorm<B>,
    mha: &MultiHeadAttention<B>,
    x: Tensor<B, 3>,
    kv: &AttnKvPair<B>,
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
    attend_q_kv_mask(mha, q, kv, None)
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

    use super::{
        super::causal_mask,
        *,
    };
    use crate::{
        burner::tensor::TensorElemOpExt,
        support::testing::CpuBackend,
    };

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    /// Cross-attention must accept a `xa` whose sequence length differs from
    /// the query's — that is the entire point of it.
    ///
    /// A contract that bound both to one `seq_len` would pass every
    /// same-length unit test and still be wrong, which is why this one uses
    /// deliberately mismatched lengths.
    #[test]
    fn test_cross_attn_accepts_a_different_cross_length() {
        let device = Default::default();
        let (batch, d_model, n_heads) = (2, 32, 4);
        let (seq_len, cross_len) = (3, 17);

        let attn = MultiHeadAttentionConfig::new(d_model, n_heads).init::<B>(&device);
        let ln = LayerNormConfig::new(d_model).init::<B>(&device);

        let x = Tensor::<B, 3>::random([batch, seq_len, d_model], Distribution::Default, &device);
        let xa =
            Tensor::<B, 3>::random([batch, cross_len, d_model], Distribution::Default, &device);

        let out = layer_norm_cross_attn(&ln, &attn, x, xa);

        // The output follows the query, and the weights span query x cross.
        assert_eq!(out.context.dims(), [batch, seq_len, d_model]);
        assert_eq!(out.weights.dims(), [batch, n_heads, seq_len, cross_len]);
    }

    /// The equal-length case still works, so the loosened contract did not
    /// simply stop checking.
    #[test]
    fn test_cross_attn_accepts_equal_lengths() {
        let device = Default::default();
        let (batch, d_model, n_heads, seq_len) = (1, 16, 2, 5);

        let attn = MultiHeadAttentionConfig::new(d_model, n_heads).init::<B>(&device);
        let ln = LayerNormConfig::new(d_model).init::<B>(&device);

        let x = Tensor::<B, 3>::random([batch, seq_len, d_model], Distribution::Default, &device);

        let out = layer_norm_cross_attn(&ln, &attn, x.clone(), x);
        assert_eq!(out.context.dims(), [batch, seq_len, d_model]);
    }

    /// A `d_model` mismatch is still rejected.
    #[test]
    #[should_panic(expected = "Shape Error")]
    fn test_cross_attn_rejects_a_d_model_mismatch() {
        let device = Default::default();
        let (batch, d_model, n_heads) = (1, 16, 2);

        let attn = MultiHeadAttentionConfig::new(d_model, n_heads).init::<B>(&device);
        let ln = LayerNormConfig::new(d_model).init::<B>(&device);

        let x = Tensor::<B, 3>::random([batch, 4, d_model], Distribution::Default, &device);
        let xa = Tensor::<B, 3>::random([batch, 9, d_model * 2], Distribution::Default, &device);

        layer_norm_cross_attn(&ln, &attn, x, xa);
    }

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
        let mut cache: Option<AttnKvPair<B>> = None;
        let mut steps = Vec::with_capacity(seq);
        for t in 0..seq {
            let step = x.clone().slice_dim(1, t as isize..(t + 1) as isize);
            steps.push(layer_norm_self_attn_w_kv_cache(
                &ln, &mha, step, None, &mut cache,
            ));
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
        let mut cache: Option<AttnKvPair<B>> = None;
        let prefill = layer_norm_self_attn_w_kv_cache(
            &ln,
            &mha,
            x.clone().slice_dim(1, 0..split as isize),
            Some(causal_mask::<B>(split, 0, &device)),
            &mut cache,
        );
        let mut parts = vec![prefill];
        for t in split..seq {
            let step = x.clone().slice_dim(1, t as isize..(t + 1) as isize);
            parts.push(layer_norm_self_attn_w_kv_cache(
                &ln, &mha, step, None, &mut cache,
            ));
        }
        let mixed: Tensor<B, 3> = Tensor::cat(parts, 1);

        // All in one go.
        let mut cache2: Option<AttnKvPair<B>> = None;
        let at_once = layer_norm_self_attn_w_kv_cache(
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

        let kv = project_kv_pair(&mha, xa);
        assert_eq!(kv.seq_len(), cross_len);

        let cached = layer_norm_cross_attn_w_kv_cache(&ln, &mha, x, &kv);

        cached
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&uncached.to_data_as::<F>(), Tolerance::permissive());
    }
}
