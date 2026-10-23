//! Multihead normalized attention utilities.

use burn::{
    Tensor,
    nn::{
        attention::{
            MhaInput,
            MhaOutput,
            MultiHeadAttention,
        },
        norm::Normalization,
    },
    prelude::Bool,
    tensor::activation::{
        quiet_softmax,
        softmax,
    },
};

use super::KVCache;

/// Asserts that `norm` normalizes `d_model` features, when it counts any
/// (an [`Identity`](Normalization::Identity) norm counts none).
#[cfg(any(debug_assertions, test))]
fn assert_norm_width(
    norm: &Normalization,
    d_model: usize,
) {
    let features = norm.num_features();
    assert!(
        features == 0 || features == d_model,
        "norm features ({features}) != d_model ({d_model})",
    );
}

/// Projects `[batch, seq, d_model]` into head-split keys and values:
/// `[batch, heads, seq, d_k]` each.
///
/// For cross-attention this is the whole win: call it once per layer against
/// the encoder output, set the result in the decode's [`KVCache`]
/// ([`set_cross_kv`](KVCache::set_cross_kv)), and it serves every step.
pub fn project_kv(
    mha: &MultiHeadAttention,
    x: Tensor<3>,
) -> (Tensor<4>, Tensor<4>) {
    (
        split_heads(mha, mha.key.forward(x.clone())),
        split_heads(mha, mha.value.forward(x)),
    )
}

/// `[batch, seq, d_model]` -> `[batch, heads, seq, d_k]`.
///
/// Mirrors `MultiHeadAttention::attention_linear`, minus the projection.
pub fn split_heads(
    mha: &MultiHeadAttention,
    x: Tensor<3>,
) -> Tensor<4> {
    let [batch, seq, _] = x.dims();
    x.reshape([batch, seq, mha.n_heads, mha.d_k])
        .swap_dims(1, 2)
}
/// Computes normalized self-attention: `attn(norm(x))`.
///
/// # Arguments
/// * `norm` - the sublayer's [`Normalization`].
/// * `mh_attn` - `MultiHeadAttention`.
/// * `x` - `[batch, seq_len, d_model]` input.
/// * `mask` - Optional `[batch, seq_len, seq_len]` attention mask.
///
/// # Returns
/// The [`MhaOutput`]:
/// * `context` : `[batch, seq_len, d_model]`.
/// * `weights` : `[batch, n_heads, seq_len, seq_len]`.
pub fn norm_self_attn(
    norm: &Normalization,
    mh_attn: &MultiHeadAttention,
    x: Tensor<3>,
    mask: Option<Tensor<3, Bool>>,
) -> MhaOutput {
    #[cfg(any(debug_assertions, test))]
    {
        use crate::contracts::*;
        let d_model = mh_attn.d_model;
        assert_norm_width(norm, d_model);

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

    let input = MhaInput::self_attn(norm.forward(x));
    let input = match mask {
        Some(mask) => input.mask_attn(mask),
        None => input,
    };
    mh_attn.forward(input)
}

/// Computes normalized cross-attention: `attn(norm(x), xa)`.
///
/// # Arguments
/// * `norm` - the sublayer's [`Normalization`].
/// * `mh_attn` - `MultiHeadAttention`.
/// * `x` - `[batch, seq_len, d_model]` input.
/// * `xa` - `[batch, cross_len, d_model]` cross-attention input.
///
/// `cross_len` is independent of `seq_len`: cross-attention exists to attend a
/// query sequence over a *different* one.
///
/// # Returns
/// The [`MhaOutput`]:
/// * `context` : `[batch, seq_len, d_model]`.
/// * `weights` : `[batch, n_heads, seq_len, cross_len]`.
pub fn norm_cross_attn(
    norm: &Normalization,
    mh_attn: &MultiHeadAttention,
    x: Tensor<3>,
    xa: Tensor<3>,
) -> MhaOutput {
    #[cfg(any(debug_assertions, test))]
    {
        crate::contracts::define_shape_contract!(CONTRACT, ["batch", "seq_len", "d_model"]);
        let d_model = mh_attn.d_model;
        assert_norm_width(norm, d_model);

        let [batch, _seq_len] =
            CONTRACT.unpack_shape(&x, &["batch", "seq_len"], &[("d_model", d_model)]);

        // `xa` shares the batch and the model width, but NOT the sequence
        // length — binding `seq_len` here would reject every real cross
        // attention.
        CONTRACT.assert_shape(&xa, &[("batch", batch), ("d_model", d_model)]);
    };

    mh_attn.forward(MhaInput::new(norm.forward(x.clone()), xa.clone(), xa))
}

/// Scaled dot-product attention over head-split tensors, then the output
/// projection.
///
/// # Arguments
/// * `q`: `[batch, heads, seq_new, d_k]`.
/// * `k`, `v`: `[batch, heads, seq_total, d_k]`.
/// * `mask`: optional `[batch, seq_new, seq_total]`.
///
/// # Returns
/// `[batch, seq_new, d_model]`.
pub fn attend_q_kv_mask(
    mha: &MultiHeadAttention,
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    mask: Option<Tensor<3, Bool>>,
) -> Tensor<3> {
    let [batch, _, seq_new, _] = q.dims();

    let scores = q.matmul(k.transpose()).div_scalar((mha.d_k as f32).sqrt());

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

    let context = weights
        .matmul(v)
        .swap_dims(1, 2)
        .reshape([batch, seq_new, mha.d_model]);

    mha.output.forward(context)
}

/// Normalized self-attention over a decode's [`KVCache`].
///
/// Writes this step's keys and values into `cache` at `layer`, and attends
/// over the layer's whole history. Unlike the uncached path, `x` is only the
/// **new** tokens.
///
/// # Arguments
/// * `x`: `[batch, seq_new, d_model]`, the new token(s) only.
/// * `mask`: optional `[batch, seq_new, seq_past + seq_new]`. Build it with
///   [`causal_mask(seq_new, seq_past, device)`](`super::causal_mask`). A single
///   new token needs no mask: it may attend to everything already cached.
/// * `cache`, `layer`: the decode's cache, and this layer's index in it.
///
/// # Returns
/// `[batch, seq_new, d_model]`.
pub fn norm_self_attn_w_kv_cache(
    norm: &Normalization,
    mha: &MultiHeadAttention,
    x: Tensor<3>,
    mask: Option<Tensor<3, Bool>>,
    cache: &mut KVCache,
    layer: usize,
) -> Tensor<3> {
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract!(
        ["batch", "seq_new", "d_model"],
        &x,
        &[("d_model", mha.d_model)],
    );

    let normed = norm.forward(x);

    let q = split_heads(mha, mha.query.forward(normed.clone()));
    let (k, v) = project_kv(mha, normed);
    let (k, v) = cache.insert_kv(layer, k, v);

    attend_q_kv_mask(mha, q, k, v, mask)
}

/// Normalized cross-attention against a decode's [`KVCache`]: the
/// cross-attention keys and values set at `layer`
/// ([`set_cross_kv`](KVCache::set_cross_kv)).
///
/// The keys and values come from the encoder output and never change, so
/// [`project_kv`] runs once per layer per decode rather than once per token.
/// No mask: every query may attend to the whole encoded sequence.
///
/// When the cache's [`group`](KVCache::group) is above one, `x` has `group`
/// rows per encoder row, laid out `row = encoder_row * group + member`, as a
/// beam search lays out its beams per audio. Cross-attention has no mask and
/// every query row is independent, so the group's rows are folded into the
/// query's sequence axis, attend against the one encoder row, and are unfolded
/// after: two reshapes on a few kilobytes of queries, and the keys and values
/// are never repeated.
///
/// # Arguments
/// * `x`: `[batch, seq_new, d_model]`, the new token(s) only.
/// * `cache`, `layer`: the decode's cache, and this layer's index in it.
///
/// # Returns
/// `[batch, seq_new, d_model]`.
///
/// # Panics
/// When `layer`'s cross-attention keys and values aren't set, or `x`'s rows
/// aren't `group` per encoder row.
pub fn norm_cross_attn_w_kv_cache(
    norm: &Normalization,
    mha: &MultiHeadAttention,
    x: Tensor<3>,
    cache: &KVCache,
    layer: usize,
) -> Tensor<3> {
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract!(
        ["batch", "seq_new", "d_model"],
        &x,
        &[("d_model", mha.d_model)],
    );

    let (k, v) = cache
        .cross_kv(layer)
        .expect("the layer's cross-attention keys and values are set");
    let group = cache.group();
    let [rows, seq_new, d_model] = x.dims();
    let encoded = k.dims()[0];
    assert_eq!(
        rows,
        encoded * group,
        "query rows ({rows}) are not {group} per encoder row ({encoded})",
    );

    let q = split_heads(mha, mha.query.forward(norm.forward(x)));
    if group == 1 {
        return attend_q_kv_mask(mha, q, k, v, None);
    }

    // [rows, H, T, d_k] -> [encoded, group, H, T, d_k] -> [encoded, H, group,
    // T, d_k] -> [encoded, H, group * T, d_k]: the group folded into the
    // sequence.
    let [_, heads, _, d_k] = q.dims();
    let q = q
        .reshape([encoded, group, heads, seq_new, d_k])
        .swap_dims(1, 2)
        .reshape([encoded, heads, group * seq_new, d_k]);

    // [encoded, group * T, d_model] -> [rows, T, d_model]: the fold undone,
    // rows back in `encoder_row * group + member` order.
    attend_q_kv_mask(mha, q, k, v, None).reshape([rows, seq_new, d_model])
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
        },
    };

    use super::{
        super::causal_mask,
        *,
    };
    use crate::support::testing::cpu_device;

    type F = f32;

    /// Cross-attention must accept a `xa` whose sequence length differs from
    /// the query's — that is the entire point of it.
    ///
    /// A contract that bound both to one `seq_len` would pass every
    /// same-length unit test and still be wrong, which is why this one uses
    /// deliberately mismatched lengths.
    #[test]
    fn test_cross_attn_accepts_a_different_cross_length() {
        let device = cpu_device();
        let (batch, d_model, n_heads) = (2, 32, 4);
        let (seq_len, cross_len) = (3, 17);

        let attn = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);
        let ln = Normalization::Layer(LayerNormConfig::new(d_model).init(&device));

        let x = Tensor::<3>::random([batch, seq_len, d_model], Distribution::Default, &device);
        let xa = Tensor::<3>::random([batch, cross_len, d_model], Distribution::Default, &device);

        let out = norm_cross_attn(&ln, &attn, x, xa);

        // The output follows the query, and the weights span query x cross.
        assert_eq!(out.context.dims(), [batch, seq_len, d_model]);
        assert_eq!(out.weights.dims(), [batch, n_heads, seq_len, cross_len]);
    }

    /// The equal-length case still works, so the loosened contract did not
    /// simply stop checking.
    #[test]
    fn test_cross_attn_accepts_equal_lengths() {
        let device = cpu_device();
        let (batch, d_model, n_heads, seq_len) = (1, 16, 2, 5);

        let attn = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);
        let ln = Normalization::Layer(LayerNormConfig::new(d_model).init(&device));

        let x = Tensor::<3>::random([batch, seq_len, d_model], Distribution::Default, &device);

        let out = norm_cross_attn(&ln, &attn, x.clone(), x);
        assert_eq!(out.context.dims(), [batch, seq_len, d_model]);
    }

    /// A `d_model` mismatch is still rejected.
    #[test]
    #[should_panic(expected = "Shape Error")]
    fn test_cross_attn_rejects_a_d_model_mismatch() {
        let device = cpu_device();
        let (batch, d_model, n_heads) = (1, 16, 2);

        let attn = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);
        let ln = Normalization::Layer(LayerNormConfig::new(d_model).init(&device));

        let x = Tensor::<3>::random([batch, 4, d_model], Distribution::Default, &device);
        let xa = Tensor::<3>::random([batch, 9, d_model * 2], Distribution::Default, &device);

        norm_cross_attn(&ln, &attn, x, xa);
    }

    /// A single-layer [`KVCache`] for `mha`, `batch` rows.
    fn cache_for(
        mha: &MultiHeadAttention,
        batch: usize,
        seq: usize,
    ) -> KVCache {
        super::super::KVCacheConfig::new(batch, mha.n_heads, seq, mha.d_k, 1).init()
    }

    /// **The contract.** Stepping a sequence through the cache one token at a
    /// time must reproduce a single uncached pass over the whole thing.
    ///
    /// This is what pins the arithmetic to `MultiHeadAttention::forward`: a
    /// wrong scale, a missed `min_float`, or an off-by-one in the causal mask
    /// all break it, and none would show up in a shape check.
    #[test]
    fn test_cached_self_attn_matches_uncached() {
        let device = cpu_device();
        let (batch, d_model, n_heads, seq) = (2, 32, 4, 5);

        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);
        let ln = Normalization::Layer(LayerNormConfig::new(d_model).init(&device));

        let x = Tensor::<3>::random([batch, seq, d_model], Distribution::Default, &device);

        // Uncached: one pass, causally masked.
        let mask = causal_mask(seq, 0, &device);
        let whole = mha
            .forward(MhaInput::self_attn(ln.forward(x.clone())).mask_attn(mask))
            .context;

        // Cached: one token at a time. A lone query needs no mask.
        let mut cache = cache_for(&mha, batch, seq);
        let mut steps = Vec::with_capacity(seq);
        for t in 0..seq {
            let step = x.clone().slice_dim(1, t as isize..(t + 1) as isize);
            steps.push(norm_self_attn_w_kv_cache(
                &ln, &mha, step, None, &mut cache, 0,
            ));
        }

        assert_eq!(cache.pos(), seq);

        let stepped: Tensor<3> = Tensor::cat(steps, 1);
        stepped
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&whole.to_data_as::<F>(), Tolerance::permissive());
    }

    /// Feeding several tokens at once must agree with feeding them singly,
    /// which is what a prompt prefill does.
    #[test]
    fn test_cached_self_attn_prefill_matches_stepping() {
        let device = cpu_device();
        let (batch, d_model, n_heads, seq) = (1, 32, 4, 6);

        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);
        let ln = Normalization::Layer(LayerNormConfig::new(d_model).init(&device));
        let x = Tensor::<3>::random([batch, seq, d_model], Distribution::Default, &device);

        // Prefill 4, then step the last 2.
        let split = 4;
        let mut cache = cache_for(&mha, batch, seq);
        let prefill = norm_self_attn_w_kv_cache(
            &ln,
            &mha,
            x.clone().slice_dim(1, 0..split as isize),
            Some(causal_mask(split, 0, &device)),
            &mut cache,
            0,
        );
        let mut parts = vec![prefill];
        for t in split..seq {
            let step = x.clone().slice_dim(1, t as isize..(t + 1) as isize);
            parts.push(norm_self_attn_w_kv_cache(
                &ln, &mha, step, None, &mut cache, 0,
            ));
        }
        let mixed: Tensor<3> = Tensor::cat(parts, 1);

        // All in one go.
        let mut cache2 = cache_for(&mha, batch, seq);
        let at_once = norm_self_attn_w_kv_cache(
            &ln,
            &mha,
            x,
            Some(causal_mask(seq, 0, &device)),
            &mut cache2,
            0,
        );

        mixed
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&at_once.to_data_as::<F>(), Tolerance::permissive());
    }

    /// Cross-attention against the cache's keys and values must equal the
    /// uncached path, and must accept a cross length unrelated to the query
    /// length.
    #[test]
    fn test_cached_cross_attn_matches_uncached() {
        let device = cpu_device();
        let (batch, d_model, n_heads) = (2, 32, 4);
        let (seq, cross_len) = (3, 17);

        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);
        let ln = Normalization::Layer(LayerNormConfig::new(d_model).init(&device));

        let x = Tensor::<3>::random([batch, seq, d_model], Distribution::Default, &device);
        let xa = Tensor::<3>::random([batch, cross_len, d_model], Distribution::Default, &device);

        let uncached = mha
            .forward(MhaInput::new(ln.forward(x.clone()), xa.clone(), xa.clone()))
            .context;

        let (k, v) = project_kv(&mha, xa);
        assert_eq!(k.dims()[2], cross_len);
        let mut cache = cache_for(&mha, batch, seq);
        cache.set_cross_kv(0, k, v);

        let cached = norm_cross_attn_w_kv_cache(&ln, &mha, x, &cache, 0);

        cached
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&uncached.to_data_as::<F>(), Tolerance::permissive());
    }

    /// Grouped cross-attention (`group` query rows per encoder row) equals
    /// cross-attention against the encoder rows repeated per member.
    #[test]
    fn test_grouped_cross_attn_matches_repeated() {
        let device = cpu_device();
        let (audio, group, d_model, n_heads, seq, cross_len) = (2, 3, 32, 4, 2, 7);
        let rows = audio * group;

        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);
        let ln = Normalization::Layer(LayerNormConfig::new(d_model).init(&device));
        let x = Tensor::<3>::random([rows, seq, d_model], Distribution::Default, &device);
        let xa = Tensor::<3>::random([audio, cross_len, d_model], Distribution::Default, &device);

        let mut shared = cache_for(&mha, rows, seq);
        let (k, v) = project_kv(&mha, xa.clone());
        shared.set_cross_kv(0, k, v);
        assert_eq!(shared.group(), group);

        let mut repeated = cache_for(&mha, rows, seq);
        let indices = Tensor::<1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(
                (0..rows).map(|r| (r / group) as i64).collect::<Vec<_>>(),
                [rows],
            ),
            &device,
        );
        let (k, v) = project_kv(&mha, xa.select(0, indices));
        repeated.set_cross_kv(0, k, v);
        assert_eq!(repeated.group(), 1);

        let a = norm_cross_attn_w_kv_cache(&ln, &mha, x.clone(), &shared, 0);
        let b = norm_cross_attn_w_kv_cache(&ln, &mha, x, &repeated, 0);
        a.to_data_as::<F>()
            .assert_approx_eq::<F>(&b.to_data_as::<F>(), Tolerance::permissive());
    }
}
