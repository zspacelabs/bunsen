//! Scaled dot-product attention, as functions.

use burn::{
    Tensor,
    config::Config,
    prelude::{
        Bool,
        Int,
    },
    tensor::{
        DType,
        Device,
        FloatDType,
        activation::softmax,
    },
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
    ops::{
        drop::dropout,
        repeat,
    },
};

/// Config options for [`scaled_dot_product_attention`].
///
/// A value object: [`CausalSelfAttention`] builds one per call, setting
/// `is_causal` from its KV-cache mode and `enable_gqa` from its head counts.
///
/// [`CausalSelfAttention`]: crate::blocks::transformers::attention::csa::CausalSelfAttention
#[derive(Config, Debug, Copy)]
pub struct ScaledDotProductAttentionConfig {
    /// Causal or not.
    #[config(default = "false")]
    pub is_causal: bool,

    /// Enable Group Query Attention: repeat each key/value head
    /// `H_q / H_kv` times to match the query heads.
    #[config(default = "false")]
    pub enable_gqa: bool,

    /// Manual Scale factor; `None` means `1 / sqrt(D)`.
    #[config(default = "None")]
    pub scale: Option<f64>,

    /// Dropout rate.
    #[config(default = "None")]
    pub dropout: Option<f64>,

    /// Enable dropout during inference.
    #[config(default = "true")]
    pub enable_dropout_during_inference: bool,

    /// The dtype the softmax runs in: the scores are cast to it, their bias
    /// built in it, and the weights cast back to the query's dtype after.
    /// `None` runs it in the query's dtype. `transformers` runs it in f32
    /// for a half model (`softmax(..., dtype=torch.float32)`).
    #[config(default = "None")]
    pub softmax_dtype: Option<FloatDType>,

    /// A soft-cap on the scaled scores, `c · tanh(s / c)`, before the bias
    /// and mask (Gemma 2's 50); `None`, none.
    #[config(default = "None")]
    pub softcap: Option<f64>,
}

/// Computes scaled dot product attention.
///
/// See:
/// - [pytorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
///
/// # Arguments
/// - `q`: the query tensor, as `[B, H_q, T_q, D]`.
/// - `k`: the key tensor, as `[B, H_k, T_kv, D]`.
/// - `v`: the value tensor, as `[B, H_v, T_kv, D]`.
/// - `bias`: optional additive bias, as `[T_q, T_kv]`.
/// - `mask`: optional boolean mask, as `[T_q, T_kv]`; `true` means attend.
/// - `key_valid`: optional per-item key mask, as `[B, T_kv]`; `true` at the
///   keys that are not padding, which alone are attended to.
/// - `config`: attention config.
///
/// # Returns
/// - the `[B, H_q, T_q, D]` attention result.
pub fn scaled_dot_product_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    bias: Option<Tensor<2>>,
    mask: Option<Tensor<2, Bool>>,
    key_valid: Option<Tensor<2, Bool>>,
    config: ScaledDotProductAttentionConfig,
) -> Tensor<4> {
    let [b, h_q, _t_q, d] = unpack_shape_contract!(["B", "H_q", "T_q", "D"], &q.dims());
    let [h_kv] = unpack_shape_contract!(
        ["B", "H_kv", "T_k", "D"],
        &k.dims(),
        &["H_kv"],
        &[("B", b), ("D", d)]
    );
    assert_shape_contract_periodically!(
        ["B", "H_kv", "T_v", "D"],
        &v.dims(),
        &[("B", b), ("H_kv", h_kv), ("D", d)]
    );

    let attn_weight = sdpa_attn_weight(q, k, bias, mask, key_valid, config);

    let mut v = v;
    if config.enable_gqa {
        let v_repeats = h_q / h_kv;
        v = repeat::repeat_interleave::<4, 5, _>(v, v_repeats, 1);
    }

    attn_weight.matmul(v)
}

/// Builds the Attention Weight for [`scaled_dot_product_attention`].
///
/// The weight is `softmax(cap(q k^T * scale) + bias)` over the keys, `cap`
/// the config's soft-cap if it has one (`c · tanh(s / c)`), then
/// dropout on those probabilities when the config sets a rate, as in
/// `PyTorch`'s reference. The scores are the query's dtype; the softmax runs
/// in the config's `softmax_dtype` when it sets one, and the weight is the
/// query's dtype either way.
///
/// # Arguments
/// - `q`: the query tensor, as `[B, H_q, T_q, D]`.
/// - `k`: the key tensor, as `[B, H_k, T_k, D]`.
/// - `bias`: optional additive bias, as `[T_q, T_k]`.
/// - `mask`: optional bias mask, as `[T_q, T_k]`.
/// - `key_valid`: optional per-item key mask, as `[B, T_k]`; `true` at the keys
///   that are not padding. A query it and the other masks leave no key (a
///   padding query whose window holds only padding) has a row of zeros.
/// - `config`: attention config.
pub fn sdpa_attn_weight(
    q: Tensor<4>,
    k: Tensor<4>,
    bias: Option<Tensor<2>>,
    mask: Option<Tensor<2, Bool>>,
    key_valid: Option<Tensor<2, Bool>>,
    config: ScaledDotProductAttentionConfig,
) -> Tensor<4> {
    let [b, h_q, t_q, d] = unpack_shape_contract!(["B", "H_q", "T_q", "D"], &q.dims());
    let [h_k, t_k] = unpack_shape_contract!(
        ["B", "H_k", "T_k", "D"],
        &k.dims(),
        &["H_k", "T_k"],
        &[("B", b), ("D", d)]
    );

    let device = q.device();
    let dtype = q.dtype();

    let mut k = k;

    if config.enable_gqa {
        let k_repeats = h_q / h_k;
        k = repeat::repeat_interleave::<4, 5, _>(k, k_repeats, 1);
    }

    let scale_factor = config.scale.unwrap_or(1.0 / (q.dims()[3] as f64).sqrt());
    let scores = q.matmul(k.swap_dims(2, 3)) * scale_factor;
    // Capped in the scores' dtype, before the softmax's cast, as
    // `transformers`' eager attention caps them.
    let scores = match config.softcap {
        Some(cap) => scores.div_scalar(cap).tanh().mul_scalar(cap),
        None => scores,
    };

    let softmax_dtype = config.softmax_dtype.map_or(dtype, DType::from);
    let scores = scores.cast(softmax_dtype);
    let attn_bias = sdpa_bias(
        t_q,
        t_k,
        config.is_causal,
        bias,
        mask,
        softmax_dtype,
        &device,
    );
    let mut scores = scores + attn_bias.unsqueeze();
    let mut blind = None;
    if let Some(valid) = key_valid {
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "T_k"],
            &valid.dims(),
            &[("B", b), ("T_k", t_k)]
        );
        scores = scores.mask_fill(valid.bool_not().reshape([b, 1, 1, t_k]), f32::NEG_INFINITY);
        // A query that sees only padding (a padding query whose window holds
        // no valid key) attends to nothing: its row is zero, not the NaN of
        // a softmax over no keys, which the next layer's zero weights would
        // carry into every query.
        blind = Some(
            scores
                .clone()
                .max_dim(3)
                .equal_elem(f32::NEG_INFINITY)
                .expand(scores.shape()),
        );
    }
    let mut attn_weight = softmax(scores, 3);
    if let Some(blind) = blind {
        attn_weight = attn_weight.mask_fill(blind, 0.0);
    }
    let mut attn_weight = attn_weight.cast(dtype);

    if let Some(prob) = config.dropout
        && (config.enable_dropout_during_inference || attn_weight.device().is_autodiff())
    {
        attn_weight = dropout(prob, attn_weight);
    }

    attn_weight
}

/// Builds the Attention Bias for [`scaled_dot_product_attention`].
///
/// # Arguments
/// - `l`: the query time dimension.
/// - `s`: the key time dimension.
/// - `causal`: whether the attention is causal.
/// - `bias`: optional additive bias, in any float dtype: cast to `dtype`.
/// - `mask`: optional bias mask.
/// - `dtype`: the desired dtype of the output tensor.
/// - `device`: the target device of the bias.
///
/// # Returns
/// - a `[l, s]` attention bias tensor.
pub fn sdpa_bias(
    l: usize,
    s: usize,
    causal: bool,
    bias: Option<Tensor<2>>,
    mask: Option<Tensor<2, Bool>>,
    dtype: DType,
    device: &Device,
) -> Tensor<2> {
    let mut attn_bias = Tensor::<2>::zeros([l, s], device).cast(dtype);
    if causal {
        attn_bias = attn_bias.mask_fill(
            Tensor::<2, Int>::ones([l, s], device)
                .tril(0)
                .bool()
                .bool_not(),
            f32::NEG_INFINITY,
        );
    }
    if let Some(bias) = bias {
        attn_bias = attn_bias + bias.cast(dtype);
    }
    if let Some(mask) = mask {
        attn_bias = attn_bias.mask_fill(mask.bool_not(), f32::NEG_INFINITY);
    }
    attn_bias
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        cpu_device,
        performance_device,
        seeded_tensor,
    };

    /// The soft-cap bounds the scaled scores before the causal mask:
    /// `softmax(c · tanh(q kᵀ · scale / c) + bias)`.
    #[test]
    fn test_sdpa_attn_weight_softcap() {
        let device = cpu_device();
        let shape = [1, 2, 4, 8];
        let q = seeded_tensor::<4>(1, shape, Distribution::Default, &device);
        let k = seeded_tensor::<4>(2, shape, Distribution::Default, &device);
        let (scale, cap) = (0.5, 0.4);
        let config = ScaledDotProductAttentionConfig::new()
            .with_is_causal(true)
            .with_scale(Some(scale))
            .with_softcap(Some(cap));
        let scores = q
            .clone()
            .matmul(k.clone().swap_dims(2, 3))
            .mul_scalar(scale);
        let capped = scores.div_scalar(cap).tanh().mul_scalar(cap);
        let bias = sdpa_bias(4, 4, true, None, None, DType::F32, &device);
        let expected = softmax(capped + bias.unsqueeze(), 3);
        crate::support::testing::assert_tensors_close(
            &sdpa_attn_weight(q, k, None, None, None, config),
            &expected,
            burn::tensor::Tolerance::default(),
        );
    }

    /// A query that the window and the padding leave no key attends to
    /// nothing: its row is zero, not NaN, and every other row is a
    /// distribution.
    #[test]
    fn test_sdpa_query_without_keys() {
        let device = cpu_device();
        let shape = [1, 1, 4, 4];
        let q = seeded_tensor::<4>(3, shape, Distribution::Default, &device);
        let k = seeded_tensor::<4>(4, shape, Distribution::Default, &device);
        let v = seeded_tensor::<4>(5, shape, Distribution::Default, &device);
        // Keys 2 and 3 are padding; query 3's window, |i - j| < 2, is keys 2
        // and 3 alone.
        let valid = crate::ops::transformers::attention::length_mask(&[2], 4, &device);
        let window = crate::ops::transformers::attention::two_sided_window_visible(4, 2, &device);
        let config = ScaledDotProductAttentionConfig::new();
        let w = sdpa_attn_weight(
            q.clone(),
            k.clone(),
            None,
            Some(window.clone()),
            Some(valid.clone()),
            config.clone(),
        );
        assert!(!w.clone().is_nan().any().into_scalar::<bool>());
        let sums: Vec<f32> = w.clone().sum_dim(3).into_data().iter::<f32>().collect();
        for (row, sum) in sums.iter().enumerate() {
            let want = if row == 3 { 0.0 } else { 1.0 };
            assert!((sum - want).abs() < 1e-5, "row {row}: {sum}");
        }
        let y = scaled_dot_product_attention(q, k, v, None, Some(window), Some(valid), config);
        assert_eq!(
            y.slice([0..1, 0..1, 3..4, 0..4])
                .abs()
                .max()
                .into_scalar::<f32>(),
            0.0
        );
    }

    /// A padding key is attended to by no query of its item: its weight is
    /// zero, and the rest of the row is the softmax over the valid keys.
    #[test]
    fn test_sdpa_key_valid() {
        let device = cpu_device();
        let shape = [2, 1, 3, 4];
        let q = seeded_tensor::<4>(1, shape, Distribution::Default, &device);
        let k = seeded_tensor::<4>(2, shape, Distribution::Default, &device);
        let valid = crate::ops::transformers::attention::length_mask(&[3, 2], 3, &device);
        let w = sdpa_attn_weight(
            q.clone(),
            k.clone(),
            None,
            None,
            Some(valid),
            ScaledDotProductAttentionConfig::new(),
        );
        // Item 1's last key is padding.
        let padded = w.clone().slice([1..2, 0..1, 0..3, 2..3]);
        assert_eq!(padded.abs().max().into_scalar::<f32>(), 0.0);
        // Item 1's rows are the softmax over its first two keys alone.
        let alone = sdpa_attn_weight(
            q.slice([1..2, 0..1, 0..3, 0..4]),
            k.slice([1..2, 0..1, 0..2, 0..4]),
            None,
            None,
            None,
            ScaledDotProductAttentionConfig::new(),
        );
        crate::support::testing::assert_tensors_close(
            &w.slice([1..2, 0..1, 0..3, 0..2]),
            &alone,
            burn::tensor::Tolerance::default(),
        );
    }

    /// Dropout acts on the attention probabilities, after the softmax, as in
    /// PyTorch's reference: each weight is either dropped (zero) or the
    /// undropped weight scaled by `1 / (1 - p)`.
    #[test]
    #[serial]
    fn test_sdpa_attn_weight_drops_out_after_softmax() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let shape = [2, 2, 4, 8];
        let q = seeded_tensor::<4>(1, shape, Distribution::Default, &device);
        let k = seeded_tensor::<4>(2, shape, Distribution::Default, &device);

        let p = 0.5;
        let config = ScaledDotProductAttentionConfig::new();
        let read = |t: Tensor<4>| t.into_data().try_to_vec_as::<f32>().unwrap();
        let kept = read(sdpa_attn_weight(
            q.clone(),
            k.clone(),
            None,
            None,
            None,
            config,
        ));
        let dropped = read(sdpa_attn_weight(
            q,
            k,
            None,
            None,
            None,
            config.with_dropout(Some(p)),
        ));

        let scale = (1.0 / (1.0 - p)) as f32;
        let mut n_dropped = 0;
        for (&w, &d) in kept.iter().zip(&dropped) {
            if d == 0.0 {
                n_dropped += 1;
            } else {
                let expected = w * scale;
                assert!(
                    (d - expected).abs() <= 1e-4 * expected.abs() + 1e-6,
                    "expected 0 or {expected} (= {w} / (1 - p)), got {d}"
                );
            }
        }
        // 128 weights, each dropped with p = 0.5.
        assert!(n_dropped > 0 && n_dropped < kept.len(), "{n_dropped}");
    }

    #[test]
    #[serial]
    fn test_scaled_dot_product_attention_bias() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let dtype = DType::F32;

        let l = 3;
        let s = 5;

        let ni = f32::NEG_INFINITY;

        sdpa_bias(l, s, false, None, None, dtype, &device)
            .to_data()
            .assert_eq(
                &Tensor::<2>::from_data(
                    [
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                    ],
                    &device,
                )
                .to_data(),
                false,
            );

        // +causal, -bias, -mask
        sdpa_bias(l, s, true, None, None, dtype, &device)
            .to_data()
            .assert_eq(
                &Tensor::<2>::from_data(
                    [
                        [0., ni, ni, ni, ni],
                        [0., 0., ni, ni, ni],
                        [0., 0., 0., ni, ni],
                    ],
                    &device,
                )
                .to_data(),
                false,
            );

        let bias = Tensor::<2>::from_data(
            [
                [1., 2., 3., 4., 5.],
                [6., 7., 8., 9., 10.],
                [11., 12., 13., 14., 15.],
            ],
            &device,
        );

        // -causal, +bias, -mask
        sdpa_bias(l, s, false, Some(bias.clone()), None, dtype, &device)
            .to_data()
            .assert_eq(
                &Tensor::<2>::from_data(
                    [
                        [1., 2., 3., 4., 5.],
                        [6., 7., 8., 9., 10.],
                        [11., 12., 13., 14., 15.],
                    ],
                    &device,
                )
                .to_data(),
                false,
            );

        let mask = Tensor::<2, Bool>::from_data(
            [
                [true, true, true, true, false],
                [true, true, true, true, true],
                [false, true, true, true, true],
            ],
            &device,
        );

        // -causal, +bias, +mask
        sdpa_bias(
            l,
            s,
            false,
            Some(bias.clone()),
            Some(mask.clone()),
            dtype,
            &device,
        )
        .to_data()
        .assert_eq(
            &Tensor::<2>::from_data(
                [
                    [1., 2., 3., 4., ni],
                    [6., 7., 8., 9., 10.],
                    [ni, 12., 13., 14., 15.],
                ],
                &device,
            )
            .to_data(),
            false,
        );

        // +causal, +mask, +bias
        sdpa_bias(
            l,
            s,
            true,
            Some(bias.clone()),
            Some(mask.clone()),
            dtype,
            &device,
        )
        .to_data()
        .assert_eq(
            &Tensor::<2>::from_data(
                [
                    [1., ni, ni, ni, ni],
                    [6., 7., ni, ni, ni],
                    [ni, 12., 13., ni, ni],
                ],
                &device,
            )
            .to_data(),
            false,
        );
    }

    /// With `softmax_dtype` f32, bf16 attention keeps the query's dtype and
    /// lands nearer the f32 attention than an all-bf16 softmax: the scores'
    /// exponentials are where bf16 loses most.
    #[test]
    fn test_softmax_dtype() {
        let device = cpu_device();
        let q = seeded_tensor::<4>(1, [1, 2, 6, 8], Distribution::Normal(0.0, 3.0), &device);
        let k = seeded_tensor::<4>(2, [1, 2, 6, 8], Distribution::Normal(0.0, 3.0), &device);
        let v = seeded_tensor::<4>(3, [1, 2, 6, 8], Distribution::Default, &device);
        let causal = ScaledDotProductAttentionConfig::new().with_is_causal(true);
        let f32_out =
            scaled_dot_product_attention(q.clone(), k.clone(), v.clone(), None, None, None, causal);
        let half = |t: &Tensor<4>| t.clone().cast(DType::BF16);
        let run = |config| {
            let out = scaled_dot_product_attention(
                half(&q),
                half(&k),
                half(&v),
                None,
                None,
                None,
                config,
            );
            assert_eq!(out.dtype(), DType::BF16);
            (out.cast(DType::F32) - f32_out.clone())
                .abs()
                .max()
                .into_scalar::<f32>()
        };
        let in_bf16 = run(causal);
        let in_f32 = run(causal.with_softmax_dtype(Some(FloatDType::F32)));
        assert!(
            in_f32 <= in_bf16,
            "f32 softmax {in_f32}, bf16 softmax {in_bf16}"
        );

        // A bias in another dtype is cast to the softmax's.
        let bias = Tensor::<2>::zeros([6, 6], &device);
        let out = scaled_dot_product_attention(
            half(&q),
            half(&k),
            half(&v),
            Some(bias),
            None,
            None,
            causal.with_softmax_dtype(Some(FloatDType::F32)),
        );
        assert_eq!(out.dtype(), DType::BF16);
    }
}
