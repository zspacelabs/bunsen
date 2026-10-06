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
        activation::softmax,
    },
};

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

    let attn_weight = sdpa_attn_weight(q, k, bias, mask, config);

    let mut v = v;
    if config.enable_gqa {
        let v_repeats = h_q / h_kv;
        v = repeat::repeat_interleave::<4, 5, _>(v, v_repeats, 1);
    }

    attn_weight.matmul(v)
}

/// Builds the Attention Weight for [`scaled_dot_product_attention`].
///
/// The weight is `softmax(q k^T * scale + bias)` over the keys, then
/// dropout on those probabilities when the config sets a rate, as in
/// `PyTorch`'s reference.
///
/// # Arguments
/// - `q`: the query tensor, as `[B, H_q, T_q, D]`.
/// - `k`: the key tensor, as `[B, H_k, T_k, D]`.
/// - `bias`: optional additive bias, as `[T_q, T_k]`.
/// - `mask`: optional bias mask, as `[T_q, T_k]`.
/// - `config`: attention config.
pub fn sdpa_attn_weight(
    q: Tensor<4>,
    k: Tensor<4>,
    bias: Option<Tensor<2>>,
    mask: Option<Tensor<2, Bool>>,
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
    let attn_weight = q.matmul(k.swap_dims(2, 3)) * scale_factor;

    let attn_bias = sdpa_bias(t_q, t_k, config.is_causal, bias, mask, dtype, &device);
    let mut attn_weight = softmax(attn_weight + attn_bias.unsqueeze(), 3);

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
/// - `bias`: optional additive bias.
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
        attn_bias = attn_bias + bias;
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
        performance_device,
        seeded_tensor,
    };

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
        let kept = read(sdpa_attn_weight(q.clone(), k.clone(), None, None, config));
        let dropped = read(sdpa_attn_weight(
            q,
            k,
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
}
