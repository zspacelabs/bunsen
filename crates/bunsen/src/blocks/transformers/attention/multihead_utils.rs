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
};

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

            assert_shape_contract!(["b", "seq_len", "seq_len"], mask, &[("seq_len", seq_len)]);
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
/// query sequence over a *different* one. Whisper's decoder attends 4 tokens
/// over 1500 audio frames.
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

#[cfg(test)]
mod tests {
    use burn::{
        nn::{
            LayerNormConfig,
            attention::MultiHeadAttentionConfig,
        },
        tensor::Distribution,
    };

    use super::*;
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;

    /// Cross-attention must accept a `xa` whose sequence length differs from
    /// the query's — that is the entire point of it.
    ///
    /// Whisper's decoder attends a handful of tokens over 1500 audio frames,
    /// and a contract that bound both to one `seq_len` made that impossible
    /// while every same-length unit test still passed.
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
}
