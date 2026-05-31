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

/// Compute layer normalized self-attn.
///
/// ## Arguments
/// * `layer_norm` - `LayerNorm`.
/// * `mh_attn` - `MultiHeadAttention`.
/// * `x` - ``[batch, seq_len, d_model]`` input.
/// * `mask` - Optional ``[batch, seq_len, seq_len]`` attention mask.
///
/// ## Returns
/// `RdabForwardRecord` - forward record.
/// * `fr.output` : ``[batch, seq_len, d_model]``.
/// * `fr.ca_weights` : ``[batch, n_heads, seq_len, seq_len]``.
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

/// Compute layer normalized cross-attn.
///
/// ## Arguments
/// * `layer_norm` - `LayerNorm`.
/// * `mh_attn` - `MultiHeadAttention`.
/// * `x` - ``[batch, seq_len, d_model]`` input.
/// * `xa` - ``[batch, seq_len, d_model]`` cross-attention input.
///
/// ## Returns
/// `RdabForwardRecord` - forward record.
/// * `fr.output` : ``[batch, seq_len, d_model]``.
/// * `fr.ca_weights` : ``[batch, n_heads, seq_len, seq_len]``.
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

        let [batch, seq_len] =
            CONTRACT.unpack_shape(&x, &["batch", "seq_len"], &[("d_model", d_model)]);

        CONTRACT.assert_shape(
            &xa,
            &[("batch", batch), ("seq_len", seq_len), ("d_model", d_model)],
        );
    };

    mh_attn.forward(MhaInput::new(layer_norm.forward(x.clone()), xa.clone(), xa))
}
