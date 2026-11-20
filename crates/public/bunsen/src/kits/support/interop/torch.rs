//! # `torch`: `PyTorch`'s own modules' names, for any kit
//!
//! A checkpoint of a model built from `torch.nn`'s modules names their
//! parameters as those modules do. The key sets here map them onto bunsen's
//! blocks:
//!
//! - an `nn.TransformerEncoderLayer`: [`transformer_encoder_layer_row_splits`]
//!   (its fused input projection, split) and
//!   [`transformer_encoder_layer_rekey`];
//! - an `nn.LSTMCell` and an `nn.LSTM`: [`lstm_cell_rekey`] and
//!   [`lstm_stack_rekey`].

use alloc::format;

#[cfg(doc)]
use crate::blocks::{
    rnn::lstm::FusedLstm,
    transformers::encoders::residual::ResidualEncoderAttentionBlock,
};
use crate::burner::module::{
    Rekey,
    TensorLoadRowSplit,
};

/// The fused projections of a `PyTorch` `nn.TransformerEncoderLayer`,
/// split by rows into the attention's three: `self_attn.in_proj_weight`
/// and `self_attn.in_proj_bias` into
/// `self_attn.{q,k,v}_proj.{weight,bias}`.
pub fn transformer_encoder_layer_row_splits() -> [TensorLoadRowSplit; 2] {
    [
        TensorLoadRowSplit::equal(
            ".self_attn.in_proj_weight",
            [
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
            ],
        ),
        TensorLoadRowSplit::equal(
            ".self_attn.in_proj_bias",
            [
                ".self_attn.q_proj.bias",
                ".self_attn.k_proj.bias",
                ".self_attn.v_proj.bias",
            ],
        ),
    ]
}

/// A `PyTorch` `nn.TransformerEncoderLayer`'s names, once
/// [`transformer_encoder_layer_row_splits`] has split its
/// projections, onto a
/// [`ResidualEncoderAttentionBlock`]'s, relative to it: `self_attn`'s
/// `{q,k,v,out}_proj` the attention, `norm1` and `norm2` the norms,
/// `linear{1,2}` the MLP; everything else as it is. With `norm_first`, the
/// block is pre-norm; its `ReLU` is the block's `activation`.
pub fn transformer_encoder_layer_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("self_attn.q_proj.", "attn.query.")
        .rename("self_attn.k_proj.", "attn.key.")
        .rename("self_attn.v_proj.", "attn.value.")
        .rename("self_attn.out_proj.", "attn.output.")
        .rename("norm1.", "attn_norm.")
        .rename("norm2.", "mlp_norm.")
        .rename("linear1.", "mlp.linear1.")
        .rename("linear2.", "mlp.linear2.")
}

/// A `PyTorch` `nn.LSTMCell`'s tensors, relative to it, onto a
/// [`FusedLstm`]: `{weight,bias}_ih` is `features`, `{weight,bias}_hh` is
/// `hidden`. Anything else is dropped.
pub fn lstm_cell_rekey() -> Rekey {
    Rekey::drop_rest()
        .exact("weight_ih", "features.weight")
        .exact("bias_ih", "features.bias")
        .exact("weight_hh", "hidden.weight")
        .exact("bias_hh", "hidden.bias")
}

/// A `PyTorch` `nn.LSTM`'s tensors, relative to it, onto a stack of
/// [`FusedLstm`]s (`Vec<FusedLstm>`, one per layer): `{param}_l{n}` is layer
/// `n`'s `param`, as a cell's
/// ([`lstm_cell_rekey`]). Anything else is
/// dropped.
pub fn lstm_stack_rekey() -> Rekey {
    let cell = lstm_cell_rekey();
    Rekey::drop_rest().custom(move |name| {
        let (param, layer) = name.rsplit_once("_l")?;
        Some(cell.apply(param).map(|path| format!("{layer}.{path}")))
    })
}
