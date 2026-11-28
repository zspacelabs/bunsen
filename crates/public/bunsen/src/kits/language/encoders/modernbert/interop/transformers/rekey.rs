//! The state-dict names.

use crate::{
    blocks::transformers::decoders::causal::{
        CausalDecoderBlock,
        DecoderLayerKeys,
    },
    burner::module::{
        Rekey,
        TensorLoadRowSplit,
    },
    kits::support::transformers::interop::llama_layer_keys,
};

/// The fused projections of a `transformers` `ModernBERT` layer, split by
/// rows into the parts the block holds apart, named in the checkpoint's
/// terms: `attn.Wqkv` into `attn.Wq`, `attn.Wk` and `attn.Wv`; `mlp.Wi`
/// into `mlp.Wi_gate`, its first half (upstream's `input`, the half the
/// activation takes), and `mlp.Wi_up`, its second (upstream's `gate`, the
/// half it multiplies).
pub fn modernbert_row_splits() -> [TensorLoadRowSplit; 2] {
    [
        TensorLoadRowSplit::equal(
            ".attn.Wqkv.weight",
            [".attn.Wq.weight", ".attn.Wk.weight", ".attn.Wv.weight"],
        ),
        TensorLoadRowSplit::equal(
            ".mlp.Wi.weight",
            [".mlp.Wi_gate.weight", ".mlp.Wi_up.weight"],
        ),
    ]
}

/// The names of a `transformers` `ModernBERT` layer's parts, once
/// [`modernbert_row_splits`] has split its projections: `attn_norm` and
/// `mlp_norm`, and `attn`'s `Wq`, `Wk`, `Wv` and `Wo`.
pub fn modernbert_layer_keys() -> DecoderLayerKeys {
    DecoderLayerKeys {
        attn: "attn".to_string(),
        attn_projs: ["Wq", "Wk", "Wv", "Wo"].map(String::from),
        ..llama_layer_keys("attn_norm", "mlp_norm")
    }
}

/// `transformers`' `ModernBertModel` state-dict names, after
/// [`modernbert_row_splits`], onto a
/// [`ModernBertModel`](crate::kits::language::encoders::modernbert::ModernBertModel)'s module paths,
/// relative to the encoder: the embedding's table and norm, each layer by
/// [`modernbert_layer_keys`] and its `GeGLU`'s parts (`Wi_gate`, `Wi_up`,
/// `Wo`), and the final norm. Anything else is dropped (a `ModernBertFor*`'s
/// head).
///
/// Normalization parameters keep their `weight` spellings, which
/// [`Rekey::load`](crate::burner::module::Rekey::load) resolves to `gamma`.
pub fn modernbert_rekey() -> Rekey {
    let layer = CausalDecoderBlock::layer_rekey(&modernbert_layer_keys())
        .rename("mlp.Wi_gate.", "mlp.gate_proj.")
        .rename("mlp.Wi_up.", "mlp.up_proj.")
        .rename("mlp.Wo.", "mlp.down_proj.");
    Rekey::drop_rest()
        .rename("embeddings.tok_embeddings.", "embeddings.")
        .rename("embeddings.norm.", "embeddings.norm.")
        .mount_indexed("layers.", "stack.layers.", layer)
        .rename("final_norm.", "stack.norm.")
}
