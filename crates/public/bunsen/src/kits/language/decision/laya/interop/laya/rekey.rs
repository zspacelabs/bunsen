//! The state-dict names.

use crate::{
    burner::module::Rekey,
    kits::language::encoders::modernbert::interop::transformers::modernbert_rekey,
};

/// A Laya `model.safetensors`' names, once its fused projections are split
/// (`load_laya_tensors`), onto a
/// [`LayaModel`](crate::kits::language::decision::laya::LayaModel)'s module paths: the
/// `ModernBERT` encoder under `encoder.`; the head's `PyTorch` encoder
/// layers, `head.layers.N.` (a
/// [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)'s)
/// ([`transformer_encoder_layer_rekey`](crate::kits::support::torch::interop::transformer_encoder_layer_rekey)); the type embedding;
/// the scorer's and the act head's `nn.Sequential` slots by name. Anything
/// else is dropped: the `temperature` buffer, which calibration reads from
/// `rl_agent_config.json` instead.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`Rekey::load`](crate::burner::module::Rekey::load) resolves to `gamma` /
/// `beta`.
pub fn laya_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount("encoder.", "encoder.", modernbert_rekey())
        .mount_indexed(
            "head.layers.",
            "head.layers.",
            crate::kits::support::torch::interop::transformer_encoder_layer_rekey(),
        )
        .rename("type_emb.", "type_emb.")
        .rename("scorer.0.", "scorer.norm.")
        .rename("scorer.1.", "scorer.mlp.linear1.")
        .rename("scorer.3.", "scorer.mlp.linear2.")
        .rename("act_head.0.", "act_head.linear1.")
        .rename("act_head.2.", "act_head.linear2.")
}
