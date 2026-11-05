//! Loading a Laya checkpoint's tensors into a [`LayaModel`], and its
//! config from its two files.
use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    LayaAgentConfig,
    LayaConfig,
    LayaModel,
};
use crate::{
    blocks::transformers::encoders::residual::ResidualEncoderAttentionBlock,
    burner::module::Rekey,
    data::pretrained::{
        NamedTensorsApplied,
        load_rekeyed,
        split_rows,
    },
    errors::BunsenResult,
    kits::text::encoders::modernbert::{
        ModernBertConfig,
        weights::{
            transformers_rekey as modernbert_rekey,
            transformers_row_splits as modernbert_row_splits,
        },
    },
};

/// The [`LayaConfig`] of a checkpoint: its encoder's (from
/// `encoder/config.json`), and its head's (from `rl_agent_config.json`).
pub fn laya_config(
    encoder: ModernBertConfig,
    agent: &LayaAgentConfig,
) -> LayaConfig {
    LayaConfig::new(encoder)
        .with_head_layers(agent.head_layers)
        .with_n_act(agent.n_act())
}

/// A Laya `model.safetensors`' names, once its fused projections are split
/// ([`load_laya_tensors`]), onto a [`LayaModel`]'s module paths: the
/// `ModernBERT` encoder under `encoder.`; the head's `PyTorch` encoder
/// layers, `head.layers.N.`
/// ([`ResidualEncoderAttentionBlock::torch_rekey`]); the type embedding;
/// the scorer's and the act head's `nn.Sequential` slots by name. Anything
/// else is dropped: the `temperature` buffer, which calibration reads from
/// `rl_agent_config.json` instead.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`load_rekeyed`] resolves to `gamma` / `beta`.
pub fn laya_rekey() -> Rekey {
    Rekey::drop_rest()
        .mount("encoder.", "encoder.", modernbert_rekey())
        .mount_indexed(
            "head.layers.",
            "head.",
            ResidualEncoderAttentionBlock::torch_rekey(),
        )
        .rename("type_emb.", "type_emb.")
        .rename("scorer.0.", "scorer.norm.")
        .rename("scorer.1.", "scorer.dense.")
        .rename("scorer.3.", "scorer.out.")
        .rename("act_head.0.", "act_head.dense.")
        .rename("act_head.2.", "act_head.out.")
}

/// Loads a Laya `model.safetensors`' tensors into `model`: the encoder's
/// and the head's fused projections split by rows, then [`load_rekeyed`]
/// under [`laya_rekey`]; in `dtype` (a load policy's resolution), or, when
/// `None`, in the dtype `model` was built in.
///
/// # Errors
/// As [`split_rows`] and [`load_rekeyed`].
pub fn load_laya_tensors(
    model: &mut LayaModel,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<burn::tensor::FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let splits: Vec<_> = modernbert_row_splits()
        .into_iter()
        .chain(ResidualEncoderAttentionBlock::torch_row_splits())
        .collect();
    let tensors = split_rows(tensors, &splits)?;
    load_rekeyed(model, tensors, &laya_rekey(), dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint's names land on the model's; the buffer is dropped.
    #[test]
    fn test_laya_rekey() {
        let rekey = laya_rekey();
        for (stored, bunsen) in [
            (
                "encoder.layers.3.attn.Wq.weight",
                "encoder.stack.layers.3.attn.c_q.weight",
            ),
            ("encoder.final_norm.weight", "encoder.stack.norm.weight"),
            (
                "head.layers.1.self_attn.q_proj.bias",
                "head.1.attn.query.bias",
            ),
            (
                "head.layers.0.self_attn.out_proj.weight",
                "head.0.attn.output.weight",
            ),
            ("head.layers.0.norm1.weight", "head.0.attn_norm.weight"),
            ("head.layers.0.norm2.bias", "head.0.mlp_norm.bias"),
            ("head.layers.1.linear2.weight", "head.1.mlp.linear2.weight"),
            ("type_emb.weight", "type_emb.weight"),
            ("scorer.0.weight", "scorer.norm.weight"),
            ("scorer.3.bias", "scorer.out.bias"),
            ("act_head.2.weight", "act_head.out.weight"),
        ] {
            assert_eq!(rekey.apply(stored).as_deref(), Some(bunsen), "{stored}");
        }
        assert_eq!(rekey.apply("temperature"), None);
    }
}
