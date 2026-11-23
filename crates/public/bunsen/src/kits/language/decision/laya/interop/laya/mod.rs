//! # Laya as its release saves it
//!
//! A Laya checkpoint (`convaiinnovations/laya`): a `transformers`
//! `ModernBERT` encoder's `encoder/config.json`, the head's
//! `rl_agent_config.json`, and one `model.safetensors` (the encoder under
//! `encoder.`, the head's `PyTorch` layers under `head.layers.`), as bunsen's
//! [`LayaModel`](crate::kits::language::decision::laya::LayaModel):
//!
//! - **Mirror:** [`LayaAgentMirror`] (`rl_agent_config.json`), its
//!   [`agent_config`](LayaAgentMirror::agent_config) the head and calibration;
//!   [`laya_config`] the model's, from both files.
//! - **Names:** [`laya_rekey`], the encoder's by [`modernbert_rekey`].
//! - **Loading** (feature `store`): `load_laya_tensors`; with `cache`,
//!   `load_laya`, a resource set's checkpoint.

use alloc::collections::BTreeMap;

use serde::Deserialize;

use crate::{
    burner::module::Rekey,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::language::{
        decision::laya::{
            LayaAgentConfig,
            LayaConfig,
        },
        encoders::modernbert::{
            ModernBertContractConfig,
            interop::transformers::modernbert_rekey,
        },
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// A Laya checkpoint's `rl_agent_config.json`, under upstream's field names:
/// the head's geometry, the prompt's token budgets, and the temperatures
/// that calibrate each question's logits. Fields the inference does not
/// read (training's) are ignored.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LayaAgentMirror {
    /// The head's encoder layers.
    pub head_layers: usize,

    /// A prompt's token budget.
    pub max_len: usize,

    /// The budget of a prompt's question and options, before its state.
    pub head_max_len: usize,

    /// Each costed act, by name.
    #[serde(default)]
    pub act_costs: BTreeMap<String, f64>,

    /// Each question kind's temperature.
    pub temperature: [f64; 3],

    /// Temperatures by question kind and option count.
    #[serde(default)]
    pub temperature_by_options: BTreeMap<String, f64>,
}

impl LayaAgentMirror {
    /// Parses an `rl_agent_config.json`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse as one.
    pub fn from_json(json: &str) -> BunsenResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("an rl_agent_config.json").with_cause(e))
    }

    /// The head and calibration the file describes.
    pub fn agent_config(&self) -> LayaAgentConfig {
        LayaAgentConfig {
            head_layers: self.head_layers,
            max_len: self.max_len,
            head_max_len: self.head_max_len,
            act_costs: self.act_costs.clone(),
            temperature: self.temperature,
            temperature_by_options: self.temperature_by_options.clone(),
        }
    }
}

/// The [`LayaConfig`] of a checkpoint: its encoder's (from
/// `encoder/config.json`), and its head's (from `rl_agent_config.json`,
/// [`LayaAgentMirror::agent_config`]).
pub fn laya_config(
    encoder: ModernBertContractConfig,
    agent: &LayaAgentConfig,
) -> LayaConfig {
    LayaConfig::new(encoder)
        .with_head_layers(agent.head_layers)
        .with_n_act(agent.n_act())
}

/// A Laya `model.safetensors`' names, once its fused projections are split
/// (`load_laya_tensors`), onto a
/// [`LayaModel`](crate::kits::language::decision::laya::LayaModel)'s module paths: the
/// `ModernBERT` encoder under `encoder.`; the head's `PyTorch` encoder
/// layers, `head.layers.N.` (a
/// [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)'s)
/// ([`transformer_encoder_layer_rekey`](crate::kits::support::interop::torch::transformer_encoder_layer_rekey)); the type embedding;
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
            crate::kits::support::interop::torch::transformer_encoder_layer_rekey(),
        )
        .rename("type_emb.", "type_emb.")
        .rename("scorer.0.", "scorer.norm.")
        .rename("scorer.1.", "scorer.mlp.linear1.")
        .rename("scorer.3.", "scorer.mlp.linear2.")
        .rename("act_head.0.", "act_head.linear1.")
        .rename("act_head.2.", "act_head.linear2.")
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
                "head.layers.1.attn.c_q.bias",
            ),
            (
                "head.layers.0.self_attn.out_proj.weight",
                "head.layers.0.attn.c_proj.weight",
            ),
            (
                "head.layers.0.norm1.weight",
                "head.layers.0.attn_norm.weight",
            ),
            ("head.layers.0.norm2.bias", "head.layers.0.mlp_norm.bias"),
            (
                "head.layers.1.linear2.weight",
                "head.layers.1.mlp.linear2.weight",
            ),
            ("type_emb.weight", "type_emb.weight"),
            ("scorer.0.weight", "scorer.norm.weight"),
            ("scorer.3.bias", "scorer.mlp.linear2.bias"),
            ("act_head.2.weight", "act_head.linear2.weight"),
        ] {
            assert_eq!(rekey.apply(stored).as_deref(), Some(bunsen), "{stored}");
        }
        assert_eq!(rekey.apply("temperature"), None);
    }

    const JSON: &str = r#"{
        "encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "max_len": 512,
        "head_max_len": 192, "act_costs": {"escalate": 0.5}, "amp_dtype": "bf16",
        "temperature": [1.5, 1.25, 2.0],
        "temperature_by_options": {"choice:3-5": 1.75, "noul:2": 1.9},
        "training": {"updates": 7313}
    }"#;

    #[test]
    fn test_parse() {
        let config = LayaAgentMirror::from_json(JSON).unwrap().agent_config();
        assert_eq!(
            (config.head_layers, config.max_len, config.head_max_len),
            (2, 512, 192)
        );
        assert_eq!(config.n_act(), 2);
        assert!(LayaAgentMirror::from_json("{}").is_err());
    }
}
