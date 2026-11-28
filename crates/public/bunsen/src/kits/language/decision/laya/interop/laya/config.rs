//! The config conversion.

use crate::kits::language::{
    decision::laya::{
        LayaAgentConfig,
        LayaConfig,
    },
    encoders::modernbert::ModernBertContractConfig,
};

/// The [`LayaConfig`] of a checkpoint: its encoder's (from
/// `encoder/config.json`), and its head's (from `rl_agent_config.json`,
/// [`LayaAgentMirror::agent_config`](super::LayaAgentMirror::agent_config)).
pub fn laya_config(
    encoder: ModernBertContractConfig,
    agent: &LayaAgentConfig,
) -> LayaConfig {
    LayaConfig::new(encoder)
        .with_head_layers(agent.head_layers)
        .with_n_act(agent.n_act())
}
