//! The config file's typed mirror.

use alloc::collections::BTreeMap;

use serde::Deserialize;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::language::decision::laya::LayaAgentConfig,
};

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
