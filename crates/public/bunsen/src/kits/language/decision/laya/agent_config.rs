//! A Laya checkpoint's head and calibration.
use alloc::collections::BTreeMap;

use super::LayaQuestionKind;

/// A Laya checkpoint's head and calibration: the head's geometry, the
/// prompt's token budgets, and the temperatures that calibrate each
/// question's logits. A release's is its `rl_agent_config.json`'s
/// ([`LayaAgentMirror::agent_config`](super::interop::laya::LayaAgentMirror::agent_config)).
#[derive(Clone, Debug, PartialEq)]
pub struct LayaAgentConfig {
    /// The head's encoder layers.
    pub head_layers: usize,

    /// A prompt's token budget.
    pub max_len: usize,

    /// The budget of a prompt's question and options, before its state.
    pub head_max_len: usize,

    /// Each costed act, by name; the act head has one output per act, and
    /// one for none.
    pub act_costs: BTreeMap<String, f64>,

    /// Each question kind's temperature, by [`LayaQuestionKind::index`].
    pub temperature: [f64; 3],

    /// Temperatures by question kind and option count, keyed as
    /// [`temperature_bucket`](Self::temperature_bucket) names them; a key
    /// absent here falls back to [`temperature`](Self::temperature).
    pub temperature_by_options: BTreeMap<String, f64>,
}

impl LayaAgentConfig {
    /// The act head's outputs.
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }

    /// The key of a question's temperature by option count: its kind's
    /// name, and its count's bucket, `2`, `3-5`, `6-10` or `11+`
    /// (`choice:3-5`).
    pub fn temperature_bucket(
        kind: LayaQuestionKind,
        options: usize,
    ) -> String {
        let size = match options {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        format!("{}:{size}", kind.name())
    }

    /// The temperature of a question of `kind` with `options` options: its
    /// bucket's, or its kind's.
    pub fn temperature(
        &self,
        kind: LayaQuestionKind,
        options: usize,
    ) -> f64 {
        self.temperature_by_options
            .get(&Self::temperature_bucket(kind, options))
            .copied()
            .unwrap_or(self.temperature[kind.index()])
    }

    /// A question's calibrated probabilities: the softmax of its options'
    /// `logits` over its temperature, in f64.
    pub fn probabilities(
        &self,
        kind: LayaQuestionKind,
        logits: &[f32],
    ) -> Vec<f64> {
        let temperature = self.temperature(kind, logits.len());
        let z: Vec<f64> = logits.iter().map(|&l| f64::from(l) / temperature).collect();
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = z.iter().map(|z| (z - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        exp.into_iter().map(|e| e / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::language::decision::laya::interop::laya::LayaAgentMirror;

    fn agent(json: &str) -> LayaAgentConfig {
        LayaAgentMirror::from_json(json).unwrap().agent_config()
    }

    const JSON: &str = r#"{
        "encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "max_len": 512,
        "head_max_len": 192, "act_costs": {"escalate": 0.5}, "amp_dtype": "bf16",
        "temperature": [1.5, 1.25, 2.0],
        "temperature_by_options": {"choice:3-5": 1.75, "noul:2": 1.9},
        "training": {"updates": 7313}
    }"#;

    /// A bucket's temperature, else the kind's.
    #[test]
    fn test_temperature() {
        use LayaQuestionKind::*;
        let config = agent(JSON);
        assert_eq!(LayaAgentConfig::temperature_bucket(Choice, 4), "choice:3-5");
        assert_eq!(LayaAgentConfig::temperature_bucket(Score, 11), "score:11+");
        assert_eq!(config.temperature(Choice, 4), 1.75);
        assert_eq!(config.temperature(Choice, 2), 1.5);
        assert_eq!(config.temperature(Noul, 2), 1.9);
        assert_eq!(config.temperature(Score, 7), 1.25);
    }

    /// The probabilities are the softmax at the temperature: two logits
    /// `ln 4 · T` apart are 0.8 and 0.2.
    #[test]
    fn test_probabilities() {
        let config = agent(JSON);
        let t = config.temperature(LayaQuestionKind::Noul, 2);
        let gap = (4.0f64.ln() * t) as f32;
        let p = config.probabilities(LayaQuestionKind::Noul, &[gap, 0.0]);
        assert!(
            (p[0] - 0.8).abs() < 1e-6 && (p[1] - 0.2).abs() < 1e-6,
            "{p:?}"
        );
    }
}
