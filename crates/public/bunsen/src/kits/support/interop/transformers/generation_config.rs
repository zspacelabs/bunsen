//! `transformers`' `generation_config.json`.
use alloc::vec::Vec;

use serde::Deserialize;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::model_config::SamplingPolicy,
};

/// A `transformers` `generation_config.json`, as it names its fields: how a
/// checkpoint generates, the fields bunsen reads. Every field is optional,
/// as the files leave them out; every other key is ignored.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct TransformersGenerationConfigMirror {
    /// The ids that end a reply: one, or several.
    #[serde(default)]
    pub eos_token_id: Option<TransformersTokenIds>,

    /// Whether a token is sampled; `false` or absent, the argmax.
    #[serde(default)]
    pub do_sample: Option<bool>,

    /// The softmax temperature; absent, `transformers`' 1.
    #[serde(default)]
    pub temperature: Option<f64>,

    /// The top-k filter.
    #[serde(default)]
    pub top_k: Option<usize>,

    /// The top-p filter.
    #[serde(default)]
    pub top_p: Option<f64>,

    /// The min-p filter.
    #[serde(default)]
    pub min_p: Option<f64>,

    /// The most tokens a reply generates.
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
}

/// One token id, or several, as `transformers` writes an `eos_token_id`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum TransformersTokenIds {
    /// One id.
    One(i64),

    /// Several, in order.
    Many(Vec<i64>),
}

impl TransformersGenerationConfigMirror {
    /// Parses a `generation_config.json`'s bytes.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it is not a JSON object of these fields' types.
    pub fn from_slice(json: &[u8]) -> BunsenResult<Self> {
        let value: serde_json::Value = serde_json::from_slice(json)
            .map_err(|e| BunsenError::invalid_resource("a generation_config.json").with_cause(e))?;
        if !value.is_object() {
            return Err(BunsenError::invalid_resource(
                "a generation_config.json that is not an object",
            ));
        }
        serde_json::from_value(value)
            .map_err(|e| BunsenError::invalid_resource("a generation_config.json").with_cause(e))
    }

    /// The ids that end a reply, in order, when the file names any.
    pub fn stop_ids(&self) -> Option<Vec<i64>> {
        self.eos_token_id.as_ref().map(|ids| match ids {
            TransformersTokenIds::One(id) => alloc::vec![*id],
            TransformersTokenIds::Many(ids) => ids.clone(),
        })
    }

    /// How a token is drawn: the file's sampling under `do_sample`, or
    /// `None` (the argmax) without it.
    pub fn sampling_policy(&self) -> Option<SamplingPolicy> {
        (self.do_sample == Some(true)).then(|| SamplingPolicy {
            temperature: self.temperature.unwrap_or(1.0),
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config() {
        let c = TransformersGenerationConfigMirror::from_slice(
            br#"{"bos_token_id": 151643, "do_sample": true, "eos_token_id": [151645, 151643],
                "temperature": 0.6, "top_k": 20, "top_p": 0.95}"#,
        )
        .unwrap();
        assert_eq!(c.stop_ids(), Some(alloc::vec![151645, 151643]));
        assert_eq!(
            c.sampling_policy(),
            Some(SamplingPolicy {
                temperature: 0.6,
                top_k: Some(20),
                top_p: Some(0.95),
                min_p: None,
            })
        );
        let greedy =
            TransformersGenerationConfigMirror::from_slice(br#"{"eos_token_id": 2}"#).unwrap();
        assert_eq!(greedy.stop_ids(), Some(alloc::vec![2]));
        assert_eq!(greedy.sampling_policy(), None);
        assert!(TransformersGenerationConfigMirror::from_slice(b"[1]").is_err());
    }
}
