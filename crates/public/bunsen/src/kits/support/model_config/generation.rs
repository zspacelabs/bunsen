//! `GenerationPolicy`: how a chat model generates, as its presets set it.

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::{
    Map,
    Value,
};

/// How a chat model generates a reply: the call-time half of its config,
/// under a `generation` field of a family's config, which its presets and
/// probes patch.
///
/// bunsen carries it uninterpreted: a chat driver reads it. The template
/// kwargs, `enable_thinking` say, are opaque here; what they mean is the
/// chat template's business. A field it is read without takes its default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationPolicy {
    /// The tokens that end a reply.
    pub stop_ids: Vec<i64>,

    /// How a token is drawn: sampled under these settings, or the argmax when
    /// `None`.
    pub sampling: Option<SamplingPolicy>,

    /// The most tokens a reply generates, when the policy names a cap.
    pub max_new_tokens: Option<usize>,

    /// The chat template's keyword arguments, as the template reads them.
    pub template_kwargs: Map<String, Value>,
}

/// How a token is sampled: a temperature, and the filters before the draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplingPolicy {
    /// The softmax temperature.
    pub temperature: f64,

    /// Keep the `k` most likely tokens, when set.
    pub top_k: Option<usize>,

    /// Keep the smallest set of tokens holding this much probability, when
    /// set.
    pub top_p: Option<f64>,

    /// Drop the tokens less likely than this share of the most likely, when
    /// set.
    pub min_p: Option<f64>,
}
