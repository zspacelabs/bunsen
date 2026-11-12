//! # Laya cross-check against upstream.
//!
//! Validates [`bunsen`]'s Laya (`kits::decision::laya`), its `ModernBERT`
//! encoder (`kits::text::encoders::modernbert`) with it, against upstream's
//! own decision model (`rl_common.py` and `rl_agent_api.py` of
//! `convaiinnovations/laya`), in float32, on each of its three checkpoints,
//! [`ROWS`]: for every question of every case, bunsen's prompt is upstream's
//! (the ids and the option markers, through bunsen's `BunsenTokenizer`);
//! its encoder's states at the first token and each marker are upstream's;
//! its option and act logits are upstream's; and its calibrated answers,
//! every question of a case decided in one padded batch, are upstream's
//! probabilities.
//!
//! ## This crate hosts nothing
//!
//! The checkpoints come through bunsen's own pinned rows, fetched at run time
//! into the weights cache by the tests that read them (feature
//! **`checkpoints`**), which is off by default. No Laya file is committed:
//! the fixture is texts, ids and recorded tensors
//! (`validation/tools/gen_laya_fixtures.py` writes it):
//!
//! - `testdata/upstream.json`: for each row, its cases (a state and its
//!   questions) and, for each question, upstream's prompt, encoder states,
//!   logits and probabilities, each question run alone.
//!
//! `cargo validate` (`validation/tools/validate`) runs this crate under a
//! profile (`validation/models.yml`). Every test of a model has the model's
//! id in its path (`upstream::laya`), which is how a profile's filter keeps
//! or drops it:
//!
//! ```sh
//! cd validation && cargo validate run --profile medium-core --crate laya
//! ```

#![cfg_attr(not(feature = "checkpoints"), allow(unused))]

use std::collections::BTreeMap;

use bunsen::{
    data::formats::LoadPolicy,
    kits::decision::laya::{
        LayaConstruct,
        LayaQuestion,
    },
    support::testing::TensorRecord,
};
use burn::{
    prelude::*,
    tensor::FloatDType,
};
use serde::Deserialize;

/// The rows `testdata/upstream.json` holds.
pub const ROWS: [&str; 3] = ["laya", "laya-multilingual", "laya-typed-decisions"];

/// One row of `testdata/upstream.json`.
#[derive(Debug, Deserialize)]
pub struct RowRecord {
    /// Its cases.
    pub cases: Vec<CaseRecord>,
}

/// A state, the questions asked about it, and upstream's answers.
#[derive(Debug, Deserialize)]
pub struct CaseRecord {
    /// The state.
    pub state: String,
    /// The questions, in bunsen's schema.
    pub questions: Vec<LayaQuestion>,
    /// Upstream's computation for each question, run alone.
    pub prompts: Vec<PromptRecord>,
}

/// What upstream computes for one question.
#[derive(Debug, Deserialize)]
pub struct PromptRecord {
    /// `build_sequence`'s prompt.
    pub ids: Vec<i64>,
    /// Its options' `[MASK]` positions.
    pub markers: Vec<usize>,
    /// The encoder's last hidden state at the first token and each marker,
    /// `[1 + options, d]`.
    pub encoder: TensorRecord,
    /// The scorer's logits at the markers.
    pub logits: Vec<f64>,
    /// The act head's logits.
    pub act_logits: Vec<f64>,
    /// The calibrated probabilities.
    pub probabilities: Vec<f64>,
    /// The act head's probability of its first act.
    pub act_probability: f64,
}

/// Reads `testdata/upstream.json`, by row.
pub fn upstream_reference() -> BTreeMap<String, RowRecord> {
    let path = format!("{}/testdata/upstream.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// The weights cache, offline under `BUNSEN_VALIDATE_OFFLINE`.
pub fn weights_cache() -> bunsen::data::resources::ResourceCache {
    use bunsen::data::resources::{
        ResourceCache,
        ResourceCacheOptions,
    };
    let offline = std::env::var_os("BUNSEN_VALIDATE_OFFLINE").is_some();
    ResourceCache::new(ResourceCacheOptions::default().with_offline(offline))
        .expect("open the resource cache")
}

/// A Laya pretrained by name, in f32 at least: what the checks hold to
/// upstream's f32.
pub fn load_construct(
    name: &str,
    device: &Device,
) -> std::sync::Arc<LayaConstruct> {
    bunsen::kits::decision::laya::pretrained::default_laya_factory()
        .expect("the laya factory")
        .load_construct(
            name,
            &weights_cache(),
            device,
            LoadPolicy::at_least(FloatDType::F32),
        )
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
}

#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod upstream;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is there, each question with its prompt, its markers inside
    /// its ids, and a distribution over its options.
    #[test]
    fn test_fixture_integrity() {
        let fixture = upstream_reference();
        assert_eq!(fixture.keys().map(String::as_str).collect::<Vec<_>>(), ROWS);
        for (row, record) in &fixture {
            assert!(!record.cases.is_empty(), "{row}");
            for case in &record.cases {
                assert_eq!(case.questions.len(), case.prompts.len(), "{row}");
                for (q, p) in case.questions.iter().zip(&case.prompts) {
                    let k = q.rendered_options().len();
                    assert_eq!(p.markers.len(), k, "{row}: {}", q.question);
                    assert!(p.markers.iter().all(|&m| m < p.ids.len()), "{row}");
                    assert_eq!(p.logits.len(), k, "{row}");
                    let sum: f64 = p.probabilities.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-9, "{row}: {sum}");
                    assert_eq!(p.encoder.shape[0], 1 + k, "{row}");
                    assert_eq!(
                        p.encoder.samples.len(),
                        p.encoder.expected_samples(),
                        "{row}"
                    );
                }
            }
        }
    }
}
