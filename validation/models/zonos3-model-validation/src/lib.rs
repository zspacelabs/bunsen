//! # ZONOS3-Mini cross-check against Zyphra's reference.
//!
//! Validates [`bunsen`]'s ZONOS3-Mini (`kits::speech::tts::zonos3`) against
//! upstream's reference implementation over the pretrained checkpoint
//! (`Zyphra/ZONOS3-Mini-hf`, bunsen's row [`MODEL`]):
//!
//! - its tokenizer's ids for the UDHR texts, and the prompt rows it lays out
//!   for each of upstream's own prompt requests ([`fixtures::REQUESTS`]), from
//!   their normalized text (`prompts`, on the CPU, without the weights);
//! - its language model's logits in float32, over the reference's prompts: the
//!   prefill, then [`fixtures::STEPS`] decode steps teacher-forced through its
//!   cache (`logits`);
//! - its codec's audio from fixed codes, in float32 (`codec`);
//! - the logits again in bf16 on CUDA, against the reference's own bf16
//!   (`tests/cuda_bf16.rs`).
//!
//! ## This crate hosts nothing
//!
//! The checkpoint comes through bunsen's own pinned row, fetched at run time
//! into the weights cache by the tests that read it (feature
//! **`checkpoints`**), which is off by default. No ZONOS3 file is committed:
//! the fixtures are rows, ids, logit summaries and audio summaries
//! (`validation/tools/gen_zonos3_fixtures.py` writes them; [`fixtures`]
//! reads them):
//!
//! - `testdata/reference.safetensors`: upstream's float32 reference language
//!   model over upstream's own prompt requests, its prefill logits and
//!   teacher-forced decode steps;
//! - `testdata/reference_bf16.safetensors`: the same in bfloat16;
//! - `testdata/codec.safetensors`: the `-hf` codec's audio from fixed codes;
//! - `testdata/tokenizer.safetensors`: the `-hf` tokenizer's ids for the UDHR
//!   texts.
//!
//! `cargo validate` (`validation/tools/validate`) runs this crate under a
//! profile (`validation/models.yml`). Every test of the model has its id in
//! its path (`logits::zonos3_mini`):
//!
//! ```sh
//! cd validation && cargo validate run --profile medium-core --crate zonos3
//! ```

#![cfg_attr(not(feature = "checkpoints"), allow(unused))]

pub mod fixtures;

use std::sync::Arc;

use bunsen::{
    data::formats::LoadPolicy,
    kits::speech::tts::zonos3::{
        Zonos3Construct,
        lm::Zonos3RowConditioning,
        prompt::Zonos3Prompt,
    },
    ops::transformers::attention::KVCacheStorage,
};
use burn::{
    prelude::*,
    tensor::FloatDType,
};
use fixtures::{
    CODEBOOKS,
    FixtureFile,
    ROW_WIDTH,
};

/// bunsen's row for ZONOS3-Mini.
pub const MODEL: &str = "zonos3-mini";

/// The weights cache the tests load the checkpoint through: bunsen's own,
/// offline when `BUNSEN_VALIDATE_OFFLINE` is set, for a run on a warm cache.
#[cfg(feature = "checkpoints")]
pub fn weights_cache() -> bunsen::data::resources::ResourceCache {
    use bunsen::data::resources::{
        ResourceCache,
        ResourceCacheOptions,
    };
    let offline = std::env::var_os("BUNSEN_VALIDATE_OFFLINE").is_some();
    ResourceCache::new(ResourceCacheOptions::default().with_offline(offline))
        .expect("open the resource cache")
}

/// ZONOS3-Mini, in f32 at least: what the f32 checks hold to the reference's
/// f32.
#[cfg(feature = "checkpoints")]
pub fn load_construct(device: &Device) -> Arc<Zonos3Construct> {
    load_construct_with(device, LoadPolicy::at_least(FloatDType::F32))
}

/// ZONOS3-Mini under `policy`.
#[cfg(feature = "checkpoints")]
pub fn load_construct_with(
    device: &Device,
    policy: LoadPolicy,
) -> Arc<Zonos3Construct> {
    bunsen::kits::speech::tts::zonos3::pretrained::default_zonos3_factory()
        .expect("the zonos3 factory")
        .load_construct(MODEL, &weights_cache(), device, policy)
        .unwrap_or_else(|e| panic!("load {MODEL}: {e}"))
}

/// The reference's prompt for the request `name`, as [`REFERENCE`] holds
/// it: its rows and its replaced rows.
///
/// [`REFERENCE`]: fixtures::REFERENCE
pub fn reference_prompt(
    reference: &FixtureFile,
    name: &str,
) -> Zonos3Prompt {
    let facts = reference.request(name);
    let at = |k: &str| format!("{name}.{k}");
    let rows = reference
        .ints(&at("rows"))
        .chunks(ROW_WIDTH)
        .map(|r| r.iter().map(|&id| id as u32).collect())
        .collect();
    Zonos3Prompt {
        rows,
        speaker: facts
            .speaker_row
            .map(|row| (row, reference.floats(&at("speaker_embedding")))),
        first_condition_row: facts.condition_row.unwrap_or(0),
        condition_buckets: facts
            .condition_buckets
            .iter()
            .map(|&b| b as usize)
            .collect(),
        silence_rows: facts.silence_rows,
    }
}

/// The rows the reference sampled for the request `name`, `[STEPS][11]`.
pub fn reference_generated(
    reference: &FixtureFile,
    name: &str,
) -> Vec<Vec<i64>> {
    reference
        .ints(&format!("{name}.generated"))
        .chunks(ROW_WIDTH)
        .map(<[i64]>::to_vec)
        .collect()
}

/// bunsen's logits over a prompt and teacher-forced rows: the prefill's,
/// `[R][10][V]` flattened, and each fed row's, `[10][V]` flattened.
pub struct LmRun {
    /// The prefill's logits.
    pub prefill: Vec<f32>,

    /// The prefill's rows.
    pub rows: usize,

    /// Each step's logits.
    pub steps: Vec<Vec<f32>>,

    /// The vocabulary.
    pub vocab: usize,
}

/// Runs `construct`'s language model over `prompt`, then feeds `fed` one
/// row at a time through its decode cache.
pub fn run_lm(
    construct: &Zonos3Construct,
    prompt: &Zonos3Prompt,
    fed: &[Vec<i64>],
    device: &Device,
) -> LmRun {
    let lm = &construct.lm;
    let rows = prompt.rows.len();
    let mut cache = lm.decode_cache(1, rows + fed.len(), KVCacheStorage::Auto, device);
    let conditioning: Zonos3RowConditioning = prompt.conditioning(device);
    let host = |t: Tensor<4>| -> Vec<f32> {
        t.cast(FloatDType::F32)
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .unwrap()
    };
    let logits = lm.forward(prompt.ids(device), &conditioning, Some(&mut cache));
    let vocab = logits.dims()[3];
    let prefill = host(logits);
    let steps = fed
        .iter()
        .map(|row| {
            let ids = Tensor::<3, Int>::from_data(
                TensorData::new(row.clone(), [1, 1, row.len()]),
                device,
            );
            host(lm.forward(ids, &conditioning, Some(&mut cache)))
        })
        .collect();
    LmRun {
        prefill,
        rows,
        steps,
        vocab,
    }
}

/// How far bunsen's logits are from a reference file's, over one request.
#[derive(Clone, Debug, Default)]
pub struct LogitReport {
    /// The (row or step, codebook) argmaxes compared.
    pub argmaxes: usize,

    /// Those that differ.
    pub argmax_differ: usize,

    /// Those that differ where the reference's margin is at least the near
    /// tie: real disagreements.
    pub argmax_differ_decided: usize,

    /// The largest margin among the disagreements.
    pub worst_differing_margin: f32,

    /// Every top-8 logit's distance, bunsen's at the reference's ids.
    pub top8_diffs: Vec<f32>,

    /// Every sampled prefill logit's distance.
    pub sample_diffs: Vec<f32>,

    /// The prefill logits' sum and absolute sum, relative to the reference's
    /// absolute sum.
    pub sums_rel: [f64; 2],
}

impl LogitReport {
    /// The largest top-8 distance.
    pub fn top8_max(&self) -> f32 {
        self.top8_diffs.iter().copied().fold(0.0, f32::max)
    }

    /// The largest sampled distance.
    pub fn sample_max(&self) -> f32 {
        self.sample_diffs.iter().copied().fold(0.0, f32::max)
    }

    /// The `q`-quantile of the top-8 distances.
    pub fn top8_quantile(
        &self,
        q: f64,
    ) -> f32 {
        quantile(&self.top8_diffs, q)
    }

    /// The fraction of argmaxes that agree.
    pub fn agreement(&self) -> f64 {
        1.0 - self.argmax_differ as f64 / self.argmaxes as f64
    }
}

/// The `q`-quantile of `values`.
pub fn quantile(
    values: &[f32],
    q: f64,
) -> f32 {
    let mut v = values.to_vec();
    v.sort_by(f32::total_cmp);
    v.get(((v.len() as f64 - 1.0) * q).round() as usize)
        .copied()
        .unwrap_or(0.0)
}

/// Compares `run` with the reference file `expected`'s logits for the
/// request `name` (its `prefill.*` and `steps.*`), a disagreement counting
/// as decided where the reference's margin is at least `near_tie`.
pub fn compare_logits(
    run: &LmRun,
    expected: &FixtureFile,
    name: &str,
    stats_stride: usize,
    near_tie: f32,
) -> LogitReport {
    let at = |k: &str| format!("{name}.{k}");
    let v = run.vocab;
    let mut report = LogitReport::default();

    let mut argmaxes = |logits: &[f32], argmax: &[i64], margin: &[f32]| {
        for ((cb, &want), &m) in logits.chunks(v).zip(argmax).zip(margin) {
            let got = cb
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as i64;
            report.argmaxes += 1;
            if got != want {
                report.argmax_differ += 1;
                report.worst_differing_margin = report.worst_differing_margin.max(m);
                if m >= near_tie {
                    report.argmax_differ_decided += 1;
                }
            }
        }
    };
    argmaxes(
        &run.prefill,
        &expected.ints(&at("prefill.argmax")),
        &expected.floats(&at("prefill.margin")),
    );
    let step_argmax = expected.ints(&at("steps.argmax"));
    let step_margin = expected.floats(&at("steps.margin"));
    for (i, step) in run.steps.iter().enumerate() {
        let span = i * CODEBOOKS..(i + 1) * CODEBOOKS;
        argmaxes(step, &step_argmax[span.clone()], &step_margin[span]);
    }

    let mut top8 = |logits: &[f32], ids: &[i64], values: &[f32]| {
        for (c, (ids, values)) in ids.chunks(8).zip(values.chunks(8)).enumerate() {
            for (&id, &value) in ids.iter().zip(values) {
                report
                    .top8_diffs
                    .push((logits[c * v + id as usize] - value).abs());
            }
        }
    };
    let last = &run.prefill[(run.rows - 1) * CODEBOOKS * v..];
    top8(
        last,
        &expected.ints(&at("prefill.top8.ids")),
        &expected.floats(&at("prefill.top8.values")),
    );
    let step_ids = expected.ints(&at("steps.top8.ids"));
    let step_values = expected.floats(&at("steps.top8.values"));
    for (i, step) in run.steps.iter().enumerate() {
        let span = i * CODEBOOKS * 8..(i + 1) * CODEBOOKS * 8;
        top8(step, &step_ids[span.clone()], &step_values[span]);
    }

    let sample = expected.floats(&at("prefill.sample"));
    report.sample_diffs = run
        .prefill
        .iter()
        .step_by(stats_stride)
        .zip(&sample)
        .map(|(a, b)| (a - b).abs())
        .collect();
    assert_eq!(
        report.sample_diffs.len(),
        sample.len(),
        "{name}: the sample"
    );
    let sums = expected.doubles(&at("prefill.sums"));
    let sum: f64 = run.prefill.iter().map(|&x| f64::from(x)).sum();
    let abs_sum: f64 = run.prefill.iter().map(|&x| f64::from(x).abs()).sum();
    report.sums_rel = [
        (sum - sums[0]).abs() / sums[1],
        (abs_sum - sums[1]).abs() / sums[1],
    ];
    report
}

#[cfg(all(test, feature = "checkpoints"))]
mod prompts;

#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod logits;

#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod codec;
