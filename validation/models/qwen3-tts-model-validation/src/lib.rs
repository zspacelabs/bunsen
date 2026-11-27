//! # Qwen3-TTS cross-check against `qwen_tts`.
//!
//! Validates [`bunsen`]'s Qwen3-TTS (`kits::speech::tts::qwen3_tts`) against
//! upstream's implementation, `qwen_tts` 0.1.1, over the pretrained checkpoint
//! (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, bunsen's row [`MODEL`]):
//!
//! - its tokenizer, built from the repository's slow files, against the
//!   processor's ids for the test texts and each request's chat-wrapped text
//!   (`tokenizer`, without the weights);
//! - its prompt, from those ids, and the talker's and code predictor's logits
//!   in float32 over a take dictated upstream's greedy codes ([`run_forced`]),
//!   against upstream's (`logits`);
//! - its greedy take, code for code (`take`);
//! - its codec's audio from fixed codes, through upstream's chunked decode,
//!   past one chunk too (`codec`);
//! - the TTS driver over the model: a sampled take, and Harvard sentences
//!   transcribed by Whisper within a word error rate (`driver`);
//! - the logits again in bf16 on CUDA (`tests/cuda_bf16.rs`).
//!
//! ## This crate hosts nothing
//!
//! The checkpoint comes through bunsen's own pinned row, fetched at run time
//! into the weights cache by the tests that read it (feature
//! **`checkpoints`**), which is off by default. No Qwen3-TTS file is
//! committed: the fixtures are ids, summaries of logits and embeddings, and
//! summaries of audio (`validation/tools/gen_qwen3_tts_fixtures.py` writes
//! them; [`fixtures`] reads them).
//!
//! `cargo validate` (`validation/tools/validate`) runs this crate under a
//! profile (`validation/models.yml`). Every test of the model has its id in
//! its path (`logits::qwen3_tts_12hz_0_6b_customvoice`):
//!
//! ```sh
//! cd validation && cargo validate run --profile medium-core --crate qwen3-tts
//! ```

#![cfg_attr(not(feature = "checkpoints"), allow(unused))]

pub mod fixtures;

use std::sync::Arc;

use bunsen::{
    data::formats::LoadPolicy,
    kits::speech::tts::qwen3_tts::{
        Qwen3TtsConstruct,
        driver::Qwen3TtsChooser,
        prompt::{
            Qwen3TtsPromptConfig,
            Qwen3TtsPromptRequest,
            Qwen3TtsPromptRows,
            Qwen3TtsTextPlacement,
        },
    },
};
use burn::{
    prelude::*,
    tensor::FloatDType,
};
use fixtures::{
    CODE_GROUPS,
    FixtureFile,
};

/// bunsen's row for Qwen3-TTS-12Hz-0.6B-CustomVoice.
pub const MODEL: &str = "qwen3-tts-12hz-0.6b-customvoice";

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

/// The model, in f32 at least: what the f32 checks hold to the reference's
/// f32.
#[cfg(feature = "checkpoints")]
pub fn load_construct(device: &Device) -> Arc<Qwen3TtsConstruct> {
    load_construct_with(device, LoadPolicy::min_precision(FloatDType::F32))
}

/// The model under `policy`.
#[cfg(feature = "checkpoints")]
pub fn load_construct_with(
    device: &Device,
    policy: LoadPolicy,
) -> Arc<Qwen3TtsConstruct> {
    bunsen::kits::speech::tts::qwen3_tts::pretrained::default_qwen3_tts_factory()
        .expect("the qwen3_tts factory")
        .load_construct(MODEL, &weights_cache(), device, policy)
        .unwrap_or_else(|e| panic!("load {MODEL}: {e}"))
}

/// The prompt request of the reference's request `name`: its ids, its
/// speaker's row, its language's (resolved by bunsen's `prompt`), its
/// placement.
pub fn reference_request(
    prompt: &Qwen3TtsPromptConfig,
    reference: &FixtureFile,
    name: &str,
) -> Qwen3TtsPromptRequest {
    let facts = reference.request(name);
    let speaker = prompt.speaker(&facts.speaker).expect("a stock speaker");
    Qwen3TtsPromptRequest {
        input_ids: reference
            .ints(&format!("{name}.input_ids"))
            .into_iter()
            .map(|id| id as u32)
            .collect(),
        instruct_ids: None,
        speaker: Some(speaker),
        language: prompt
            .language(&facts.language, Some(&facts.speaker))
            .expect("a language"),
        placement: if facts.non_streaming {
            Qwen3TtsTextPlacement::Upfront
        } else {
            Qwen3TtsTextPlacement::Interleaved
        },
    }
}

/// The reference take's codes for the request `name`, `[F][16]`.
pub fn reference_codes(
    reference: &FixtureFile,
    name: &str,
) -> Vec<Vec<u32>> {
    reference
        .ints(&format!("{name}.codes"))
        .chunks(CODE_GROUPS)
        .map(|frame| frame.iter().map(|&c| c as u32).collect())
        .collect()
}

/// What bunsen computed over a take dictated the reference's codes: the
/// prompt's rows, and the logits it chose from.
pub struct ForcedTake {
    /// Each prefill row's L2 norm.
    pub prompt_norms: Vec<f32>,

    /// The prefill rows, flattened.
    pub prompt: Vec<f32>,

    /// Each trailing text row's L2 norm.
    pub trailing_norms: Vec<f32>,

    /// The talker's logits at the prefill and each step, `[F + 1][V]`.
    pub talker: Vec<Vec<f32>>,

    /// The code predictor's logits, `[F][15][V_c]`, flattened by frame.
    pub predictor: Vec<Vec<Vec<f32>>>,
}

/// `t`'s values, in f32, on the host.
fn host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(FloatDType::F32)
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
}

/// Each row's L2 norm of `[1, L, W]`.
fn row_norms(t: Tensor<3>) -> Vec<f32> {
    host(t.cast(FloatDType::F32).powi_scalar(2).sum_dim(2).sqrt())
}

/// Dictates the codes; keeps the logits; ends with EOS once past them.
struct Script<'a> {
    codes: &'a [Vec<u32>],
    eos: i64,
    talker: Vec<Vec<f32>>,
    predictor: Vec<Vec<Vec<f32>>>,
}

impl Qwen3TtsChooser for Script<'_> {
    fn codebook0(
        &mut self,
        frame: usize,
        logits: Tensor<2>,
        _blocked: &[i64],
    ) -> i64 {
        self.talker.push(host(logits));
        match self.codes.get(frame) {
            Some(codes) => {
                self.predictor.push(Vec::new());
                i64::from(codes[0])
            }
            None => self.eos,
        }
    }

    fn codebook(
        &mut self,
        frame: usize,
        k: usize,
        logits: Tensor<2>,
    ) -> i64 {
        self.predictor[frame].push(host(logits));
        i64::from(self.codes[frame][k])
    }
}

/// Runs `construct` over `request`, dictated `codes`: the prompt, and the
/// logits of every choice, the talker's one past the last frame too (the
/// reference's last row).
pub fn run_forced(
    construct: &Qwen3TtsConstruct,
    request: &Qwen3TtsPromptRequest,
    codes: &[Vec<u32>],
    device: &Device,
) -> ForcedTake {
    let prompt = Qwen3TtsPromptRows::build(&construct.prompt, request)
        .expect("the prompt's rows")
        .embed(&construct.model.talker, device);
    let mut script = Script {
        codes,
        eos: i64::from(construct.prompt.codec_eos),
        talker: Vec::new(),
        predictor: Vec::new(),
    };
    let take =
        construct
            .model
            .run_take(&prompt, &construct.prompt, codes.len() + 1, 0, &mut script);
    assert_eq!(take.frames, codes, "the dictated take");
    ForcedTake {
        prompt_norms: row_norms(prompt.embeds.clone()),
        prompt: host(prompt.embeds),
        trailing_norms: row_norms(prompt.trailing),
        talker: script.talker,
        predictor: script.predictor,
    }
}

/// How far bunsen's logits are from the reference's, over one request.
#[derive(Clone, Debug, Default)]
pub struct LogitReport {
    /// The argmaxes compared.
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

    /// The largest distance of a row's sum and absolute sum, relative to its
    /// absolute sum.
    pub sums_rel: f64,
}

impl LogitReport {
    /// The largest top-8 distance.
    pub fn top8_max(&self) -> f32 {
        self.top8_diffs.iter().copied().fold(0.0, f32::max)
    }

    /// The `q`-quantile of the top-8 distances.
    pub fn top8_quantile(
        &self,
        q: f64,
    ) -> f32 {
        let mut v = self.top8_diffs.clone();
        v.sort_by(f32::total_cmp);
        v.get(((v.len() as f64 - 1.0) * q).round() as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// The fraction of argmaxes that agree.
    pub fn agreement(&self) -> f64 {
        1.0 - self.argmax_differ as f64 / self.argmaxes as f64
    }

    /// Adds `rows` against the reference's summaries under `prefix`
    /// (`<prefix>.argmax`, `.margin`, `.top8.ids`, `.top8.values`, `.sums`),
    /// one summary row per logits row.
    pub fn add(
        &mut self,
        rows: &[&[f32]],
        expected: &FixtureFile,
        prefix: &str,
        near_tie: f32,
    ) {
        let argmax = expected.ints(&format!("{prefix}.argmax"));
        let margin = expected.floats(&format!("{prefix}.margin"));
        let ids = expected.ints(&format!("{prefix}.top8.ids"));
        let values = expected.floats(&format!("{prefix}.top8.values"));
        let sums = expected.doubles(&format!("{prefix}.sums"));
        assert_eq!(rows.len(), argmax.len(), "{prefix}: the rows");
        for (r, row) in rows.iter().enumerate() {
            let got = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as i64;
            self.argmaxes += 1;
            if got != argmax[r] {
                self.argmax_differ += 1;
                self.worst_differing_margin = self.worst_differing_margin.max(margin[r]);
                if margin[r] >= near_tie {
                    self.argmax_differ_decided += 1;
                }
            }
            for j in 0..8 {
                let i = r * 8 + j;
                self.top8_diffs
                    .push((row[ids[i] as usize] - values[i]).abs());
            }
            let sum: f64 = row.iter().map(|&x| f64::from(x)).sum();
            let abs: f64 = row.iter().map(|&x| f64::from(x).abs()).sum();
            let [want, want_abs] = [sums[2 * r], sums[2 * r + 1]];
            self.sums_rel = self
                .sums_rel
                .max((sum - want).abs() / want_abs)
                .max((abs - want_abs).abs() / want_abs);
        }
    }
}

/// bunsen's talker and code predictor logits over `run` against the
/// reference's, for the request `name`.
pub fn compare_logits(
    run: &ForcedTake,
    expected: &FixtureFile,
    name: &str,
    near_tie: f32,
) -> [LogitReport; 2] {
    let mut talker = LogitReport::default();
    let rows: Vec<&[f32]> = run.talker.iter().map(Vec::as_slice).collect();
    talker.add(&rows, expected, &format!("{name}.talker"), near_tie);
    let mut predictor = LogitReport::default();
    let rows: Vec<&[f32]> = run.predictor.iter().flatten().map(Vec::as_slice).collect();
    predictor.add(&rows, expected, &format!("{name}.predictor"), near_tie);
    [talker, predictor]
}

#[cfg(all(test, feature = "checkpoints"))]
mod tokenizer;

#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod logits;

#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod take;

#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod codec;

#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod driver;
