//! # Gemma cross-check against `transformers`.
//!
//! Validates [`bunsen`]'s Gemma (`kits::gpts::gemma`) against `transformers`'
//! own greedy decodes, in float32, of `gemma-3-270m`, `gemma-3-1b-pt`,
//! `gemma-3-4b-pt` (its text model, run through
//! `Gemma3ForConditionalGeneration`; the smallest Gemma 3 with the linearly
//! scaled global table) and `gemma-2-2b` (Gemma 2's caps and alternating
//! layers, stored in float32), and in bf16 of `gemma-2-9b` (on `--features
//! cuda`, which runs bunsen in bf16 too: in float32 it does not fit a 24 GB
//! device). Every comparison runs bunsen through its
//! [`KVCache`](bunsen::ops::transformers::attention::KVCache), under each
//! [`KVCacheStorage`]; the window checks run past each row's sliding window,
//! where a sliding layer's cache keeps only its window.
//!
//! ## This crate hosts nothing
//!
//! The checkpoints come through bunsen's own pinned rows on the `unsloth/`
//! mirrors, fetched at run time into the weights cache by the tests that
//! read them (feature **`checkpoints`**), which is off by default. No
//! Gemma file is committed: the fixtures are ids, texts and logits
//! (`validation/tools/gen_gemma_fixtures.py` writes them):
//!
//! - `testdata/transformers.json`: the float32 greedy decodes;
//! - `testdata/transformers_window.json`: the float32 decodes after a prompt
//!   longer than the row's window;
//! - `testdata/transformers_bf16.json`: `gemma-2-9b`'s bf16 decode, and how far
//!   `transformers`' own bf16 prefill logits are from its float32 ones;
//! - `testdata/chat.json`: the tokenizer and chat template of `gemma-3-1b-it`
//!   and `gemma-2-2b-it` on fixed texts and conversations.
//!
//! `cargo validate` (`validation/tools/validate`) runs this crate under a
//! profile (`validation/models.yml`). Every test of a model has the model's
//! id in its path (`greedy::gemma_3_270m`), which is how a profile's filter
//! keeps or drops it:
//!
//! ```sh
//! cd validation && cargo validate run --profile small-core --crate gemma
//! ```

#![cfg_attr(not(feature = "checkpoints"), allow(unused))]

use std::collections::BTreeMap;

use bunsen::{
    kits::{
        gpts::gemma::GemmaConstruct,
        support::construct::LoadPolicy,
    },
    ops::transformers::attention::{
        KVCacheModel,
        KVCacheStorage,
    },
    support::testing::TensorRecord,
};
use burn::{
    prelude::*,
    tensor::FloatDType,
};
use serde::Deserialize;

/// The rows `testdata/transformers.json` and `transformers_window.json` hold,
/// in the order the tests run them.
pub const ROWS: [&str; 4] = [
    "gemma-3-270m",
    "gemma-3-1b-pt",
    "gemma-3-4b-pt",
    "gemma-2-2b",
];

/// The [`ROWS`] whose float32 checks run on the CPU: their embedding table,
/// in float32, is larger than the 2 GiB a wgpu buffer binding may hold
/// (`gemma-3-4b-pt`'s 262208 by 2560, 2.7 GB; `gemma-2-2b`'s 256000 by 2304,
/// 2.4 GB), and CUDA's float32 matmuls are not exact enough to hold
/// `transformers`' float32 logits to the same tolerance.
pub const CPU_ROWS: [&str; 2] = ["gemma-3-4b-pt", "gemma-2-2b"];

/// The device a float32 check of `row` runs on: the CPU for [`CPU_ROWS`],
/// else the performance device.
pub fn f32_device(row: &str) -> Device {
    if CPU_ROWS.contains(&row) {
        bunsen::support::testing::cpu_device()
    } else {
        bunsen::support::testing::performance_device()
    }
}

/// The rows held to `transformers` in bf16 on CUDA alone: those too large
/// for a float32 run on a 24 GB device.
pub const BF16_ROWS: [&str; 1] = ["gemma-2-9b"];

/// The rows `testdata/chat.json` holds.
pub const CHAT_ROWS: [&str; 2] = ["gemma-3-1b-it", "gemma-2-2b-it"];

/// Every [`KVCacheStorage`], [`Auto`](KVCacheStorage::Auto) first.
pub const STORAGES: [KVCacheStorage; 4] = [
    KVCacheStorage::Auto,
    KVCacheStorage::Concat,
    KVCacheStorage::PerLayer,
    KVCacheStorage::MultiLayer,
];

/// The positions a validation cache allocates up front, and grows by: small,
/// so that every decode here grows its cache several times.
pub const CACHE_CHUNK: usize = 64;

/// The top-1 minus top-2 logit under which a greedy step is a near-tie: two
/// float32 implementations may pick either token there.
pub const NEAR_TIE: f64 = 0.02;

/// The bf16 counterpart of [`NEAR_TIE`]: two bf16 implementations can each
/// be bf16's own error from float32, in opposite directions.
pub const BF16_NEAR_TIE: f64 = 0.5;

/// One greedy decode, through `transformers`' own KV cache.
#[derive(Debug, Deserialize)]
pub struct GreedyRecord {
    /// The prompt's ids, `<bos>` first.
    pub prompt_ids: Vec<i64>,
    /// The generated ids.
    pub ids: Vec<i64>,
    /// The generated text.
    pub text: String,
    /// Each step's top-1 minus top-2 logit.
    pub margins: Vec<f64>,
    /// The prefill's last-position logits, `[1, vocab]`.
    pub last_logits: TensorRecord,
    /// The row's sliding window, for a window record.
    #[serde(default)]
    pub window: Option<usize>,
    /// For a bf16 record, how far `transformers`' float32 prefill logits are
    /// from its bf16 ones: bf16's own error.
    #[serde(default)]
    pub f32_last_logits_max_abs: Option<f64>,
}

fn read<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = format!("{}/testdata/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Reads `testdata/transformers.json`, by row.
pub fn transformers_reference() -> BTreeMap<String, GreedyRecord> {
    read("transformers.json")
}

/// Reads `testdata/transformers_window.json`, by row.
pub fn transformers_window_reference() -> BTreeMap<String, GreedyRecord> {
    read("transformers_window.json")
}

/// Reads `testdata/transformers_bf16.json`, by row.
pub fn transformers_bf16_reference() -> BTreeMap<String, GreedyRecord> {
    read("transformers_bf16.json")
}

/// One row of `testdata/chat.json`.
#[derive(Debug, Deserialize)]
pub struct ChatRecord {
    /// Texts, and the ids `transformers` encodes them to.
    pub encode: Vec<EncodeCase>,
    /// Conversations rendered and encoded.
    pub templates: Vec<TemplateCase>,
}

/// One text and its ids.
#[derive(Debug, Deserialize)]
pub struct EncodeCase {
    /// The text.
    pub text: String,
    /// Its ids, no special tokens added.
    pub ids: Vec<i64>,
}

/// One conversation, rendered.
#[derive(Debug, Deserialize)]
pub struct TemplateCase {
    /// The conversation's name.
    pub name: String,
    /// The conversation.
    pub messages: Vec<MessageCase>,
    /// Whether the assistant's next turn is opened.
    pub add_generation_prompt: bool,
    /// What `apply_chat_template` rendered.
    pub text: String,
    /// The text's ids.
    pub ids: Vec<i64>,
}

/// One turn of a [`TemplateCase`].
#[derive(Debug, Deserialize)]
pub struct MessageCase {
    /// The role.
    pub role: String,
    /// The content.
    pub content: String,
}

/// Reads `testdata/chat.json`, by row.
pub fn chat_reference() -> BTreeMap<String, ChatRecord> {
    read("chat.json")
}

/// The weights cache the tests load the checkpoints through: bunsen's own,
/// offline when `BUNSEN_VALIDATE_OFFLINE` is set, for a run on a warm cache.
pub fn weights_cache() -> bunsen::data::pretrained::PretrainedCache {
    use bunsen::data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    };
    let offline = std::env::var_os("BUNSEN_VALIDATE_OFFLINE").is_some();
    PretrainedCache::new(PretrainedCacheOptions::default().with_offline(offline))
        .expect("open the pretrained cache")
}

/// A Gemma pretrained by name, in f32 at least: what the f32 suites hold to
/// `transformers` in f32.
pub fn load_construct(
    name: &str,
    device: &Device,
) -> std::sync::Arc<GemmaConstruct> {
    load_construct_with(name, device, LoadPolicy::at_least(FloatDType::F32))
}

/// A Gemma pretrained by name under `policy`.
pub fn load_construct_with(
    name: &str,
    device: &Device,
    policy: LoadPolicy,
) -> std::sync::Arc<GemmaConstruct> {
    bunsen::kits::gpts::gemma::pretrained::default_gemma_factory()
        .expect("the gemma factory")
        .load_construct(name, &weights_cache(), device, policy)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// A bunsen cached decode: a prefill of `prompt`, then one step per token
/// of `fed`, through the model's cache (its sliding layers keeping only
/// their window) under `storage`, starting at [`CACHE_CHUNK`] positions and
/// growing by as many.
///
/// # Returns
/// The `[1, vocab]` logits after the prefill and after each step, one more
/// row than `fed` has tokens; and the earliest position the cache still
/// holds for every layer.
pub fn bunsen_decode(
    bundle: &GemmaConstruct,
    prompt: &[i64],
    fed: &[i64],
    storage: KVCacheStorage,
    device: &Device,
) -> (Vec<Tensor<2>>, usize) {
    let model = &bundle.model;
    let total = prompt.len() + fed.len();
    let r_emb = model.rotary(total, device);
    let mut cache = model
        .kv_cache_config()
        .with_seq_len(CACHE_CHUNK)
        .with_chunk_size(CACHE_CHUNK)
        .with_storage(storage)
        .init();
    let last = |ids: &[i64], cache: &mut _| {
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device);
        let hidden = model.forward_hidden(model.embed(tokens), &r_emb, Some(cache));
        let n = hidden.dims()[1];
        model
            .logits(hidden.slice_dim(1, n - 1..n))
            .squeeze_dim::<2>(1)
    };
    let mut out = vec![last(prompt, &mut cache)];
    for &id in fed {
        out.push(last(&[id], &mut cache));
    }
    assert_eq!(cache.pos(), total);
    (out, cache.retained_from())
}

/// The greedy token of `[1, vocab]` logits.
pub fn argmax(logits: &Tensor<2>) -> i64 {
    logits.clone().argmax(1).into_scalar::<i64>()
}

/// Holds `steps`, bunsen's decode fed `record`'s tokens, to it: the
/// prefill's logits within `(rel, abs)` of `transformers`', and each step's
/// argmax its token, but where `transformers`' margin is under `near_tie`.
/// Returns how many near-ties parted them.
pub fn check_decode(
    what: &str,
    record: &GreedyRecord,
    steps: &[Tensor<2>],
    tolerance: (f64, f64),
    near_tie: f64,
) -> usize {
    let error = record.last_logits.assert_matches(
        &format!("{what} prefill logits"),
        steps[0].clone(),
        tolerance.0,
        tolerance.1,
    );
    let mut ties = 0;
    for (k, logits) in steps.iter().enumerate() {
        let got = argmax(logits);
        if got != record.ids[k] {
            assert!(
                record.margins[k] < near_tie,
                "{what}, step {k}: {got}, transformers {} by a margin of {}",
                record.ids[k],
                record.margins[k]
            );
            ties += 1;
        }
    }
    eprintln!(
        "{what}: prefill logits within {error}; {} of {} tokens agree ({ties} near-ties): {:?}",
        steps.len() - ties,
        steps.len(),
        record.text
    );
    ties
}

/// Each row's prefab against its checkpoint's `config.json`. Fetches the
/// configs alone.
#[cfg(all(test, feature = "checkpoints"))]
mod prefabs;

/// `bunsen_tool::chat`'s tokenizer and chat template against `transformers`'.
/// Fetches each row's tokenizer files alone.
#[cfg(all(test, feature = "checkpoints"))]
mod chat;

/// bunsen's float32 greedy decodes against `transformers`', for every row.
#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod greedy;

/// bunsen's float32 decodes past each row's sliding window against
/// `transformers`'.
#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod window;

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures are self-consistent: every record's samples cover its
    /// shape, every decode ran its whole budget, and each window prompt is
    /// longer than its window.
    #[test]
    fn test_fixture_integrity() {
        let mut rows = ROWS.to_vec();
        rows.sort_unstable();
        for (fixture, window) in [
            (transformers_reference(), false),
            (transformers_window_reference(), true),
        ] {
            assert_eq!(fixture.keys().map(String::as_str).collect::<Vec<_>>(), rows);
            for (row, record) in &fixture {
                assert_eq!(record.ids.len(), record.margins.len(), "{row}");
                assert_eq!(record.prompt_ids[0], 2, "{row}: <bos> first");
                assert_eq!(
                    record.last_logits.samples.len(),
                    record.last_logits.expected_samples(),
                    "{row}"
                );
                if window {
                    let w = record.window.expect("a window");
                    assert!(record.prompt_ids.len() > w, "{row}");
                }
            }
        }
        let bf16 = transformers_bf16_reference();
        assert_eq!(
            bf16.keys().map(String::as_str).collect::<Vec<_>>(),
            BF16_ROWS
        );
        for (row, record) in &bf16 {
            let gap = record.f32_last_logits_max_abs.expect("bf16's own error");
            assert!(gap < BF16_NEAR_TIE * 4.0, "{row}: {gap}");
        }
        let chat = chat_reference();
        let mut rows = CHAT_ROWS.to_vec();
        rows.sort_unstable();
        assert_eq!(chat.keys().map(String::as_str).collect::<Vec<_>>(), rows);
        for record in chat.values() {
            assert!(!record.encode.is_empty() && !record.templates.is_empty());
        }
    }
}
