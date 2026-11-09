//! # Qwen3.5 cross-check against `transformers`.
//!
//! Validates [`bunsen`]'s dense Qwen3.5 (`kits::language::qwen3_5`) against
//! `transformers`' own greedy decodes, in float32, of `qwen3.5-0.8b` (equal
//! linear key and value heads, a tied head), `qwen3.5-2b` (wider) and
//! `qwen3.5-4b` (twice the value heads), and in bf16 of `qwen3.5-9b` (on
//! `--features cuda`, which runs bunsen in bf16 too: in float32 it does not fit
//! a 24 GB device; its head is its own). Every comparison runs bunsen through
//! its decode state, a KV cache for the full-attention layers under each
//! [`KVCacheStorage`] and a state for each linear one: a prefill, which runs
//! the delta rule a chunk at a time, then one step per token. The chunk checks
//! run a prompt of several chunks first.
//!
//! ## This crate hosts nothing
//!
//! The checkpoints come through bunsen's own pinned rows, fetched at run time
//! into the weights cache by the tests that read them (feature
//! **`checkpoints`**), which is off by default. No Qwen3.5 file is committed:
//! the fixtures are ids, texts and logits
//! (`validation/tools/gen_qwen3_5_fixtures.py` writes them):
//!
//! - `testdata/transformers.json`: the float32 greedy decodes, of [`ROWS`];
//! - `testdata/transformers_chunks.json`: the float32 decodes after a prompt of
//!   several of the delta rule's chunks;
//! - `testdata/transformers_bf16.json`: `qwen3.5-9b`'s bf16 decode, and how far
//!   `transformers`' own bf16 prefill logits are from its float32 ones;
//! - `testdata/chat.json`: the tokenizers and chat templates of [`CHAT_ROWS`]
//!   on fixed texts and conversations
//!   (`validation/tools/gen_qwen3_5_chat_fixtures.py`);
//! - `testdata/chat_replies.json`: `qwen3.5-0.8b`'s greedy replies in
//!   multi-turn sessions, which `bunsen_tool::chat`'s sessions are held to.
//!
//! `cargo validate` (`validation/tools/validate`) runs this crate under a
//! profile (`validation/models.yml`). Every test of a model has the model's
//! id in its path (`greedy::qwen3_5_0_8b`), which is how a profile's filter
//! keeps or drops it:
//!
//! ```sh
//! cd validation && cargo validate run --profile medium-core --crate qwen3-5
//! ```

#![cfg_attr(not(feature = "checkpoints"), allow(unused))]

use std::collections::BTreeMap;

use bunsen::{
    blocks::transformers::models::language_decoder::LmDecoderOps,
    kits::{
        language::qwen3_5::Qwen3_5Construct,
        support::{
            construct::LoadPolicy,
            decoding::{
                CachedStep,
                StackedLm,
            },
        },
    },
    ops::transformers::attention::KVCacheStorage,
    support::testing::TensorRecord,
};
use burn::{
    prelude::*,
    tensor::FloatDType,
};
use serde::Deserialize;

/// The rows `testdata/transformers.json` and `transformers_chunks.json` hold,
/// in the order the tests run them.
pub const ROWS: [&str; 3] = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"];

/// The [`ROWS`] whose float32 checks run on the CPU: their embedding table,
/// in float32, is larger than the 2 GiB a wgpu buffer binding may hold
/// (`qwen3.5-4b`'s 248320 by 2560, 2.5 GB), and CUDA's float32 matmuls (TF32
/// on its tensor cores) are not exact enough to hold `transformers`' float32
/// logits to the same tolerance.
pub const CPU_ROWS: [&str; 1] = ["qwen3.5-4b"];

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
pub const BF16_ROWS: [&str; 1] = ["qwen3.5-9b"];

/// The fewest positions the chunk checks' prompt has: several of the delta
/// rule's chunks of 64.
pub const CHUNKS_MIN_POSITIONS: usize = 200;

/// Every [`KVCacheStorage`], [`Auto`](KVCacheStorage::Auto) first.
pub const STORAGES: [KVCacheStorage; 4] = [
    KVCacheStorage::Auto,
    KVCacheStorage::Concat,
    KVCacheStorage::PerLayer,
    KVCacheStorage::MultiLayer,
];

/// The top-1 minus top-2 logit under which a greedy step is a near-tie: two
/// float32 implementations may pick either token there.
pub const NEAR_TIE: f64 = 0.02;

/// The bf16 counterpart of [`NEAR_TIE`]: two bf16 implementations can each
/// be bf16's own error from float32, in opposite directions.
pub const BF16_NEAR_TIE: f64 = 0.5;

/// One greedy decode, through `transformers`' own cache.
#[derive(Debug, Deserialize)]
pub struct GreedyRecord {
    /// The prompt's ids.
    pub prompt_ids: Vec<i64>,
    /// The generated ids.
    pub ids: Vec<i64>,
    /// The generated text.
    pub text: String,
    /// Each step's top-1 minus top-2 logit.
    pub margins: Vec<f64>,
    /// The prefill's last-position logits, `[1, vocab]`.
    pub last_logits: TensorRecord,
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

/// Reads `testdata/transformers_chunks.json`, by row.
pub fn transformers_chunks_reference() -> BTreeMap<String, GreedyRecord> {
    read("transformers_chunks.json")
}

/// Reads `testdata/transformers_bf16.json`, by row.
pub fn transformers_bf16_reference() -> BTreeMap<String, GreedyRecord> {
    read("transformers_bf16.json")
}

/// The rows `testdata/chat.json` holds: one of each chat template (the
/// 0.8B's, which the 2B shares, and the 4B's, which the 9B and 27B share),
/// and the `-base` rows' tokenizer.
pub const CHAT_ROWS: [&str; 3] = ["qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-0.8b-base"];

/// The rows `testdata/chat_replies.json` holds.
pub const SESSION_ROWS: [&str; 1] = ["qwen3.5-0.8b"];

/// One row of `testdata/chat.json`.
#[derive(Debug, Deserialize)]
pub struct ChatRecord {
    /// Texts, and the ids `transformers` encodes them to.
    pub encode: Vec<EncodeCase>,
    /// Conversations rendered and encoded; none for a `-base` row.
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
    /// The thinking switch; `None` when it was not passed.
    pub enable_thinking: Option<bool>,
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
    /// An assistant's thinking, kept apart.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

/// Reads `testdata/chat.json`, by row.
pub fn chat_reference() -> BTreeMap<String, ChatRecord> {
    read("chat.json")
}

/// One session of `testdata/chat_replies.json`.
#[derive(Debug, Deserialize)]
pub struct SessionRecord {
    /// The session's name.
    pub name: String,
    /// The thinking switch; `None` when it was not passed.
    pub enable_thinking: Option<bool>,
    /// The most tokens a reply generates.
    pub budget: usize,
    /// Each turn and `transformers`' reply.
    pub replies: Vec<ReplyRecord>,
}

/// One turn of a [`SessionRecord`].
#[derive(Debug, Deserialize)]
pub struct ReplyRecord {
    /// The user's message.
    pub user: String,
    /// The rendered conversation's ids.
    pub prompt_ids: Vec<i64>,
    /// The reply's ids, its stop token included.
    pub ids: Vec<i64>,
    /// The reply's text, without it.
    pub text: String,
    /// Each step's top-1 minus top-2 logit.
    pub margins: Vec<f64>,
}

/// Reads `testdata/chat_replies.json`, by row.
pub fn chat_replies_reference() -> BTreeMap<String, Vec<SessionRecord>> {
    read("chat_replies.json")
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

/// A Qwen3.5 pretrained by name, in f32 at least: what the f32 suites hold
/// to `transformers` in f32.
pub fn load_construct(
    name: &str,
    device: &Device,
) -> std::sync::Arc<Qwen3_5Construct> {
    load_construct_with(name, device, LoadPolicy::at_least(FloatDType::F32))
}

/// A Qwen3.5 pretrained by name under `policy`.
pub fn load_construct_with(
    name: &str,
    device: &Device,
    policy: LoadPolicy,
) -> std::sync::Arc<Qwen3_5Construct> {
    bunsen::kits::language::qwen3_5::pretrained::default_qwen3_5_factory()
        .expect("the qwen3.5 factory")
        .load_construct(name, &weights_cache(), device, policy)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// A bunsen cached decode: a prefill of `prompt`, then one step per token of
/// `fed`, through the model's decode state (the full-attention layers' KV
/// cache under `storage`, and the linear layers' states).
///
/// # Returns
/// The `[1, vocab]` logits after the prefill and after each step, one more
/// row than `fed` has tokens.
pub fn bunsen_decode(
    bundle: &Qwen3_5Construct,
    prompt: &[i64],
    fed: &[i64],
    storage: KVCacheStorage,
    device: &Device,
) -> Vec<Tensor<2>> {
    let model = &bundle.model;
    let total = prompt.len() + fed.len();
    let mut state = model.decode_state(1, total, storage, device);
    let tokens = |ids: &[i64]| {
        Tensor::<2, Int>::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device)
    };
    let mut out = vec![model.prefill(model.embed(tokens(prompt)), &mut state)];
    for &id in fed {
        out.push(model.step(tokens(&[id]), &mut state));
    }
    assert_eq!(state.cache.pos(), total);
    out
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

/// `bunsen_tool::chat`'s sessions against `transformers`' greedy replies.
#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod chat_session;

/// bunsen's float32 greedy decodes against `transformers`', for every row.
#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod greedy;

/// bunsen's float32 decodes after a prompt of several chunks against
/// `transformers`'.
#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod chunks;

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures are self-consistent: every record's samples cover its
    /// shape, every decode ran its whole budget, and each chunk prompt spans
    /// several chunks and ends inside one.
    #[test]
    fn test_fixture_integrity() {
        for (fixture, chunks) in [
            (transformers_reference(), false),
            (transformers_chunks_reference(), true),
        ] {
            let mut rows = ROWS.to_vec();
            rows.sort_unstable();
            assert_eq!(fixture.keys().map(String::as_str).collect::<Vec<_>>(), rows);
            for (row, record) in &fixture {
                assert_eq!(record.ids.len(), record.margins.len(), "{row}");
                assert_eq!(
                    record.last_logits.samples.len(),
                    record.last_logits.expected_samples(),
                    "{row}"
                );
                if chunks {
                    let n = record.prompt_ids.len();
                    assert!(n >= CHUNKS_MIN_POSITIONS && n % 64 != 0, "{row}: {n}");
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
        for (row, record) in &chat {
            assert!(!record.encode.is_empty(), "{row}");
            assert_eq!(record.templates.is_empty(), row.ends_with("-base"), "{row}");
        }
        let replies = chat_replies_reference();
        assert_eq!(
            replies.keys().map(String::as_str).collect::<Vec<_>>(),
            SESSION_ROWS
        );
        for (row, sessions) in &replies {
            for session in sessions {
                for reply in &session.replies {
                    assert_eq!(reply.ids.len(), reply.margins.len(), "{row}");
                    assert!(reply.ids.len() <= session.budget, "{row}");
                }
            }
        }
    }
}
