//! # Qwen3 cross-check against pretrained references.
//!
//! Validates [`bunsen`]'s Qwen3 (`kits::gpts::qwen3`) against an independent
//! implementation, generated from `onnx-community/Qwen3-0.6B-ONNX` (an export
//! of `Qwen/Qwen3-0.6B`) by `burn_onnx::ModelGen`, and against
//! `transformers`' own greedy decodes of the 0.6B, 1.7B, 4B and
//! 4B-Instruct-2507 checkpoints. Every comparison runs bunsen through its
//! [`KVCache`](bunsen::ops::transformers::attention::KVCache), under each
//! [`KVCacheStorage`].
//!
//! ## This crate hosts nothing, and generates nothing
//!
//! The generated reference comes from `bunsen-bundled-qwen3`'s `onnx_gen`
//! feature, and the checkpoints from its `checkpoint` and
//! `large_checkpoints` features, which fetch their assets, pin each to a
//! SHA-256, and cache them, all under the **`download`** feature, off by
//! default so an ordinary `cargo build --workspace` never reaches the
//! network.
//!
//! ```sh
//! cargo test --release -p qwen3-model-validation \
//!   --features download,gpu-tests,wgpu
//! ```
//!
//! `gpu-tests` compiles every test that loads a model, `download` fetches
//! what they compare against, and a backend feature says what they run on.
//! Without both `download` and `gpu-tests` the suite is the fixture-integrity
//! checks, which load no model.
//!
//! ## The reference is itself checked
//!
//! The export uses onnxruntime-only ops that `burn-onnx` cannot import, so
//! `bunsen-bundled-qwen3` rewrites them to standard ONNX first.
//! `testdata/onnx_reference.json` records what onnxruntime computes for the
//! *published* graph on fixed ids, and `testdata/transformers.json` what
//! `transformers` decodes (`tools/gen_qwen3_fixtures.py` writes both).
//! `testdata/chat.json` records `transformers`' tokenizer and chat template
//! on fixed texts and conversations (`tools/gen_qwen3_chat_fixtures.py`),
//! which `bunsen-chat`'s are held to. The
//! generated model is held to the first, so neither a defect in the rewrite
//! nor one in the ONNX import can pass for a defect in bunsen.

#![cfg_attr(not(feature = "download"), allow(unused))]

use std::collections::BTreeMap;

use bunsen::{
    kits::gpts::qwen3::Qwen3Bundle,
    ops::transformers::attention::{
        KVCacheModel,
        KVCacheStorage,
    },
    support::testing::TensorRecord,
};
use burn::prelude::*;
use serde::Deserialize;

/// The reference model, generated from the rewritten export.
#[cfg(feature = "download")]
pub mod reference {
    pub use bunsen_bundled_qwen3::onnx_gen::*;
}

#[cfg(feature = "download")]
mod reference_decode;
#[cfg(feature = "download")]
pub use reference_decode::*;

/// `Qwen3-0.6B`'s layer count.
pub const N_LAYERS: usize = 28;

/// `Qwen3-0.6B`'s key/value heads.
pub const N_KV_HEADS: usize = 8;

/// `Qwen3-0.6B`'s head width.
pub const HEAD_DIM: usize = 128;

/// The vocabulary size of every Qwen3 checkpoint.
pub const N_VOCAB: usize = 151_936;

/// The rows `testdata/transformers.json` holds, in the order the tests run
/// them.
pub const ROWS: [&str; 4] = [
    "qwen3-0.6b",
    "qwen3-1.7b",
    "qwen3-4b",
    "qwen3-4b-instruct-2507",
];

/// Every [`KVCacheStorage`], [`Auto`](KVCacheStorage::Auto) first.
pub const STORAGES: [KVCacheStorage; 4] = [
    KVCacheStorage::Auto,
    KVCacheStorage::Concat,
    KVCacheStorage::PerLayer,
    KVCacheStorage::MultiLayer,
];

/// The positions a validation cache allocates up front, and grows by: small,
/// so that every decode here grows its cache several times.
pub const CACHE_CHUNK: usize = 8;

/// The top-1 minus top-2 logit under which a greedy step is a near-tie: two
/// float32 implementations may pick either token, and their decodes part
/// there.
pub const NEAR_TIE: f64 = 0.02;

/// `testdata/onnx_reference.json`: onnxruntime's run of the published graph.
#[derive(Debug, Deserialize)]
pub struct OnnxReference {
    /// The export and revision the fixture was made from.
    pub source: String,
    /// The prefill's ids.
    pub ids: Vec<i64>,
    /// `[1, ids, vocab]`.
    pub prefill_logits: TensorRecord,
    /// Layer 0's key cache after the prefill, `[1, 8, ids, 128]`.
    pub prefill_key_0: TensorRecord,
    /// Layer 27's value cache after the prefill.
    pub prefill_value_27: TensorRecord,
    /// The prefill's greedy next token, the step's input.
    pub step_id: i64,
    /// `[1, 1, vocab]`.
    pub step_logits: TensorRecord,
    /// Layer 27's key cache after the step, `[1, 8, ids + 1, 128]`.
    pub step_key_27: TensorRecord,
}

/// Reads `testdata/onnx_reference.json`.
pub fn onnx_reference() -> OnnxReference {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/onnx_reference.json");
    let text = std::fs::read_to_string(path).expect("read the onnx reference fixture");
    serde_json::from_str(&text).expect("parse the onnx reference fixture")
}

/// One row's entry in `testdata/transformers.json`: `transformers`' greedy
/// decode, through its own KV cache, in float32.
#[derive(Debug, Deserialize)]
pub struct GreedyRecord {
    /// The prompt, no chat template.
    pub prompt: String,
    /// The prompt as the row's tokenizer encodes it.
    pub prompt_ids: Vec<i64>,
    /// The generated ids.
    pub ids: Vec<i64>,
    /// The generated text.
    pub text: String,
    /// Each step's top-1 minus top-2 logit.
    pub margins: Vec<f64>,
    /// The prefill's last-position logits, `[1, vocab]`.
    pub last_logits: TensorRecord,
    /// How far the `lm_head.weight` the checkpoint carries is from its
    /// `embed_tokens`, where the fixture measured it: `Qwen3-0.6B`'s is a
    /// copy, which a tied loader may drop.
    #[serde(default)]
    pub lm_head_vs_embed_max_abs: Option<f64>,
}

/// Reads `testdata/transformers.json`, by row.
pub fn transformers_reference() -> BTreeMap<String, GreedyRecord> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/transformers.json");
    let text = std::fs::read_to_string(path).expect("read the transformers fixture");
    serde_json::from_str(&text).expect("parse the transformers fixture")
}

/// `testdata/chat.json`, by row: `transformers`' tokenizer and chat template
/// on fixed texts and conversations.
#[derive(Debug, Deserialize)]
pub struct ChatRecord {
    /// Texts, and the ids `transformers` encodes them to.
    pub encode: Vec<EncodeCase>,
    /// Conversations, under each setting, rendered and encoded.
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

/// One conversation under one setting.
#[derive(Debug, Deserialize)]
pub struct TemplateCase {
    /// The conversation's name.
    pub name: String,
    /// The conversation.
    pub messages: Vec<MessageCase>,
    /// Whether the assistant's next turn is opened.
    pub add_generation_prompt: bool,
    /// Qwen3's thinking switch; `None` when it was not passed.
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/chat.json");
    let text = std::fs::read_to_string(path).expect("read the chat fixture");
    serde_json::from_str(&text).expect("parse the chat fixture")
}

/// A pretrained cache over the bundled checkpoints, offline.
///
/// Without `download` there are none: an ordinary cache, which the tests
/// that need it are not compiled without.
pub fn weights_cache() -> bunsen::data::pretrained::PretrainedCache {
    use bunsen::data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    };
    #[cfg(feature = "download")]
    let options = PretrainedCacheOptions::default()
        .with_disk(
            bunsen::data::cache::BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(bunsen_bundled_qwen3::cache_dir().to_path_buf())),
        )
        .with_offline(true);
    #[cfg(not(feature = "download"))]
    let options = PretrainedCacheOptions::default();
    PretrainedCache::new(options).expect("open the pretrained cache")
}

/// A Qwen3 pretrained by name, through bunsen's factory and
/// [`weights_cache`].
pub fn load_bundle(
    name: &str,
    device: &Device,
) -> std::sync::Arc<Qwen3Bundle> {
    bunsen::kits::gpts::qwen3::pretrained::default_qwen3_factory()
        .expect("the qwen3 factory")
        .load_bundle(name, &weights_cache(), device)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// A bunsen cached decode: a prefill of `prompt`, then one step per token
/// of `fed`, through a cache of `storage` that starts at [`CACHE_CHUNK`]
/// positions and grows by as many.
///
/// # Returns
/// The `[1, vocab]` logits after the prefill and after each step, one more
/// row than `fed` has tokens.
pub fn bunsen_decode(
    bundle: &Qwen3Bundle,
    prompt: &[i64],
    fed: &[i64],
    storage: KVCacheStorage,
    device: &Device,
) -> Vec<Tensor<2>> {
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
    out
}

/// The greedy token of `[1, vocab]` logits.
pub fn argmax(logits: &Tensor<2>) -> i64 {
    logits.clone().argmax(1).into_scalar::<i64>()
}

/// The generated model against onnxruntime's run of the published graph.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod reference_tests;

/// bunsen's `Qwen3-0.6B` against onnxruntime, and against the generated
/// model over a long decode.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod cross;

/// `bunsen-chat`'s tokenizer and chat template against `transformers`'.
/// Reads the bundled tokenizer files; loads no model.
#[cfg(all(test, feature = "download"))]
mod chat;

/// bunsen's greedy decodes against `transformers`', for every row.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod greedy;

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures are self-consistent: every record's samples cover its
    /// shape, the shapes are the model's, and every row decoded its whole
    /// budget.
    #[test]
    fn test_fixture_integrity() {
        let fixture = onnx_reference();
        let seq = fixture.ids.len();
        for record in [
            &fixture.prefill_logits,
            &fixture.prefill_key_0,
            &fixture.prefill_value_27,
            &fixture.step_logits,
            &fixture.step_key_27,
        ] {
            assert_eq!(record.samples.len(), record.expected_samples());
        }
        assert_eq!(fixture.prefill_logits.shape, [1, seq, N_VOCAB]);
        assert_eq!(fixture.prefill_key_0.shape, [1, N_KV_HEADS, seq, HEAD_DIM]);
        assert_eq!(fixture.step_logits.shape, [1, 1, N_VOCAB]);
        assert_eq!(
            fixture.step_key_27.shape,
            [1, N_KV_HEADS, seq + 1, HEAD_DIM]
        );

        let greedy = transformers_reference();
        let mut rows = ROWS.to_vec();
        rows.sort_unstable();
        assert_eq!(greedy.keys().map(String::as_str).collect::<Vec<_>>(), rows);
        for (row, record) in &greedy {
            assert_eq!(record.ids.len(), record.margins.len(), "{row}");
            assert_eq!(record.last_logits.shape, [1, N_VOCAB], "{row}");
            assert_eq!(
                record.last_logits.samples.len(),
                record.last_logits.expected_samples(),
                "{row}"
            );
        }
        assert_eq!(greedy["qwen3-0.6b"].lm_head_vs_embed_max_abs, Some(0.0));

        let chat = chat_reference();
        assert_eq!(
            chat.keys().map(String::as_str).collect::<Vec<_>>(),
            ["qwen3-0.6b", "qwen3-4b-instruct-2507"]
        );
        for record in chat.values() {
            assert!(!record.encode.is_empty() && !record.templates.is_empty());
        }
    }
}
