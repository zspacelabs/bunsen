//! # nanochat cross-check against upstream nanochat
//!
//! Validates [`bunsen`]'s nanochat (`kits::gpts::nanochat`) against upstream
//! nanochat's own code (`karpathy/nanochat`), on the community checkpoints of
//! both of its models: `nanochat-students`' d20 base, mid and RL
//! (upstream's 2025 model) and `nitishpandey04`'s d18 base and SFT (its 2026
//! model). Every comparison runs bunsen through its
//! [`NanoChatDecodeState`](bunsen::kits::gpts::nanochat::NanoChatDecodeState),
//! under each [`KVCacheStorage`].
//!
//! ## This crate hosts nothing
//!
//! The checkpoints are upstream's own files (`model_<step>.pt`,
//! `meta_<step>.json`, the tokenizer), through bunsen's pinned rows, fetched
//! at run time into the weights cache by the tests that read them (feature
//! **`checkpoints`**, off by default, so a plain `cargo test` here reaches no
//! network and runs the fixture-integrity checks alone).
//!
//! `cargo validate` (`validation/tools/validate`) runs this crate under a
//! profile, which chooses the models (`validation/models.yml`). Every test of a
//! model has the model's id in its path (`greedy::nanochat_d20_rl`,
//! `chat::nanochat_d18_sft::tokenizer`, `bf16::nanochat_d20_base`):
//!
//! ```sh
//! cd validation && cargo validate run --profile medium-core --crate nanochat
//! ```
//!
//! `gpu-tests` compiles the tests that run a model on a backend, and a
//! backend feature says which; the bf16 checks are a test binary of their
//! own, `cuda_bf16`, which runs on `cuda`.
//!
//! ## The reference
//!
//! There is no independent port of nanochat to hold bunsen to, and need for
//! none: upstream's model is one file of `PyTorch`, run as it is.
//! `testdata/upstream.json` records, for each row, upstream's code at the
//! row's model's pinned commit (`a088b7a6`, `a3ca42a6`) on the row's own
//! files, on the CPU in float32: a prompt as upstream's tokenizer encodes it,
//! the prefill's last logits, and a greedy decode through upstream's KV
//! cache, with each step's margin. `testdata/upstream_bf16.json` records the
//! same on CUDA in bf16, as upstream serves there; on `--features cuda` the
//! model is also held in bf16, its bf16 tensors loaded as they are, and
//! checked against it. `validation/tools/gen_nanochat_fixtures.py` writes
//! both.

use std::collections::BTreeMap;

use bunsen::{
    kits::gpts::nanochat::NanoChatBundle,
    ops::transformers::attention::KVCacheStorage,
    support::testing::TensorRecord,
};
use burn::prelude::*;
use serde::Deserialize;

/// The rows the fixtures hold, in the order the tests run them.
pub const ROWS: [&str; 5] = [
    "nanochat-students/base-d20",
    "nanochat-students/mid-d20",
    "nanochat-students/rl-d20",
    "nitishpandey04/nanochat-d18-base",
    "nitishpandey04/nanochat-d18-sft",
];

/// The vocabulary size of a row.
pub fn n_vocab(row: &str) -> usize {
    if row.starts_with("nitishpandey04/") {
        32768
    } else {
        65536
    }
}

/// Every [`KVCacheStorage`], [`Auto`](KVCacheStorage::Auto) first.
pub const STORAGES: [KVCacheStorage; 4] = [
    KVCacheStorage::Auto,
    KVCacheStorage::Concat,
    KVCacheStorage::PerLayer,
    KVCacheStorage::MultiLayer,
];

/// The top-1 minus top-2 logit under which a greedy step is a near-tie: two
/// implementations may pick either token, and their decodes part there.
pub const NEAR_TIE: f64 = 0.02;

/// The bf16 counterpart of [`NEAR_TIE`]. Upstream's own bf16 and float32
/// prefill logits differ by up to 0.22 on these rows; two bf16
/// implementations can each be that far from float32, in opposite
/// directions, so a step whose margin is under twice it may go either way.
pub const BF16_NEAR_TIE: f64 = 0.5;

/// One row's upstream decode: upstream's code on the row's files.
#[derive(Debug, Deserialize)]
pub struct UpstreamRecord {
    /// The row and the upstream commit that decoded it.
    pub source: String,
    /// The prompt, after `<|bos|>`, no chat format.
    pub prompt: String,
    /// The prompt as the row's tokenizer encodes it, `<|bos|>` first.
    pub prompt_ids: Vec<i64>,
    /// The generated ids.
    pub ids: Vec<i64>,
    /// The generated text.
    pub text: String,
    /// Each step's top-1 minus top-2 logit.
    pub margins: Vec<f64>,
    /// The prefill's last-position logits, `[1, vocab]`.
    pub last_logits: TensorRecord,
}

/// `testdata/upstream_bf16.json`: the decodes in bf16 on CUDA.
#[derive(Debug, Deserialize)]
pub struct UpstreamBf16 {
    /// The `torch` that ran them.
    pub torch: String,
    /// By row.
    pub rows: BTreeMap<String, UpstreamRecord>,
}

/// Reads `testdata/upstream.json`, by row: the float32 decodes.
pub fn upstream_reference() -> BTreeMap<String, UpstreamRecord> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/upstream.json");
    let text = std::fs::read_to_string(path).expect("read the upstream fixture");
    serde_json::from_str(&text).expect("parse the upstream fixture")
}

/// Reads `testdata/upstream_bf16.json`: the bf16 decodes.
pub fn upstream_bf16_reference() -> UpstreamBf16 {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/upstream_bf16.json");
    let text = std::fs::read_to_string(path).expect("read the upstream bf16 fixture");
    serde_json::from_str(&text).expect("parse the upstream bf16 fixture")
}

/// One tokenizer's entry in `testdata/chat.json`: upstream's tokenizer on
/// fixed texts.
#[derive(Debug, Deserialize)]
pub struct ChatRecord {
    /// The token count, special tokens included.
    pub vocab_size: usize,
    /// The special tokens' ids, by name.
    pub specials: BTreeMap<String, i64>,
    /// Texts as upstream's `encode` gives them.
    pub encode: Vec<EncodeCase>,
    /// Conversations as upstream renders them for the assistant's next turn.
    pub conversations: Vec<ConversationCase>,
}

/// One conversation, rendered.
#[derive(Debug, Deserialize)]
pub struct ConversationCase {
    /// The conversation, its last turn a user's.
    pub messages: Vec<MessageCase>,
    /// Upstream's `render_for_completion` of it.
    pub ids: Vec<i64>,
}

/// One turn of a [`ConversationCase`].
#[derive(Debug, Deserialize)]
pub struct MessageCase {
    /// `system`, `user` or `assistant`.
    pub role: String,
    /// The turn's text.
    pub content: String,
}

/// One text, encoded.
#[derive(Debug, Deserialize)]
pub struct EncodeCase {
    /// The text.
    pub text: String,
    /// Upstream's ids for it.
    pub ids: Vec<i64>,
    /// Upstream's `decode` of them.
    pub decoded: String,
}

/// Reads `testdata/chat.json`, by tokenizer: `2025` (`nanochat-students`'
/// rows) and `2026` (`nitishpandey04`'s).
pub fn chat_reference() -> BTreeMap<String, ChatRecord> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/chat.json");
    let text = std::fs::read_to_string(path).expect("read the chat fixture");
    serde_json::from_str(&text).expect("parse the chat fixture")
}

/// The row whose tokenizer is `generation`'s: `2025` or `2026`.
pub fn tokenizer_row(generation: &str) -> &'static str {
    match generation {
        "2025" => "nanochat-students/rl-d20",
        "2026" => "nitishpandey04/nanochat-d18-sft",
        other => panic!("no tokenizer {other}"),
    }
}

/// The reply budget `testdata/chat_replies.json` was recorded with.
pub const REPLY_TOKENS: usize = 48;

/// One turn of a chat in `testdata/chat_replies.json`: upstream's greedy
/// reply to a user turn, the conversation before it rendered for the
/// assistant's next turn.
#[derive(Debug, Deserialize)]
pub struct ReplyRecord {
    /// The user's turn.
    pub user: String,
    /// The conversation, rendered.
    pub prompt_ids: Vec<i64>,
    /// The reply's ids, its stop token last when one ended it.
    pub ids: Vec<i64>,
    /// The reply's text, its stop token left out.
    pub text: String,
    /// Each step's top-1 minus top-2 logit.
    pub margins: Vec<f64>,
}

/// Reads `testdata/chat_replies.json`: by row, a list of chats, each a list
/// of turns.
pub fn chat_replies_reference() -> BTreeMap<String, Vec<Vec<ReplyRecord>>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/chat_replies.json");
    let text = std::fs::read_to_string(path).expect("read the chat replies fixture");
    serde_json::from_str(&text).expect("parse the chat replies fixture")
}

/// `row`'s file under `key`, fetched into [`weights_cache`] alone, without
/// the rest of the row.
pub fn row_file(
    row: &str,
    key: &str,
) -> std::path::PathBuf {
    use bunsen::kits::gpts::nanochat::pretrained::{
        NANOCHAT_KIT,
        default_nanochat_factory,
    };
    let (_, found) = default_nanochat_factory()
        .expect("the nanochat factory")
        .lookup(row)
        .unwrap_or_else(|e| panic!("{row}: {e}"));
    let mut map = bunsen::data::pretrained::ResourceMap::new(found.resources.name.clone());
    map.insert(found.resources.try_get(key).expect("the resource").clone());
    let loaded = weights_cache()
        .load(NANOCHAT_KIT, &map)
        .unwrap_or_else(|e| panic!("{row}: {e}"));
    loaded.expect(key).expect("the fetched file").to_path_buf()
}

/// The weights cache the tests load the checkpoints through: bunsen's own
/// (`BUNSEN_CACHE_DIR`, or the user's cache directory), which fetches a
/// pinned file the first time it is asked for; offline when
/// `BUNSEN_VALIDATE_OFFLINE` is set.
pub fn weights_cache() -> bunsen::data::pretrained::PretrainedCache {
    use bunsen::data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    };
    let offline = std::env::var_os("BUNSEN_VALIDATE_OFFLINE").is_some();
    PretrainedCache::new(PretrainedCacheOptions::default().with_offline(offline))
        .expect("open the pretrained cache")
}

/// A nanochat pretrained by name, through bunsen's factory and
/// [`weights_cache`].
pub fn load_bundle(
    name: &str,
    device: &Device,
) -> std::sync::Arc<NanoChatBundle> {
    bunsen::kits::gpts::nanochat::pretrained::default_nanochat_factory()
        .expect("the nanochat factory")
        .load_bundle(name, &weights_cache(), device)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// A bunsen cached decode: a prefill of `prompt`, then one step per token of
/// `fed`, through a cache of `storage` sized for them.
///
/// # Returns
/// The `[1, vocab]` logits after the prefill and after each step, one more
/// row than `fed` has tokens.
pub fn bunsen_decode(
    bundle: &NanoChatBundle,
    prompt: &[i64],
    fed: &[i64],
    storage: KVCacheStorage,
    device: &Device,
) -> Vec<Tensor<2>> {
    let model = &bundle.model;
    let total = prompt.len() + fed.len();
    let mut state = model.decode_state(1, total, storage);
    let ids = |ids: &[i64]| {
        Tensor::<2, Int>::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device)
    };
    let mut out = vec![model.prefill(ids(prompt), &mut state)];
    for &id in fed {
        out.push(model.prefill(ids(&[id]), &mut state));
    }
    assert_eq!(state.cache.pos(), total);
    out
}

/// The greedy token of `[1, vocab]` logits.
pub fn argmax(logits: &Tensor<2>) -> i64 {
    logits.clone().argmax(1).into_scalar::<i64>()
}

/// `bunsen_tool::chat`'s nanochat tokenizer against upstream's. Reads the
/// fetched tokenizer files; loads no model.
#[cfg(all(test, feature = "checkpoints"))]
mod chat;

/// `bunsen_tool::chat`'s sessions against upstream's greedy chat replies.
#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod chat_session;

/// bunsen's greedy decodes against upstream's, for every row, in float32.
#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod greedy;

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures are self-consistent: one record per row, every record's
    /// samples cover its shape, the shapes are the rows', and every row
    /// decoded its whole budget after a prompt that starts with `<|bos|>`.
    #[test]
    fn test_fixture_integrity() {
        let fp32 = upstream_reference();
        let bf16 = upstream_bf16_reference();
        let mut rows = ROWS.to_vec();
        rows.sort_unstable();
        for fixture in [&fp32, &bf16.rows] {
            assert_eq!(fixture.keys().map(String::as_str).collect::<Vec<_>>(), rows);
            for (row, record) in fixture {
                assert!(record.source.starts_with(row.as_str()), "{row}");
                assert_eq!(record.ids.len(), record.margins.len(), "{row}");
                assert_eq!(record.last_logits.shape, [1, n_vocab(row)], "{row}");
                assert_eq!(
                    record.last_logits.samples.len(),
                    record.last_logits.expected_samples(),
                    "{row}"
                );
                assert_eq!(record.prompt_ids, fp32[row].prompt_ids, "{row}");
            }
        }
        // `<|bos|>` is the first special token after each vocabulary's merges.
        assert_eq!(fp32["nanochat-students/rl-d20"].prompt_ids[0], 65527);
        assert_eq!(fp32["nitishpandey04/nanochat-d18-sft"].prompt_ids[0], 32759);
        assert!(
            bf16.torch.contains("cu"),
            "the bf16 fixture ran on CUDA: {}",
            bf16.torch
        );

        let replies = chat_replies_reference();
        assert_eq!(
            replies.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "nanochat-students/rl-d20",
                "nitishpandey04/nanochat-d18-sft"
            ]
        );
        for (row, chats) in &replies {
            for turns in chats {
                for turn in turns {
                    assert_eq!(turn.ids.len(), turn.margins.len(), "{row}");
                    assert!(turn.ids.len() <= REPLY_TOKENS, "{row}");
                    assert_eq!(
                        turn.prompt_ids[0], fp32[row].prompt_ids[0],
                        "{row}: <|bos|>"
                    );
                }
            }
        }

        let chat = chat_reference();
        assert_eq!(
            chat.keys().map(String::as_str).collect::<Vec<_>>(),
            ["2025", "2026"]
        );
        for (generation, record) in &chat {
            assert_eq!(
                record.vocab_size,
                n_vocab(tokenizer_row(generation)),
                "{generation}"
            );
            assert_eq!(record.specials.len(), 9, "{generation}");
            assert_eq!(
                record.specials["<|bos|>"],
                fp32[tokenizer_row(generation)].prompt_ids[0]
            );
            for case in &record.encode {
                assert_eq!(case.decoded, case.text, "{generation}: {:?}", case.text);
            }
        }
    }
}
