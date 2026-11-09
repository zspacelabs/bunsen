//! # Toy tokenizers, for tests
//!
//! Small tokenizer files in the shapes real checkpoints ship, for tests of
//! code that reads text through a
//! [`Tokenizer`](crate::kits::support::tokens::Tokenizer). Each is the file's
//! contents, for `BunsenTokenizer::from_json` or
//! `BunsenTokenizer::from_tiktoken_pickle` (feature `tokenizer`):
//!
//! - [`toy_sentencepiece_json`]: a `tokenizer.json` as Gemma's is built.
//! - [`toy_byte_level_json`]: a `tokenizer.json` as Qwen3's is built.
//! - `toy_tiktoken_pickle` (feature `tokenizer`): a pickled `tiktoken`
//!   `Encoding`, as nanochat's `tokenizer.pkl` is.

use serde_json::json;

use crate::kits::support::tokens::byte_level::byte_alphabet;

/// A toy SentencePiece-style tokenizer, as Gemma's `tokenizer.json` is
/// built: spaces as `▁`, byte fallback, `<pad>` 0, `<eos>` 1, `<bos>` 2 and
/// `<unk>` 3, and two special tokens after the pieces (`<start_of_turn>`,
/// `<end_of_turn>`).
pub fn toy_sentencepiece_json() -> String {
    let mut vocab: Vec<String> = ["<pad>", "<eos>", "<bos>", "<unk>"]
        .iter()
        .map(ToString::to_string)
        .collect();
    for b in 0..=255u8 {
        vocab.push(format!("<0x{b:02X}>"));
    }
    for piece in [
        "▁", "h", "i", "▁hi", "▁there", "t", "e", "r", "o", "▁to", "model", "user",
    ] {
        vocab.push(piece.to_string());
    }
    let n = vocab.len();
    let merges: Vec<String> = Vec::new();
    let vocab_map: serde_json::Map<String, serde_json::Value> = vocab
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), json!(i)))
        .collect();
    json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": n, "content": "<start_of_turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": n + 1, "content": "<end_of_turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"}
        ]},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
            "vocab": vocab_map, "merges": merges
        }
    })
    .to_string()
}

/// A toy byte-level tokenizer, as Qwen3's `tokenizer.json` is built: GPT-2's
/// byte alphabet, a few merges (to `hello`, 259, and ` w`), an NFC
/// normalizer, Qwen's `Split` then `ByteLevel` pre-tokenizer, unused pieces up
/// to 300, and two special tokens (`<|im_start|>` 300, `<|im_end|>` 301).
pub fn toy_byte_level_json() -> String {
    // GPT-2's byte alphabet: each byte's character, its id the byte.
    let alphabet = byte_alphabet();
    let mut vocab: serde_json::Map<String, serde_json::Value> = alphabet
        .iter()
        .enumerate()
        .map(|(b, c)| (c.to_string(), json!(b)))
        .collect();
    let space = alphabet[b' ' as usize];
    let merges = [
        ("h".to_string(), "e".to_string()),
        ("l".to_string(), "l".to_string()),
        ("he".to_string(), "ll".to_string()),
        ("hell".to_string(), "o".to_string()),
        (space.to_string(), "w".to_string()),
    ];
    for (i, (a, b)) in merges.iter().enumerate() {
        vocab.insert(format!("{a}{b}"), json!(256 + i));
    }
    // Unused pieces up to the added tokens, which follow without a gap.
    for id in vocab.len()..300 {
        vocab.insert(format!("<unused{id}>"), json!(id));
    }
    let merges: Vec<String> = merges.iter().map(|(a, b)| format!("{a} {b}")).collect();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 300, "content": "<|im_start|>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false},
            {"id": 301, "content": "<|im_end|>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": merges
        }
    })
    .to_string()
}

/// A pickle as `tiktoken` writes an `Encoding`: `STACK_GLOBAL
/// tiktoken.core Encoding`, `NEWOBJ` on no arguments, then `BUILD` with
/// its state; the 256 bytes, the merges to `hello` (`he` 256, `ll` 257,
/// `hell` 258, `hello` 259), and nanochat's five layout tokens (`<|bos|>`
/// 260, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`,
/// `<|assistant_end|>` 264).
#[cfg(feature = "tokenizer")]
pub fn toy_tiktoken_pickle() -> Vec<u8> {
    use alloc::collections::BTreeMap;

    use serde_pickle::{
        HashableValue,
        Value,
    };

    let mut ranks = BTreeMap::new();
    for b in 0..=255u8 {
        ranks.insert(HashableValue::Bytes(vec![b]), Value::I64(b as i64));
    }
    for (rank, piece) in ["he", "ll", "hell", "hello"].iter().enumerate() {
        ranks.insert(
            HashableValue::Bytes(piece.as_bytes().to_vec()),
            Value::I64(256 + rank as i64),
        );
    }
    let mut specials = BTreeMap::new();
    for (i, name) in [
        "<|bos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
    ]
    .iter()
    .enumerate()
    {
        specials.insert(
            HashableValue::String(name.to_string()),
            Value::I64(260 + i as i64),
        );
    }
    let mut state = BTreeMap::new();
    state.insert(
        HashableValue::String("name".into()),
        Value::String("toy".into()),
    );
    state.insert(
        HashableValue::String("pat_str".into()),
        Value::String(r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+".into()),
    );
    state.insert(
        HashableValue::String("mergeable_ranks".into()),
        Value::Dict(ranks),
    );
    state.insert(
        HashableValue::String("special_tokens".into()),
        Value::Dict(specials),
    );
    let state = serde_pickle::value_to_vec(&Value::Dict(state), Default::default())
        .expect("a pickle of plain values");
    // `PROTO 2`, the class, `NEWOBJ` on `()`, the state (its own `PROTO`
    // and `STOP` stripped), `BUILD`, `STOP`.
    let mut out = vec![0x80, 2];
    out.extend_from_slice(b"ctiktoken.core\nEncoding\n");
    out.extend_from_slice(b")\x81");
    out.extend_from_slice(&state[2..state.len() - 1]);
    out.extend_from_slice(b"b.");
    out
}
