//! A pickled `tiktoken` `Encoding`, as an [`HfTokenizer`].
use std::{
    collections::{
        BTreeMap,
        BTreeSet,
    },
    path::Path,
};

use bunsen::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};
use serde_json::json;
use serde_pickle::{
    DeOptions,
    HashableValue,
    Value,
};

use crate::chat::HfTokenizer;

/// `key` of the pickled `Encoding`'s state, or an `InvalidResource` naming
/// it.
fn entry<'a>(
    state: &'a BTreeMap<HashableValue, Value>,
    key: &str,
) -> BunsenResult<&'a Value> {
    state
        .get(&HashableValue::String(key.to_string()))
        .ok_or_else(|| BunsenError::invalid_resource(format!("tokenizer.pkl: no `{key}`")))
}

/// An id, from a pickled int.
fn id_of(value: &Value) -> BunsenResult<u32> {
    match value {
        Value::I64(id) if *id >= 0 && *id <= u32::MAX as i64 => Ok(*id as u32),
        other => Err(BunsenError::invalid_resource(format!(
            "tokenizer.pkl: a token id that is not one: {other:?}"
        ))),
    }
}

/// GPT-2's byte alphabet, as a byte-level `tokenizer.json` spells bytes:
/// each printable byte its own character, the others the characters from
/// U+0100 on, in byte order.
pub(crate) fn byte_alphabet() -> [char; 256] {
    let mut alphabet = ['\0'; 256];
    let mut next = 0x100;
    for b in 0..=255u8 {
        let printable = (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
        alphabet[b as usize] = if printable {
            char::from(b)
        } else {
            let c = char::from_u32(next).expect("a character below U+0144");
            next += 1;
            c
        };
    }
    alphabet
}

impl HfTokenizer {
    /// Reads a pickled `tiktoken` `Encoding` file: a nanochat checkpoint's
    /// `tokenizer.pkl`.
    ///
    /// # Errors
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) or
    /// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) when it cannot be
    /// read; as [`from_tiktoken_pickle`](Self::from_tiktoken_pickle).
    pub fn from_tiktoken_pickle_file(path: &Path) -> BunsenResult<Self> {
        let bytes = std::fs::read(path).map_err(sys_at("read", path))?;
        Self::from_tiktoken_pickle(&bytes).map_err(|e| e.context(path.display()))
    }

    /// Reads a pickled `tiktoken` `Encoding`, as the `tokenizer.json`
    /// `transformers`' tiktoken converter writes for it, so it encodes as
    /// `tiktoken` does (nanochat's `RustBPETokenizer`).
    ///
    /// `serde_pickle` reads the `Encoding`'s state as data (the class is not
    /// looked up, and nothing in it runs): the split pattern, `pat_str`; the
    /// merge-ranked vocabulary, `mergeable_ranks`, each piece's rank its id;
    /// and the special tokens (`<|bos|>`, `<|user_start|>`, ...), after the
    /// ranks. The `tokenizer.json` has:
    ///
    /// - no normalizer;
    /// - a `Split` on `pat_str`, then a byte-level pre-tokenizer;
    /// - a byte-level BPE whose vocabulary is the ranks', in GPT-2's byte
    ///   alphabet, and whose merges are every split of a piece into two pieces,
    ///   in the order of the piece's rank, then of the parts' ranks. A piece of
    ///   the split text that is itself a token is that token, unmerged
    ///   (`ignore_merges`), as `tiktoken` looks a piece up whole before merging
    ///   it;
    /// - the special tokens, as special added tokens;
    /// - a byte-level decoder, which renders bytes that are not UTF-8 as
    ///   replacement characters, as `tiktoken`'s `decode` does.
    ///
    /// Upstream's `encode` is `tiktoken`'s `encode_ordinary`, a special
    /// token's spelling in it text:
    /// [`encode_plain`](crate::chat::ChatTokenizer::encode_plain).
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when it does not unpickle to an `Encoding`'s state (`pat_str`,
    /// `mergeable_ranks`, `special_tokens`), when a byte has no token, or
    /// when two tokens share an id; as [`from_json`](Self::from_json).
    pub fn from_tiktoken_pickle(pickle: &[u8]) -> BunsenResult<Self> {
        let state = match serde_pickle::value_from_slice(pickle, DeOptions::new()) {
            Ok(Value::Dict(state)) => state,
            Ok(other) => {
                return Err(BunsenError::invalid_resource(format!(
                    "tokenizer.pkl: not a tiktoken Encoding's state: a {}",
                    match other {
                        Value::List(_) => "list",
                        Value::None => "None",
                        _ => "value that is not a dict",
                    }
                )));
            }
            Err(e) => {
                return Err(
                    BunsenError::invalid_resource("tokenizer.pkl: not a pickle").with_cause(e)
                );
            }
        };
        let Value::String(pattern) = entry(&state, "pat_str")? else {
            return Err(BunsenError::invalid_resource(
                "tokenizer.pkl: `pat_str` is not a string",
            ));
        };
        let Value::Dict(mergeable_ranks) = entry(&state, "mergeable_ranks")? else {
            return Err(BunsenError::invalid_resource(
                "tokenizer.pkl: `mergeable_ranks` is not a dict",
            ));
        };
        let Value::Dict(special_tokens) = entry(&state, "special_tokens")? else {
            return Err(BunsenError::invalid_resource(
                "tokenizer.pkl: `special_tokens` is not a dict",
            ));
        };

        let mut ids: BTreeSet<u32> = BTreeSet::new();
        let mut claim = |id: u32| -> BunsenResult<()> {
            if !ids.insert(id) {
                return Err(BunsenError::invalid_resource(format!(
                    "tokenizer.pkl: two tokens with the id {id}"
                )));
            }
            Ok(())
        };
        let mut ranks: hashbrown::HashMap<&[u8], u32> = hashbrown::HashMap::new();
        for (piece, rank) in mergeable_ranks {
            let HashableValue::Bytes(piece) = piece else {
                return Err(BunsenError::invalid_resource(
                    "tokenizer.pkl: a merge-ranked piece that is not bytes",
                ));
            };
            let rank = id_of(rank)?;
            claim(rank)?;
            ranks.insert(piece.as_slice(), rank);
        }
        let mut specials = Vec::new();
        for (name, id) in special_tokens {
            let HashableValue::String(name) = name else {
                return Err(BunsenError::invalid_resource(
                    "tokenizer.pkl: a special token whose name is not a string",
                ));
            };
            let id = id_of(id)?;
            claim(id)?;
            specials.push((id, name.clone()));
        }
        specials.sort();
        if let Some(b) = (0..=255u8).find(|&b| !ranks.contains_key([b].as_slice())) {
            return Err(BunsenError::invalid_resource(format!(
                "tokenizer.pkl: no token for the byte {b:#04x}"
            )));
        }

        let alphabet = byte_alphabet();
        let spell =
            |piece: &[u8]| -> String { piece.iter().map(|&b| alphabet[b as usize]).collect() };
        let mut pieces: Vec<(&[u8], u32)> = ranks.iter().map(|(&p, &r)| (p, r)).collect();
        pieces.sort_by_key(|&(_, rank)| rank);
        let mut vocab = serde_json::Map::new();
        let mut merges = Vec::new();
        for &(piece, rank) in &pieces {
            vocab.insert(spell(piece), json!(rank));
            let mut splits: Vec<(u32, u32, usize)> = (1..piece.len())
                .filter_map(|at| {
                    let left = ranks.get(&piece[..at])?;
                    let right = ranks.get(&piece[at..])?;
                    Some((*left, *right, at))
                })
                .collect();
            splits.sort();
            merges.extend(
                splits
                    .into_iter()
                    .map(|(_, _, at)| json!([spell(&piece[..at]), spell(&piece[at..])])),
            );
        }
        let added: Vec<serde_json::Value> = specials
            .iter()
            .map(|(id, name)| {
                json!({
                    "id": id, "content": name, "special": true, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false
                })
            })
            .collect();
        let byte_level = json!({
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
            "use_regex": false
        });
        let tokenizer_json = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated",
                 "invert": false},
                byte_level,
            ]},
            "post_processor": null,
            "decoder": byte_level,
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
                "vocab": vocab, "merges": merges
            }
        });
        Self::from_json(&tokenizer_json.to_string())
            .map_err(|e| e.context("tokenizer.pkl, as a tokenizer.json"))
    }
}

#[cfg(test)]
/// A pickle as `tiktoken` writes an `Encoding`: `STACK_GLOBAL
/// tiktoken.core Encoding`, `NEWOBJ` on no arguments, then `BUILD` with
/// its state; the 256 bytes, the merges to `hello` (`he`, `ll`, `hell`,
/// `hello`), and nanochat's five layout tokens.
pub(crate) fn toy_pickle() -> Vec<u8> {
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
    let state = serde_pickle::value_to_vec(&Value::Dict(state), Default::default()).unwrap();
    // `PROTO 2`, the class, `NEWOBJ` on `()`, the state (its own `PROTO`
    // and `STOP` stripped), `BUILD`, `STOP`.
    let mut out = vec![0x80, 2];
    out.extend_from_slice(b"ctiktoken.core\nEncoding\n");
    out.extend_from_slice(b")\x81");
    out.extend_from_slice(&state[2..state.len() - 1]);
    out.extend_from_slice(b"b.");
    out
}

#[cfg(test)]
mod tests {
    use bunsen::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::support::tokens::{
            Detokenizer,
            Tokenizer,
        },
    };

    use super::*;
    use crate::chat::ChatTokenizer;

    #[test]
    fn test_encode_and_decode() {
        let tok = HfTokenizer::from_tiktoken_pickle(&toy_pickle()).unwrap();
        assert_eq!(tok.len(), 256 + 4 + 5);
        assert_eq!(tok.token_id("<|bos|>"), Some(260));
        assert_eq!(
            tok.encode_plain("hello").unwrap(),
            [259],
            "`hello` is one token"
        );
        assert_eq!(tok.encode_plain("hell").unwrap(), [258]);
        assert_eq!(tok.encode_plain("hellohe").unwrap(), [259, 256]);
        assert_eq!(tok.encode_plain("").unwrap(), Vec::<i64>::new());
        let ids = tok.encode_plain("hello world").unwrap();
        assert_eq!(ids[0], 259);
        assert_eq!(tok.detokenize(&ids).unwrap(), "hello world");
        // A special token's spelling is text, as `encode_ordinary` has it.
        assert!(tok.encode_plain("<|bos|>").unwrap().len() > 1);
        assert_eq!(tok.encode("<|bos|>hello").unwrap(), [260, 259]);
        assert_eq!(tok.detokenize(&[260, 259]).unwrap(), "<|bos|>hello");
        // Bytes that are not UTF-8 decode as a replacement character.
        assert_eq!(tok.detokenize(&[0xC3]).unwrap(), "\u{FFFD}");
        let ids = tok.encode_plain("é").unwrap();
        assert_eq!(ids, [0xC3, 0xA9]);
        assert_eq!(tok.detokenize(&ids).unwrap(), "é");
    }

    #[test]
    fn test_byte_alphabet() {
        let alphabet = byte_alphabet();
        assert_eq!(alphabet[b'a' as usize], 'a');
        assert_eq!(alphabet[0], '\u{100}');
        assert_eq!(alphabet[b' ' as usize], 'Ġ');
        let distinct: std::collections::BTreeSet<char> = alphabet.iter().copied().collect();
        assert_eq!(distinct.len(), 256);
    }

    #[test]
    fn test_refuses_what_is_not_an_encoding() {
        let not_a_dict =
            serde_pickle::value_to_vec(&Value::List(vec![]), Default::default()).unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("not a tiktoken Encoding")
            .assert_err(&HfTokenizer::from_tiktoken_pickle(&not_a_dict));
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("not a pickle")
            .assert_err(&HfTokenizer::from_tiktoken_pickle(b"junk"));
    }
}
