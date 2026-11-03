//! Mistral's tekken vocabularies.
use std::collections::BTreeMap;

use serde::Deserialize;

use super::tiktoken_util::lenient_decode_base64;
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The ids and bytes of a tekken vocabulary, Mistral's byte-level BPE
/// (`tekken.json`, as `mistral_common` reads it): the `{ id -> bytes }` table
/// `WordchipperDetokenizer::from_spans` (feature `tokenizer`) takes.
///
/// The first `num_special` ids are the special tokens, in rank order (`<s>`
/// is 1, `</s>` 2, Voxtral Realtime's `[STREAMING_PAD]` 32); vocabulary entry
/// of rank `r` is id `num_special + r`, its bytes the file's base64
/// `token_bytes`. The file may list more entries than the model's vocabulary
/// holds; only the ids below `vocab_size` are read. A special token renders
/// as its own spelling; which ids to render is the caller's policy
/// ([`is_special`](Self::is_special)).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TekkenVocabulary {
    spans: BTreeMap<usize, Vec<u8>>,
    specials: BTreeMap<String, usize>,
    num_special: usize,
    vocab_size: usize,
}

/// `tekken.json`'s fields, as the file names them.
#[derive(Deserialize)]
struct TekkenJson {
    config: TekkenConfigJson,
    vocab: Vec<TekkenTokenJson>,
    special_tokens: Vec<TekkenSpecialJson>,
}

#[derive(Deserialize)]
struct TekkenConfigJson {
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(Deserialize)]
struct TekkenTokenJson {
    rank: usize,
    token_bytes: String,
}

#[derive(Deserialize)]
struct TekkenSpecialJson {
    rank: usize,
    token_str: String,
}

impl TekkenVocabulary {
    /// Reads a `tekken.json`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse, a special token's rank is not below the
    /// special count, or a vocabulary entry's bytes are not base64.
    pub fn from_tekken_json(json: &str) -> BunsenResult<Self> {
        let file: TekkenJson = serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a tekken.json").with_cause(e))?;
        let num_special = file.config.default_num_special_tokens;
        let vocab_size = file.config.default_vocab_size;
        let mut spans = BTreeMap::new();
        let mut specials = BTreeMap::new();
        for special in file.special_tokens {
            if special.rank >= num_special {
                return Err(BunsenError::invalid_resource(format!(
                    "special token {:?} has rank {}, not below the {num_special} specials",
                    special.token_str, special.rank
                )));
            }
            spans.insert(special.rank, special.token_str.as_bytes().to_vec());
            specials.insert(special.token_str, special.rank);
        }
        for token in file.vocab {
            let id = num_special + token.rank;
            if id >= vocab_size {
                continue;
            }
            let bytes = lenient_decode_base64(&token.token_bytes).ok_or_else(|| {
                BunsenError::invalid_resource(format!(
                    "vocabulary rank {} is not base64: {:?}",
                    token.rank, token.token_bytes
                ))
            })?;
            spans.insert(id, bytes);
        }
        Ok(Self {
            spans,
            specials,
            num_special,
            vocab_size,
        })
    }

    /// The token count, special tokens included.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The model's vocabulary size: one past the largest id read.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The special-token count: the first id that is a vocabulary entry.
    pub fn num_special(&self) -> usize {
        self.num_special
    }

    /// Whether `id` is a special token's.
    pub fn is_special(
        &self,
        id: usize,
    ) -> bool {
        id < self.num_special
    }

    /// The id of a special token, by its spelling: `[STREAMING_PAD]`, say.
    pub fn special_id(
        &self,
        spelling: &str,
    ) -> Option<usize> {
        self.specials.get(spelling).copied()
    }

    /// The bytes of `id`, if the vocabulary has it.
    pub fn bytes(
        &self,
        id: usize,
    ) -> Option<&[u8]> {
        self.spans.get(&id).map(Vec::as_slice)
    }

    /// The `{ id -> bytes }` table, in id order.
    pub fn spans(&self) -> impl Iterator<Item = (usize, Vec<u8>)> + '_ {
        self.spans.iter().map(|(&id, b)| (id, b.clone()))
    }

    /// The text `ids` spell: their bytes joined, then decoded as UTF-8,
    /// replacing what is not.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if an id is outside the vocabulary.
    pub fn text(
        &self,
        ids: &[usize],
    ) -> BunsenResult<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.bytes(id).ok_or_else(|| {
                BunsenError::invalid_resource(format!("token id {id} is outside the vocabulary"))
            })?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    /// Four specials, then " In" (`IElu`), " the" (`IHRoZQ==`), and an
    /// entry past the vocabulary size.
    const TEKKEN: &str = r#"{
        "config": {"pattern": "", "num_vocab_tokens": 3, "default_vocab_size": 6,
                   "default_num_special_tokens": 4, "version": "v7"},
        "vocab": [
            {"rank": 0, "token_bytes": "IElu", "token_str": " In"},
            {"rank": 1, "token_bytes": "IHRoZQ==", "token_str": " the"},
            {"rank": 2, "token_bytes": "eA==", "token_str": "x"}
        ],
        "special_tokens": [
            {"rank": 0, "token_str": "<unk>", "is_control": true},
            {"rank": 1, "token_str": "<s>", "is_control": true},
            {"rank": 2, "token_str": "</s>", "is_control": true},
            {"rank": 3, "token_str": "[STREAMING_PAD]", "is_control": true}
        ]
    }"#;

    #[test]
    fn test_from_tekken_json() {
        let v = TekkenVocabulary::from_tekken_json(TEKKEN).unwrap();
        assert_eq!(v.num_special(), 4);
        assert_eq!(v.vocab_size(), 6);
        // Rank 2 is id 6: past the vocabulary, not read.
        assert_eq!(v.len(), 6);
        assert_eq!(v.bytes(6), None);
        assert_eq!(v.special_id("[STREAMING_PAD]"), Some(3));
        assert!(v.is_special(3) && !v.is_special(4));
        assert_eq!(v.text(&[4, 5]).unwrap(), " In the");
        assert_eq!(v.text(&[1]).unwrap(), "<s>");
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("outside the vocabulary")
            .assert_err(&v.text(&[6]));
    }

    #[test]
    fn test_refuses_a_malformed_file() {
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("tekken.json")
            .assert_err(&TekkenVocabulary::from_tekken_json("{}"));
        let bad = TEKKEN.replace("IElu", "I@lu");
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("not base64")
            .assert_err(&TekkenVocabulary::from_tekken_json(&bad));
        let bad = TEKKEN.replace(r#""rank": 3, "token_str""#, r#""rank": 9, "token_str""#);
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("not below the 4 specials")
            .assert_err(&TekkenVocabulary::from_tekken_json(&bad));
    }
}
