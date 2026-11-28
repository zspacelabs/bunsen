//! # `mistral_common` interop: Mistral's tokenizer files, as its package reads them
//!
//! `mistral_common` defines `tekken.json`, the vocabulary of Mistral's
//! byte-level BPE (tekken) and, for an audio model, its `audio` section.
//! Every Mistral model reads it alike.
//!
//! - [`TekkenMirror`]: the file, its sections [`TekkenConfigMirror`],
//!   [`TekkenTokenMirror`], [`TekkenSpecialMirror`] and [`TekkenAudioMirror`];
//!   [`from_json`](TekkenMirror::from_json) parses it, and
//!   [`tekken_vocabulary`](TekkenMirror::tekken_vocabulary) is the
//!   [`TekkenVocabulary`] it gives.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec::Vec,
};

use serde::Deserialize;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::tokens::{
        TekkenVocabulary,
        tiktoken_util::lenient_decode_base64,
    },
};

/// `tekken.json`, as `mistral_common` names its fields: the sections bunsen
/// reads. Every other key is ignored.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TekkenMirror {
    /// The vocabulary's sizes.
    pub config: TekkenConfigMirror,

    /// The vocabulary entries, by rank.
    pub vocab: Vec<TekkenTokenMirror>,

    /// The special tokens, by rank.
    pub special_tokens: Vec<TekkenSpecialMirror>,

    /// An audio model's stream geometry; a text model's file has none.
    #[serde(default)]
    pub audio: Option<TekkenAudioMirror>,
}

/// `tekken.json`'s `config` section: the sizes the model's vocabulary takes.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct TekkenConfigMirror {
    /// The model's vocabulary size, the special tokens included.
    pub default_vocab_size: usize,

    /// The special tokens, which take the first ids.
    pub default_num_special_tokens: usize,
}

/// One vocabulary entry.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct TekkenTokenMirror {
    /// Its rank among the entries.
    pub rank: usize,

    /// Its bytes, in base64.
    pub token_bytes: String,
}

/// One special token.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct TekkenSpecialMirror {
    /// Its rank, its id.
    pub rank: usize,

    /// Its spelling.
    pub token_str: String,
}

/// `tekken.json`'s `audio` section: an audio model's stream geometry.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TekkenAudioMirror {
    /// The model's positions per second.
    pub frame_rate: f64,

    /// The transcription's delay behind the audio, in ms.
    pub transcription_delay_ms: usize,

    /// The positions of left padding a stream starts with.
    pub streaming_n_left_pad_tokens: usize,
}

impl TekkenMirror {
    /// Parses a `tekken.json`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse or lacks a section.
    pub fn from_json(json: &str) -> BunsenResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a tekken.json").with_cause(e))
    }

    /// The [`TekkenVocabulary`] the file gives: the specials at their ranks,
    /// each entry of rank `r` at id `default_num_special_tokens + r`, the
    /// entries at or past `default_vocab_size` not read.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when a special token's rank is not below the special count, or a
    /// vocabulary entry's bytes are not base64.
    pub fn tekken_vocabulary(&self) -> BunsenResult<TekkenVocabulary> {
        let num_special = self.config.default_num_special_tokens;
        let vocab_size = self.config.default_vocab_size;
        let mut spans = BTreeMap::new();
        let mut specials = BTreeMap::new();
        for special in &self.special_tokens {
            if special.rank >= num_special {
                return Err(BunsenError::invalid_resource(format!(
                    "special token {:?} has rank {}, not below the {num_special} specials",
                    special.token_str, special.rank
                )));
            }
            spans.insert(special.rank, special.token_str.as_bytes().to_vec());
            specials.insert(special.token_str.clone(), special.rank);
        }
        for token in &self.vocab {
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
        Ok(TekkenVocabulary::from_parts(
            spans,
            specials,
            num_special,
            vocab_size,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::testing::ErrorMatcher;

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
    fn test_tekken_vocabulary() {
        let v = TekkenMirror::from_json(TEKKEN)
            .unwrap()
            .tekken_vocabulary()
            .unwrap();
        assert_eq!(v.num_special(), 4);
        assert_eq!(v.vocab_size(), 6);
        // Rank 2 is id 6: past the vocabulary, not read.
        assert_eq!(v.len(), 6);
        assert_eq!(v.bytes(6), None);
        assert_eq!(v.special_id("[STREAMING_PAD]"), Some(3));
        assert!(v.is_special(3) && !v.is_special(4));
        assert_eq!(v.text(&[4, 5]).unwrap(), " In the");
        assert_eq!(v.text(&[1]).unwrap(), "<s>");
        ErrorMatcher::invalid_resource()
            .message_contains("outside the vocabulary")
            .assert_err(&v.text(&[6]));
    }

    #[test]
    fn test_refuses_a_malformed_file() {
        ErrorMatcher::invalid_resource()
            .message_contains("tekken.json")
            .assert_err(&TekkenMirror::from_json("{}"));
        let bad = TEKKEN.replace("IElu", "I@lu");
        ErrorMatcher::invalid_resource()
            .message_contains("not base64")
            .assert_err(&TekkenMirror::from_json(&bad).and_then(|m| m.tekken_vocabulary()));
        let bad = TEKKEN.replace(r#""rank": 3, "token_str""#, r#""rank": 9, "token_str""#);
        ErrorMatcher::invalid_resource()
            .message_contains("not below the 4 specials")
            .assert_err(&TekkenMirror::from_json(&bad).and_then(|m| m.tekken_vocabulary()));
    }
}
