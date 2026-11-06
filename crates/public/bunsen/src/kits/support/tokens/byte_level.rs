//! Byte-level BPE vocabularies, as `transformers` stores them.
use alloc::collections::BTreeMap;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// GPT-2's printable spelling of each byte: the alphabet `vocab.json` writes
/// byte-level tokens in. Bytes that print as themselves keep their code
/// point; the rest are moved past 255, in byte order.
fn byte_alphabet() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..=255u8 {
        let printable = (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
        table[b as usize] = if printable {
            char::from(b)
        } else {
            let c = char::from_u32(next).expect("a code point below 512");
            next += 1;
            c
        };
    }
    table
}

/// The ids and bytes of a byte-level BPE vocabulary (GPT-2's, Qwen's): the
/// `{ id -> bytes }` table
/// `WordchipperDetokenizer::from_spans` (feature `tokenizer`) takes.
///
/// Read from `transformers`' files: a `vocab.json` (`{ token: id }`, each
/// token spelled in GPT-2's byte alphabet) with the added tokens of a
/// `tokenizer_config.json`, or a `tokenizer.json` holding both. Added tokens
/// (the chat template's specials) are their content's UTF-8 bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ByteLevelVocabulary {
    spans: BTreeMap<usize, Vec<u8>>,
    added: BTreeMap<String, usize>,
}

impl ByteLevelVocabulary {
    /// Reads a `vocab.json`, and the `added_tokens_decoder` of a
    /// `tokenizer_config.json` when one is given.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when either does not parse, a token is spelled outside the byte
    /// alphabet, or an id is not an integer.
    pub fn from_vocab_json(
        vocab_json: &str,
        tokenizer_config_json: Option<&str>,
    ) -> BunsenResult<Self> {
        let vocab: BTreeMap<String, u64> = serde_json::from_str(vocab_json)
            .map_err(|e| BunsenError::invalid_resource("a vocab.json").with_cause(e))?;
        let mut out = Self::from_pieces(vocab.iter().map(|(t, &id)| (t.as_str(), id as usize)))?;
        if let Some(config) = tokenizer_config_json {
            let config: serde_json::Value = serde_json::from_str(config).map_err(|e| {
                BunsenError::invalid_resource("a tokenizer_config.json").with_cause(e)
            })?;
            if let Some(added) = config
                .get("added_tokens_decoder")
                .and_then(|a| a.as_object())
            {
                for (id, token) in added {
                    let id: usize = id.parse().map_err(|e| {
                        BunsenError::invalid_resource(format!("an added-token id `{id}`"))
                            .with_cause(e)
                    })?;
                    let content =
                        token
                            .get("content")
                            .and_then(|c| c.as_str())
                            .ok_or_else(|| {
                                BunsenError::invalid_resource(format!(
                                    "added token {id} has no content"
                                ))
                            })?;
                    out.add(id, content);
                }
            }
        }
        Ok(out)
    }

    /// Reads a `tokenizer.json`: its `model.vocab` and `added_tokens`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse or has no `model.vocab` map, or a token is
    /// spelled outside the byte alphabet.
    pub fn from_tokenizer_json(json: &str) -> BunsenResult<Self> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a tokenizer.json").with_cause(e))?;
        let vocab = value
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                BunsenError::invalid_resource("a tokenizer.json without `model.vocab`")
            })?;
        let pieces = vocab
            .iter()
            .map(|(t, id)| {
                id.as_u64()
                    .map(|id| (t.as_str(), id as usize))
                    .ok_or_else(|| {
                        BunsenError::invalid_resource(format!("`{t}` has no integer id"))
                    })
            })
            .collect::<BunsenResult<Vec<_>>>()?;
        let mut out = Self::from_pieces(pieces)?;
        for token in value
            .get("added_tokens")
            .and_then(|a| a.as_array())
            .into_iter()
            .flatten()
        {
            if let (Some(id), Some(content)) = (
                token.get("id").and_then(|i| i.as_u64()),
                token.get("content").and_then(|c| c.as_str()),
            ) {
                out.add(id as usize, content);
            }
        }
        Ok(out)
    }

    /// Builds from `(token, id)` pairs spelled in GPT-2's byte alphabet.
    fn from_pieces<'a>(pieces: impl IntoIterator<Item = (&'a str, usize)>) -> BunsenResult<Self> {
        let mut inverse = BTreeMap::new();
        for (b, c) in byte_alphabet().iter().enumerate() {
            inverse.insert(*c, b as u8);
        }
        let mut spans = BTreeMap::new();
        for (token, id) in pieces {
            let bytes = token
                .chars()
                .map(|c| {
                    inverse.get(&c).copied().ok_or_else(|| {
                        BunsenError::invalid_resource(format!(
                            "token {id} is spelled outside the byte alphabet: {token:?}"
                        ))
                    })
                })
                .collect::<BunsenResult<Vec<u8>>>()?;
            spans.insert(id, bytes);
        }
        Ok(Self {
            spans,
            added: BTreeMap::new(),
        })
    }

    /// Adds (or replaces) an added token.
    fn add(
        &mut self,
        id: usize,
        content: &str,
    ) {
        self.spans.insert(id, content.as_bytes().to_vec());
        self.added.insert(content.to_string(), id);
    }

    /// The token count, added tokens included.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The bytes of `id`, if the vocabulary has it.
    pub fn bytes(
        &self,
        id: usize,
    ) -> Option<&[u8]> {
        self.spans.get(&id).map(Vec::as_slice)
    }

    /// The id of an added token, by its content: `<|im_end|>`, say.
    pub fn added_id(
        &self,
        content: &str,
    ) -> Option<usize> {
        self.added.get(content).copied()
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

    #[test]
    fn test_byte_alphabet() {
        let table = byte_alphabet();
        assert_eq!(table[b'a' as usize], 'a');
        // Space and newline are not printable in GPT-2's alphabet.
        assert_eq!(table[b' ' as usize], '\u{120}');
        assert_eq!(table[b'\n' as usize], '\u{10a}');
        let distinct: alloc::collections::BTreeSet<char> = table.iter().copied().collect();
        assert_eq!(distinct.len(), 256);
    }

    #[test]
    fn test_readers_agree() {
        let vocab = r#"{"Ġthe": 0, "ĠmoonĊ": 1, "ä½": 2, "ł": 3}"#;
        let config = r#"{"added_tokens_decoder": {"5": {"content": "<|im_end|>"}}}"#;
        let a = ByteLevelVocabulary::from_vocab_json(vocab, Some(config)).unwrap();
        assert_eq!(a.text(&[0, 1]).unwrap(), " the moon\n");
        assert_eq!(a.added_id("<|im_end|>"), Some(5));
        assert_eq!(a.text(&[5]).unwrap(), "<|im_end|>");
        assert!(a.text(&[9]).is_err());

        let tok = r#"{"model": {"vocab": {"Ġthe": 0, "ĠmoonĊ": 1, "ä½": 2, "ł": 3}},
                      "added_tokens": [{"id": 5, "content": "<|im_end|>"}]}"#;
        let b = ByteLevelVocabulary::from_tokenizer_json(tok).unwrap();
        assert_eq!(a, b);
        assert!(ByteLevelVocabulary::from_vocab_json(r#"{"☃": 0}"#, None).is_err());
    }
}
