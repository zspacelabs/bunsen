//! nanochat's tokenizer, on wordchipper.
use std::{
    collections::BTreeMap,
    path::Path,
    sync::Arc,
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::tokens::Detokenizer,
};
use serde_pickle::{
    DeOptions,
    HashableValue,
    Value,
};
use wordchipper::{
    TokenEncoder,
    TokenizerOptions,
    UnifiedTokenVocab,
    spanners::TextSpanningConfig,
    vocab::{
        ByteMapVocab,
        SpanMapVocab,
        SpanTokenMap,
    },
};

/// A nanochat checkpoint's tokenizer, from its `tokenizer.pkl`: text to ids
/// and back, as upstream's `RustBPETokenizer` encodes.
///
/// The file is a pickled `tiktoken` `Encoding`. `serde_pickle` reads its
/// state as data (the class is not looked up, and nothing in it runs): the
/// split pattern, `pat_str`; the merge-ranked vocabulary, `mergeable_ranks`,
/// each piece's rank its id; and the special tokens (`<|bos|>`,
/// `<|user_start|>`, ...), after the ranks. Text encodes as upstream's
/// `encode` does, `tiktoken`'s `encode_ordinary`: no normalization, and a
/// special token's spelling in it is text; a special token is its own id,
/// [`special`](Self::special).
#[derive(Clone)]
pub struct NanoChatTokenizer {
    encoder: Arc<wordchipper::Tokenizer<u32>>,
    bytes: BTreeMap<i64, Vec<u8>>,
    specials: BTreeMap<String, i64>,
}

impl std::fmt::Debug for NanoChatTokenizer {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("NanoChatTokenizer")
            .field("tokens", &self.bytes.len())
            .field("specials", &self.specials.len())
            .finish()
    }
}

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
fn id_of(value: &Value) -> BunsenResult<i64> {
    match value {
        Value::I64(id) if *id >= 0 && *id <= u32::MAX as i64 => Ok(*id),
        other => Err(BunsenError::invalid_resource(format!(
            "tokenizer.pkl: a token id that is not one: {other:?}"
        ))),
    }
}

impl NanoChatTokenizer {
    /// Reads a checkpoint's `tokenizer.pkl`.
    ///
    /// # Errors
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) or
    /// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) when it cannot be
    /// read; as [`from_pickle`](Self::from_pickle).
    pub fn from_file(path: &Path) -> BunsenResult<Self> {
        let bytes = std::fs::read(path).map_err(sys_at("read", path))?;
        Self::from_pickle(&bytes).map_err(|e| e.context(path.display()))
    }

    /// Reads a pickled `tiktoken` `Encoding`.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when it does not unpickle to an `Encoding`'s state (`pat_str`,
    /// `mergeable_ranks`, `special_tokens`), when a byte has no token, or
    /// when two tokens share an id.
    pub fn from_pickle(pickle: &[u8]) -> BunsenResult<Self> {
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
        let Value::Dict(ranks) = entry(&state, "mergeable_ranks")? else {
            return Err(BunsenError::invalid_resource(
                "tokenizer.pkl: `mergeable_ranks` is not a dict",
            ));
        };
        let Value::Dict(special_tokens) = entry(&state, "special_tokens")? else {
            return Err(BunsenError::invalid_resource(
                "tokenizer.pkl: `special_tokens` is not a dict",
            ));
        };

        let mut spans: SpanTokenMap<u32> = Default::default();
        let mut bytes: BTreeMap<i64, Vec<u8>> = BTreeMap::new();
        let mut claim = |id: i64, piece: Vec<u8>| -> BunsenResult<()> {
            if bytes.insert(id, piece).is_some() {
                return Err(BunsenError::invalid_resource(format!(
                    "tokenizer.pkl: two tokens with the id {id}"
                )));
            }
            Ok(())
        };
        for (piece, rank) in ranks {
            let HashableValue::Bytes(piece) = piece else {
                return Err(BunsenError::invalid_resource(
                    "tokenizer.pkl: a merge-ranked piece that is not bytes",
                ));
            };
            let id = id_of(rank)?;
            spans.insert(piece.clone(), id as u32);
            claim(id, piece.clone())?;
        }
        let mut specials = BTreeMap::new();
        for (name, id) in special_tokens {
            let HashableValue::String(name) = name else {
                return Err(BunsenError::invalid_resource(
                    "tokenizer.pkl: a special token whose name is not a string",
                ));
            };
            let id = id_of(id)?;
            claim(id, name.as_bytes().to_vec())?;
            specials.insert(name.clone(), id);
        }

        let byte_tokens = (0..=255u8)
            .map(|b| {
                spans.get(&vec![b]).copied().ok_or_else(|| {
                    BunsenError::invalid_resource(format!(
                        "tokenizer.pkl: no token for the byte {b:#04x}"
                    ))
                })
            })
            .collect::<BunsenResult<Vec<u32>>>()?;
        let span_vocab =
            SpanMapVocab::new(ByteMapVocab::from_byte_to_token(&byte_tokens), spans)
                .map_err(|e| BunsenError::invalid_resource("the vocabulary").with_cause(e))?;
        let config: TextSpanningConfig<u32> = TextSpanningConfig::from_pattern(pattern.clone());
        let vocab = UnifiedTokenVocab::from_span_vocab(config, span_vocab)
            .map_err(|e| BunsenError::invalid_resource("the vocabulary").with_cause(e))?;
        let encoder = TokenizerOptions::default().build(Arc::new(vocab));
        Ok(Self {
            encoder,
            bytes,
            specials,
        })
    }

    /// The id of the special token `name`: `<|bos|>`, say.
    pub fn special(
        &self,
        name: &str,
    ) -> Option<i64> {
        self.specials.get(name).copied()
    }

    /// The special tokens, by name.
    pub fn specials(&self) -> &BTreeMap<String, i64> {
        &self.specials
    }

    /// Whether `id` is a token.
    pub fn contains(
        &self,
        id: i64,
    ) -> bool {
        self.bytes.contains_key(&id)
    }

    /// The token count, the special tokens included.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Encodes text as upstream's `encode` does: every character text, a
    /// special token's spelling included.
    ///
    /// # Errors
    /// As wordchipper's encoder.
    pub fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let ids = self
            .encoder
            .try_encode(text, Some(&wordchipper::vocab::SpecialFilter::IncludeNone))
            .map_err(|e| {
                BunsenError::invalid_resource("text wordchipper cannot encode").with_cause(e)
            })?;
        Ok(ids.into_iter().map(i64::from).collect())
    }

    /// The bytes `ids` spell, a special token as its name.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// for an id outside the vocabulary.
    pub fn decode_bytes(
        &self,
        ids: &[i64],
    ) -> BunsenResult<Vec<u8>> {
        let mut out = Vec::new();
        for id in ids {
            out.extend_from_slice(self.bytes.get(id).ok_or_else(|| {
                BunsenError::invalid_resource(format!("token id {id} is outside the vocabulary"))
            })?);
        }
        Ok(out)
    }
}

/// Upstream's `decode`: the bytes, as UTF-8 with replacement characters.
impl Detokenizer for NanoChatTokenizer {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use bunsen::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    use super::*;

    /// A pickle as `tiktoken` writes an `Encoding`: `STACK_GLOBAL
    /// tiktoken.core Encoding`, `NEWOBJ` on no arguments, then `BUILD` with
    /// its state; the 256 bytes, the merges to `hello` (`he`, `ll`, `hell`,
    /// `hello`), and two special tokens.
    fn toy_pickle() -> Vec<u8> {
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
        specials.insert(HashableValue::String("<|bos|>".into()), Value::I64(260));
        specials.insert(
            HashableValue::String("<|user_start|>".into()),
            Value::I64(261),
        );
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

    #[test]
    fn test_encode_and_decode() {
        let tok = NanoChatTokenizer::from_pickle(&toy_pickle()).unwrap();
        assert_eq!(tok.len(), 256 + 4 + 2);
        assert_eq!(tok.special("<|bos|>"), Some(260));
        assert_eq!(tok.encode("hello").unwrap(), [259], "`hello` is one token");
        assert_eq!(tok.encode("hell").unwrap(), [258]);
        assert_eq!(tok.encode("").unwrap(), Vec::<i64>::new());
        let ids = tok.encode("hello world").unwrap();
        assert_eq!(tok.detokenize(&ids).unwrap(), "hello world");
        // A special token's spelling is text, as `encode_ordinary` has it.
        assert!(tok.encode("<|bos|>").unwrap().len() > 1);
        assert_eq!(tok.detokenize(&[260, 259]).unwrap(), "<|bos|>hello");
    }

    #[test]
    fn test_refuses_what_is_not_an_encoding() {
        let not_a_dict =
            serde_pickle::value_to_vec(&Value::List(vec![]), Default::default()).unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("not a tiktoken Encoding")
            .assert_err(&NanoChatTokenizer::from_pickle(&not_a_dict));
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("not a pickle")
            .assert_err(&NanoChatTokenizer::from_pickle(b"junk"));
    }
}
