//! Qwen3's tokenizer, on wordchipper.
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
    kits::tokens::{
        ByteLevelVocabulary,
        Detokenizer,
    },
};
use serde_json::Value;
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

/// A Qwen3 checkpoint's tokenizer: text to ids and back, as `transformers`'
/// `Qwen2TokenizerFast` encodes, from its `tokenizer.json` and
/// `tokenizer_config.json`.
///
/// Encoding finds the added tokens first (`<|im_start|>`, `<think>`, ...:
/// every one, special or not, leftmost and then longest), NFC-normalizes the
/// text between them with the Unicode tables `transformers`' tokenizers use,
/// and encodes that with wordchipper under the checkpoint's pre-tokenizer
/// and merge-ranked vocabulary. The added tokens are those of both files: a
/// `-Base` checkpoint's `tokenizer.json` omits four that its
/// `tokenizer_config.json` lists, and `transformers` adds them back.
///
/// It reads only the byte-level BPE `transformers` writes for Qwen3: an NFC
/// normalizer, a `Split` pre-tokenizer then `ByteLevel`, no byte fallback, no
/// merge-skipping, and added tokens that strip nothing. Anything else is
/// refused rather than misencoded.
#[derive(Clone)]
pub struct Qwen3Tokenizer {
    encoder: Arc<wordchipper::Tokenizer<u32>>,
    bytes: BTreeMap<i64, Vec<u8>>,
    added: Vec<(String, i64)>,
}

impl std::fmt::Debug for Qwen3Tokenizer {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("Qwen3Tokenizer")
            .field("tokens", &self.bytes.len())
            .field("added", &self.added.len())
            .finish()
    }
}

/// `field` of `value`, or an `InvalidResource` naming it.
fn field<'a>(
    value: &'a Value,
    path: &str,
) -> BunsenResult<&'a Value> {
    path.split('.')
        .try_fold(value, |v, key| v.get(key))
        .ok_or_else(|| BunsenError::invalid_resource(format!("tokenizer.json: no `{path}`")))
}

/// Refuses a `tokenizer.json` this encoder would not reproduce.
fn check_unsupported(
    what: &str,
    ok: bool,
) -> BunsenResult<()> {
    if ok {
        Ok(())
    } else {
        Err(BunsenError::unsupported(format!(
            "a Qwen3 tokenizer.json with {what}"
        )))
    }
}

impl Qwen3Tokenizer {
    /// Reads a checkpoint's `tokenizer.json`, and its `tokenizer_config.json`
    /// when it has one.
    ///
    /// # Errors
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) or
    /// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) when a file cannot
    /// be read; as [`from_json`](Self::from_json).
    pub fn from_files(
        tokenizer_json: &Path,
        tokenizer_config_json: Option<&Path>,
    ) -> BunsenResult<Self> {
        let read = |p: &Path| std::fs::read_to_string(p).map_err(sys_at("read", p));
        let tokenizer = read(tokenizer_json)?;
        let config = tokenizer_config_json.map(read).transpose()?;
        Self::from_json(&tokenizer, config.as_deref())
            .map_err(|e| e.context(tokenizer_json.display()))
    }

    /// Reads a `tokenizer.json`, and a `tokenizer_config.json` when given.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when either does not parse or lacks a field;
    /// [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) for a
    /// tokenizer this encoder does not reproduce, naming what.
    pub fn from_json(
        tokenizer_json: &str,
        tokenizer_config_json: Option<&str>,
    ) -> BunsenResult<Self> {
        let t: Value = serde_json::from_str(tokenizer_json)
            .map_err(|e| BunsenError::invalid_resource("a tokenizer.json").with_cause(e))?;

        let model = field(&t, "model")?;
        check_unsupported(
            "a model other than BPE",
            model.get("type").and_then(Value::as_str) == Some("BPE"),
        )?;
        for flag in ["byte_fallback", "ignore_merges"] {
            check_unsupported(
                &format!("`model.{flag}`"),
                !model.get(flag).and_then(Value::as_bool).unwrap_or(false),
            )?;
        }
        let normalizer = t.get("normalizer").filter(|n| !n.is_null());
        check_unsupported(
            "a normalizer other than NFC",
            normalizer.is_none_or(|n| n.get("type").and_then(Value::as_str) == Some("NFC")),
        )?;
        let pretokenizers = field(&t, "pre_tokenizer.pretokenizers")?
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default();
        let pattern = match pretokenizers {
            [split, byte_level]
                if split.get("type").and_then(Value::as_str) == Some("Split")
                    && byte_level.get("type").and_then(Value::as_str) == Some("ByteLevel")
                    && !byte_level
                        .get("use_regex")
                        .and_then(Value::as_bool)
                        .unwrap_or(true) =>
            {
                field(split, "pattern.Regex")?
                    .as_str()
                    .ok_or_else(|| BunsenError::invalid_resource("a non-string split pattern"))?
                    .to_string()
            }
            _ => {
                return Err(BunsenError::unsupported(
                    "a Qwen3 tokenizer.json whose pre-tokenizer is not Split then ByteLevel",
                ));
            }
        };

        // Every added token, from both files, by id.
        let mut added: BTreeMap<i64, String> = BTreeMap::new();
        let mut add = |id: i64, token: &Value| -> BunsenResult<()> {
            for flag in ["lstrip", "rstrip", "single_word"] {
                check_unsupported(
                    &format!("an added token that sets `{flag}`"),
                    !token.get(flag).and_then(Value::as_bool).unwrap_or(false),
                )?;
            }
            let content = field(token, "content")?
                .as_str()
                .ok_or_else(|| BunsenError::invalid_resource("an added token's content"))?;
            added.insert(id, content.to_string());
            Ok(())
        };
        for token in field(&t, "added_tokens")?.as_array().into_iter().flatten() {
            let id = field(token, "id")?
                .as_i64()
                .ok_or_else(|| BunsenError::invalid_resource("an added token's id"))?;
            add(id, token)?;
        }
        if let Some(config) = tokenizer_config_json {
            let c: Value = serde_json::from_str(config).map_err(|e| {
                BunsenError::invalid_resource("a tokenizer_config.json").with_cause(e)
            })?;
            if let Some(decoder) = c.get("added_tokens_decoder").and_then(Value::as_object) {
                for (id, token) in decoder {
                    let id: i64 = id.parse().map_err(|_| {
                        BunsenError::invalid_resource(format!("an added token id `{id}`"))
                    })?;
                    add(id, token)?;
                }
            }
        }

        // The merge-ranked vocabulary, its added tokens left out: wordchipper
        // derives its merges from the spans, which every added token would
        // break.
        let vocabulary = ByteLevelVocabulary::from_tokenizer_json(tokenizer_json)?;
        let mut spans: SpanTokenMap<u32> = Default::default();
        let mut bytes: BTreeMap<i64, Vec<u8>> = BTreeMap::new();
        for (id, span) in vocabulary.spans() {
            let id = id as i64;
            if !added.contains_key(&id) {
                spans.insert(span.clone(), id as u32);
            }
            bytes.insert(id, span);
        }
        for (&id, content) in &added {
            bytes.insert(id, content.as_bytes().to_vec());
        }
        let byte_tokens = (0..=255u8)
            .map(|b| {
                spans.get(&vec![b]).copied().ok_or_else(|| {
                    BunsenError::invalid_resource(format!("no token for the byte {b:#04x}"))
                })
            })
            .collect::<BunsenResult<Vec<u32>>>()?;
        let span_vocab =
            SpanMapVocab::new(ByteMapVocab::from_byte_to_token(&byte_tokens), spans)
                .map_err(|e| BunsenError::invalid_resource("the vocabulary").with_cause(e))?;
        let config: TextSpanningConfig<u32> = TextSpanningConfig::from_pattern(pattern);
        let vocab = UnifiedTokenVocab::from_span_vocab(config, span_vocab)
            .map_err(|e| BunsenError::invalid_resource("the vocabulary").with_cause(e))?;
        let encoder = TokenizerOptions::default().build(Arc::new(vocab));

        let mut added: Vec<(String, i64)> = added.into_iter().map(|(id, c)| (c, id)).collect();
        // Longest first, so a match at a position is the longest there.
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        Ok(Self {
            encoder,
            bytes,
            added,
        })
    }

    /// The id of an added token, by its content: `<|im_end|>`, say.
    pub fn token_id(
        &self,
        content: &str,
    ) -> Option<i64> {
        self.added
            .iter()
            .find(|(c, _)| c == content)
            .map(|&(_, id)| id)
    }

    /// The token count, added tokens included.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Encodes text, an added token's spelling anywhere in it as that token:
    /// what a rendered chat prompt is encoded with.
    ///
    /// # Errors
    /// As wordchipper's encoder.
    pub fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        let mut out = Vec::new();
        let mut rest = text;
        loop {
            // The leftmost match; at one position, the longest, as `added`
            // is sorted longest first.
            let hit = self
                .added
                .iter()
                .filter_map(|(content, id)| {
                    rest.find(content.as_str())
                        .map(|at| (at, content.len(), *id))
                })
                .min_by_key(|&(at, _, _)| at);
            match hit {
                Some((at, len, id)) => {
                    self.encode_plain_into(&rest[..at], &mut out)?;
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain_into(rest, &mut out)?;
                    return Ok(out);
                }
            }
        }
    }

    /// Encodes text with no added tokens: their spellings are text.
    ///
    /// # Errors
    /// As wordchipper's encoder.
    pub fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        let mut out = Vec::new();
        self.encode_plain_into(text, &mut out)?;
        Ok(out)
    }

    fn encode_plain_into(
        &self,
        text: &str,
        out: &mut Vec<i64>,
    ) -> BunsenResult<()> {
        if text.is_empty() {
            return Ok(());
        }
        let normalized: String = unicode_normalization_alignments::UnicodeNormalization::nfc(text)
            .map(|(c, _)| c)
            .collect();
        let ids = self
            .encoder
            .try_encode(
                &normalized,
                Some(&wordchipper::vocab::SpecialFilter::IncludeNone),
            )
            .map_err(|e| {
                BunsenError::invalid_resource("text wordchipper cannot encode").with_cause(e)
            })?;
        out.extend(ids.into_iter().map(i64::from));
        Ok(())
    }

    /// The bytes `ids` spell, an added token as its spelling.
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

impl Detokenizer for Qwen3Tokenizer {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}

/// What the tests build a tokenizer from: a byte-level vocabulary of the
/// 256 bytes and a few merges, Qwen's pre-tokenizer, and three added tokens.
#[cfg(test)]
pub(crate) fn toy_tokenizer_json() -> String {
    // GPT-2's byte alphabet, as `ByteLevelVocabulary` reads it.
    let mut alphabet = Vec::new();
    let mut next = 256u32;
    for b in 0..=255u8 {
        let printable = (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
        alphabet.push(if printable {
            char::from(b)
        } else {
            let c = char::from_u32(next).unwrap();
            next += 1;
            c
        });
    }
    let mut vocab: std::collections::HashMap<String, usize> = alphabet
        .iter()
        .enumerate()
        .map(|(b, c)| (c.to_string(), b))
        .collect();
    let space = alphabet[b' ' as usize];
    for (i, piece) in ["he", "ll", "hell", "hello", &format!("{space}w")]
        .iter()
        .enumerate()
    {
        vocab.insert(piece.to_string(), 256 + i);
    }
    serde_json::json!({
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]},
        "added_tokens": [
            {"id": 300, "content": "<|im_start|>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false},
            {"id": 301, "content": "<|im_end|>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}
        ],
        "model": {"type": "BPE", "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": []}
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use bunsen::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    use super::*;

    fn toy() -> Qwen3Tokenizer {
        let config =
            r#"{"added_tokens_decoder": {"302": {"content": "<think>", "special": false}}}"#;
        Qwen3Tokenizer::from_json(&toy_tokenizer_json(), Some(config)).unwrap()
    }

    #[test]
    fn test_encode_and_decode() {
        let tok = toy();
        assert_eq!(tok.len(), 256 + 5 + 3);
        let ids = tok.encode("<|im_start|>hello world<|im_end|>").unwrap();
        assert_eq!(ids[0], 300);
        assert_eq!(ids[1], 259, "`hello` is one token");
        assert_eq!(*ids.last().unwrap(), 301);
        assert_eq!(
            tok.detokenize(&ids).unwrap(),
            "<|im_start|>hello world<|im_end|>"
        );
    }

    /// The config's added tokens count, an added token's spelling is text
    /// to `encode_plain`, and the text between added tokens is NFC.
    #[test]
    fn test_added_tokens_and_nfc() {
        let tok = toy();
        assert_eq!(tok.token_id("<think>"), Some(302));
        assert_eq!(tok.encode("<think>").unwrap(), [302]);
        assert!(tok.encode_plain("<think>").unwrap().len() > 1);

        // `e` + combining acute is `é` (C3 A9) under NFC.
        assert_eq!(
            tok.encode("e\u{301}").unwrap(),
            tok.encode("\u{e9}").unwrap()
        );
        assert_eq!(tok.encode("e\u{301}").unwrap(), [0xC3, 0xA9]);
    }

    #[test]
    fn test_refuses_what_it_would_misencode() {
        let fallback =
            toy_tokenizer_json().replace("\"byte_fallback\":false", "\"byte_fallback\":true");
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("byte_fallback")
            .assert_err(&Qwen3Tokenizer::from_json(&fallback, None));
    }
}
