//! The tokenizer seam a chat's families share.
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
    kits::support::tokens::{
        Detokenizer,
        Tokenizer,
    },
};
use serde_json::Value;

use crate::chat::Qwen3Tokenizer;

/// A checkpoint's tokenizer, as a chat uses it: a rendered prompt to ids and
/// its tokens by name ([`Tokenizer`]), its ids back to text
/// ([`Detokenizer`]).
///
/// [`HfTokenizer`] reads any `tokenizer.json` (Gemma's); [`Qwen3Tokenizer`]
/// reads Qwen3's on wordchipper.
pub trait ChatTokenizer: Detokenizer + Tokenizer {
    /// Encodes text with its special tokens' spellings as text.
    ///
    /// # Errors
    /// As the encoder.
    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>>;

    /// Whether `id` is a token.
    fn contains(
        &self,
        id: i64,
    ) -> bool;

    /// The token count, added tokens included.
    fn len(&self) -> usize;

    /// Whether there are no tokens.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Tokenizer for Qwen3Tokenizer {
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Qwen3Tokenizer::encode(self, text)
    }

    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64> {
        Qwen3Tokenizer::token_id(self, content)
    }
}

impl ChatTokenizer for Qwen3Tokenizer {
    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Qwen3Tokenizer::encode_plain(self, text)
    }

    fn contains(
        &self,
        id: i64,
    ) -> bool {
        Qwen3Tokenizer::contains(self, id)
    }

    fn len(&self) -> usize {
        Qwen3Tokenizer::len(self)
    }
}

/// A checkpoint's tokenizer, on the `tokenizers` crate: `transformers`' own
/// encoder, reading the normalizer, the added tokens, the pre-tokenizer, the
/// model and the decoder from one `tokenizer.json`, so it encodes as
/// `transformers` does by construction. Gemma's SentencePiece-style BPE,
/// with its byte fallback, reads here.
///
/// A checkpoint's `tokenizer_config.json`, when given, adds its added tokens
/// (`added_tokens_decoder`) that the `tokenizer.json` lacks, at their ids, as
/// `transformers` loads them: a Qwen3 `-Base` checkpoint's `tokenizer.json`
/// omits four (`<tool_response>`, `</tool_response>`, `<think>`, `</think>`)
/// that its `tokenizer_config.json` lists.
///
/// Decoding is the whole sequence's at once (the decoder joins byte-fallback
/// tokens before the UTF-8 decode), special tokens as their spelling.
#[derive(Clone)]
pub struct HfTokenizer {
    tokenizer: Arc<tokenizers::Tokenizer>,
    plain: Arc<tokenizers::Tokenizer>,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("tokens", &self.len())
            .finish()
    }
}

/// `e`, a `tokenizers` error, as an `InvalidResource` naming `what`.
fn tokenizers_error(
    what: &str,
    e: impl std::fmt::Display,
) -> BunsenError {
    BunsenError::invalid_resource(format!("{what}: {e}"))
}

impl HfTokenizer {
    /// Reads a `tokenizer.json`.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when it cannot be read or does not parse.
    pub fn from_file(tokenizer_json: &Path) -> BunsenResult<Self> {
        Self::from_files(tokenizer_json, None)
    }

    /// Reads a checkpoint's `tokenizer.json`, and its `tokenizer_config.json`
    /// when given.
    ///
    /// # Errors
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) or
    /// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) when a file cannot
    /// be read; as [`from_json_with_config`](Self::from_json_with_config).
    pub fn from_files(
        tokenizer_json: &Path,
        tokenizer_config_json: Option<&Path>,
    ) -> BunsenResult<Self> {
        let read = |p: &Path| std::fs::read_to_string(p).map_err(sys_at("read", p));
        let tokenizer = read(tokenizer_json)?;
        let config = tokenizer_config_json.map(read).transpose()?;
        Self::from_json_with_config(&tokenizer, config.as_deref())
            .map_err(|e| e.context(tokenizer_json.display()))
    }

    /// Reads a `tokenizer.json`'s contents.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse.
    pub fn from_json(tokenizer_json: &str) -> BunsenResult<Self> {
        Self::from_json_with_config(tokenizer_json, None)
    }

    /// Reads a `tokenizer.json`'s contents, and a `tokenizer_config.json`'s
    /// when given.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when either does not parse, or the config names an added token the
    /// `tokenizer.json` spells otherwise;
    /// [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) when an
    /// added token only the config lists cannot take its id (it is not the
    /// next free one).
    pub fn from_json_with_config(
        tokenizer_json: &str,
        tokenizer_config_json: Option<&str>,
    ) -> BunsenResult<Self> {
        let mut tokenizer: tokenizers::Tokenizer = tokenizer_json
            .parse()
            .map_err(|e| tokenizers_error("a tokenizer.json", e))?;
        if let Some(config) = tokenizer_config_json {
            add_config_tokens(&mut tokenizer, config)?;
        }
        Ok(Self::new(tokenizer))
    }

    fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        let mut plain = tokenizer.clone();
        plain.set_encode_special_tokens(true);
        Self {
            tokenizer: Arc::new(tokenizer),
            plain: Arc::new(plain),
        }
    }

    fn encode_with(
        tokenizer: &tokenizers::Tokenizer,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| tokenizers_error("text the tokenizer cannot encode", e))?;
        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }
}

/// Adds the added tokens of `config_json`, a `tokenizer_config.json`, that
/// `tokenizer` lacks, in id order, each at its id.
fn add_config_tokens(
    tokenizer: &mut tokenizers::Tokenizer,
    config_json: &str,
) -> BunsenResult<()> {
    let config: Value = serde_json::from_str(config_json)
        .map_err(|e| BunsenError::invalid_resource("a tokenizer_config.json").with_cause(e))?;
    let Some(decoder) = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
    else {
        return Ok(());
    };
    let mut missing: BTreeMap<u32, tokenizers::AddedToken> = BTreeMap::new();
    for (id, token) in decoder {
        let id: u32 = id
            .parse()
            .map_err(|_| BunsenError::invalid_resource(format!("an added token id `{id}`")))?;
        let content = token
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| BunsenError::invalid_resource(format!("added token {id}'s content")))?;
        match tokenizer.id_to_token(id) {
            Some(known) if known == content => continue,
            Some(known) => {
                return Err(BunsenError::invalid_resource(format!(
                    "added token {id} is `{content}` in tokenizer_config.json and `{known}` in \
                     tokenizer.json"
                )));
            }
            None => {}
        }
        let flag = |key: &str| token.get(key).and_then(Value::as_bool);
        let special = flag("special").unwrap_or(false);
        let added = tokenizers::AddedToken::from(content, special)
            .single_word(flag("single_word").unwrap_or(false))
            .lstrip(flag("lstrip").unwrap_or(false))
            .rstrip(flag("rstrip").unwrap_or(false))
            .normalized(flag("normalized").unwrap_or(!special));
        missing.insert(id, added);
    }
    for (id, token) in missing {
        let content = token.content.clone();
        if token.special {
            tokenizer.add_special_tokens(&[token]);
        } else {
            tokenizer.add_tokens(&[token]);
        }
        let got = tokenizer.token_to_id(&content);
        if got != Some(id) {
            return Err(BunsenError::unsupported(format!(
                "tokenizer_config.json's added token {id} (`{content}`), which tokenizer.json \
                 lacks, takes id {got:?}, not its own"
            )));
        }
    }
    Ok(())
}

impl Detokenizer for HfTokenizer {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        let ids = ids
            .iter()
            .map(|&id| {
                if self.contains(id) {
                    Ok(id as u32)
                } else {
                    Err(BunsenError::invalid_resource(format!(
                        "token id {id} is outside the vocabulary"
                    )))
                }
            })
            .collect::<BunsenResult<Vec<u32>>>()?;
        self.tokenizer
            .decode(&ids, false)
            .map_err(|e| tokenizers_error("ids the tokenizer cannot decode", e))
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Self::encode_with(&self.tokenizer, text)
    }

    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64> {
        self.tokenizer.token_to_id(content).map(i64::from)
    }
}

impl ChatTokenizer for HfTokenizer {
    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Self::encode_with(&self.plain, text)
    }

    fn contains(
        &self,
        id: i64,
    ) -> bool {
        u32::try_from(id).is_ok_and(|id| self.tokenizer.id_to_token(id).is_some())
    }

    fn len(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

/// A toy SentencePiece-style tokenizer, as Gemma's `tokenizer.json` is
/// built: spaces as `▁`, byte fallback, two special tokens.
#[cfg(test)]
pub(crate) fn toy_sentencepiece_json() -> String {
    let mut vocab: Vec<(String, f64)> = vec![
        ("<pad>".into(), 0.0),
        ("<eos>".into(), 0.0),
        ("<bos>".into(), 0.0),
        ("<unk>".into(), 0.0),
    ];
    for b in 0..=255u8 {
        vocab.push((format!("<0x{b:02X}>"), 0.0));
    }
    for piece in [
        "▁", "h", "i", "▁hi", "▁there", "t", "e", "r", "o", "▁to", "model", "user",
    ] {
        vocab.push((piece.to_string(), -1.0));
    }
    let n = vocab.len();
    let merges: Vec<String> = Vec::new();
    let vocab_map: serde_json::Map<String, serde_json::Value> = vocab
        .iter()
        .enumerate()
        .map(|(i, (p, _))| (p.clone(), serde_json::json!(i)))
        .collect();
    serde_json::json!({
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Special tokens' spellings are tokens in a rendered prompt and text
    /// under `encode_plain`; a character the vocabulary lacks falls back to
    /// its bytes, which decode back to it, joined.
    #[test]
    fn test_hf_tokenizer() {
        let tok = HfTokenizer::from_json(&toy_sentencepiece_json()).unwrap();
        let start = tok.token_id("<start_of_turn>").unwrap();
        let end = tok.token_id("<end_of_turn>").unwrap();
        assert_eq!(tok.len(), end as usize + 1);
        assert!(tok.contains(end) && !tok.contains(end + 1) && !tok.contains(-1));

        let ids = tok.encode("<start_of_turn>user hi<end_of_turn>").unwrap();
        assert_eq!(ids.first(), Some(&start));
        assert_eq!(ids.last(), Some(&end));
        assert_eq!(
            tok.detokenize(&ids).unwrap(),
            "<start_of_turn>user hi<end_of_turn>"
        );
        let plain = tok.encode_plain("<end_of_turn>").unwrap();
        assert!(!plain.contains(&end), "spelled out: {plain:?}");

        // `é` is no piece: its two UTF-8 bytes, joined on decode.
        let ids = tok.encode("hé").unwrap();
        assert!(ids.len() >= 3, "{ids:?}");
        assert_eq!(tok.detokenize(&ids).unwrap(), "hé");
        assert!(tok.detokenize(&[end + 7]).is_err());
    }

    /// A `tokenizer_config.json`'s added tokens that the `tokenizer.json`
    /// lacks are added at their ids and matched in text; the ones it has are
    /// left as they are.
    #[test]
    fn test_hf_tokenizer_config_tokens() {
        let json = toy_sentencepiece_json();
        let n = HfTokenizer::from_json(&json).unwrap().len() as i64;
        let config = |tokens: &[(i64, &str, bool)]| {
            let decoder: serde_json::Map<String, serde_json::Value> = tokens
                .iter()
                .map(|&(id, content, special)| {
                    (
                        id.to_string(),
                        serde_json::json!({
                            "content": content, "lstrip": false, "normalized": false,
                            "rstrip": false, "single_word": false, "special": special
                        }),
                    )
                })
                .collect();
            serde_json::json!({ "added_tokens_decoder": decoder }).to_string()
        };

        let tok = HfTokenizer::from_json_with_config(
            &json,
            Some(&config(&[
                (n - 1, "<end_of_turn>", true),
                (n, "<think>", false),
                (n + 1, "</think>", false),
            ])),
        )
        .unwrap();
        assert_eq!(tok.len(), n as usize + 2);
        assert_eq!(tok.token_id("<think>"), Some(n));
        assert_eq!(tok.token_id("</think>"), Some(n + 1));
        let ids = tok.encode("<think>hi</think>").unwrap();
        assert_eq!(ids.first(), Some(&n));
        assert_eq!(ids.last(), Some(&(n + 1)));
        assert_eq!(tok.detokenize(&ids).unwrap(), "<think>hi</think>");

        // A config with no added tokens changes nothing.
        let tok = HfTokenizer::from_json_with_config(&json, Some("{}")).unwrap();
        assert_eq!(tok.len(), n as usize);

        // An id the tokenizer.json spells otherwise, or one past the next
        // free id, is refused.
        assert!(
            HfTokenizer::from_json_with_config(&json, Some(&config(&[(n - 1, "<other>", true)])))
                .is_err()
        );
        assert!(
            HfTokenizer::from_json_with_config(&json, Some(&config(&[(n + 1, "<think>", false)])))
                .is_err()
        );
    }

    /// Qwen3's tokenizer is a `ChatTokenizer` through its own methods.
    #[test]
    fn test_qwen3_is_a_chat_tokenizer() {
        let tok: Arc<dyn ChatTokenizer> = Arc::new(
            Qwen3Tokenizer::from_json(&crate::chat::tokenizer::toy_tokenizer_json(), None).unwrap(),
        );
        assert!(!tok.is_empty());
        let ids = tok.encode("hi").unwrap();
        assert_eq!(tok.detokenize(&ids).unwrap(), "hi");
    }
}
