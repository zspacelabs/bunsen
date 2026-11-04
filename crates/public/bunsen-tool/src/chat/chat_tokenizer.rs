//! The tokenizer seam a chat's families share.
use std::{
    path::Path,
    sync::Arc,
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::tokens::Detokenizer,
};

use crate::chat::Qwen3Tokenizer;

/// A checkpoint's tokenizer, as a chat uses it: a rendered prompt to ids,
/// its ids back to text ([`Detokenizer`]), and its tokens by name.
///
/// [`HfTokenizer`] reads any `tokenizer.json` (Gemma's); [`Qwen3Tokenizer`]
/// reads Qwen3's on wordchipper.
pub trait ChatTokenizer: Detokenizer {
    /// Encodes text, an added token's spelling anywhere in it as that token:
    /// what a rendered chat prompt is encoded with.
    ///
    /// # Errors
    /// As the encoder.
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>>;

    /// Encodes text with its special tokens' spellings as text.
    ///
    /// # Errors
    /// As the encoder.
    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>>;

    /// The id of an added token, by its content: `<end_of_turn>`, say.
    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64>;

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

impl ChatTokenizer for Qwen3Tokenizer {
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Qwen3Tokenizer::encode(self, text)
    }

    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Qwen3Tokenizer::encode_plain(self, text)
    }

    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64> {
        Qwen3Tokenizer::token_id(self, content)
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
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json)
            .map_err(|e| tokenizers_error("a tokenizer.json", e))
            .map_err(|e| e.context(tokenizer_json.display()))?;
        Ok(Self::new(tokenizer))
    }

    /// Reads a `tokenizer.json`'s contents.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse.
    pub fn from_json(tokenizer_json: &str) -> BunsenResult<Self> {
        let tokenizer: tokenizers::Tokenizer = tokenizer_json
            .parse()
            .map_err(|e| tokenizers_error("a tokenizer.json", e))?;
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

impl ChatTokenizer for HfTokenizer {
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Self::encode_with(&self.tokenizer, text)
    }

    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Self::encode_with(&self.plain, text)
    }

    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64> {
        self.tokenizer.token_to_id(content).map(i64::from)
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
