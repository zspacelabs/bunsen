//! The Parakeet vocabulary.
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::tokens::Detokenizer,
};

/// The word-boundary mark of a `SentencePiece` vocabulary: a piece that begins
/// with it begins a word.
pub const WORD_MARK: char = '\u{2581}';

/// A Parakeet model's `SentencePiece` vocabulary: one piece per token id,
/// the blank not included.
///
/// Two files carry it, with the same pieces: `NeMo`'s model config lists them
/// as `labels`, and `transformers`' `tokenizer.json` as its BPE vocabulary.
///
/// Pieces spelled `<...>` (`<unk>`, `<pad>`, `<|endoftext|>`, ...) are
/// special. As a [`Detokenizer`] the vocabulary renders every piece it is
/// handed, specials included; [`transcript`](Self::transcript) drops them,
/// which is the text a transcription means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParakeetVocabulary {
    pieces: Vec<String>,
}

impl ParakeetVocabulary {
    /// A vocabulary of `pieces`, in id order.
    pub fn new(pieces: Vec<String>) -> Self {
        Self { pieces }
    }

    /// Reads the `labels` of a `NeMo` model config (`config.json`).
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when `json` does not parse, or has no `labels` array of strings.
    pub fn from_nemo_config(json: &str) -> BunsenResult<Self> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            BunsenError::invalid_resource("a NeMo config that is not JSON").with_cause(e)
        })?;
        let labels = value
            .get("labels")
            .and_then(|l| l.as_array())
            .ok_or_else(|| BunsenError::invalid_resource("a NeMo config without `labels`"))?;
        let pieces = labels
            .iter()
            .map(|p| {
                p.as_str().map(str::to_string).ok_or_else(|| {
                    BunsenError::invalid_resource("a NeMo label that is not a string")
                })
            })
            .collect::<BunsenResult<Vec<_>>>()?;
        Ok(Self::new(pieces))
    }

    /// Reads the vocabulary of a `transformers` `tokenizer.json`: its
    /// `model.vocab`, a `{ piece: id }` map over contiguous ids.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when `json` does not parse, has no `model.vocab` map, or its ids are
    /// not `0..len`.
    pub fn from_tokenizer_json(json: &str) -> BunsenResult<Self> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            BunsenError::invalid_resource("a tokenizer.json that is not JSON").with_cause(e)
        })?;
        let vocab = value
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                BunsenError::invalid_resource("a tokenizer.json without `model.vocab`")
            })?;
        let mut pieces = vec![None; vocab.len()];
        for (piece, id) in vocab {
            let id = id
                .as_u64()
                .map(|id| id as usize)
                .filter(|&id| id < pieces.len())
                .ok_or_else(|| {
                    BunsenError::invalid_resource(format!(
                        "tokenizer.json: `{piece}` has id {id}, outside 0..{}",
                        pieces.len()
                    ))
                })?;
            pieces[id] = Some(piece.clone());
        }
        let pieces = pieces
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| BunsenError::invalid_resource("tokenizer.json: the ids have a gap"))?;
        Ok(Self::new(pieces))
    }

    /// The piece count.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether there are no pieces.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The piece of `id`, if it is in the vocabulary.
    pub fn piece(
        &self,
        id: usize,
    ) -> Option<&str> {
        self.pieces.get(id).map(String::as_str)
    }

    /// Whether `id` is a special piece, spelled `<...>`.
    pub fn is_special(
        &self,
        id: usize,
    ) -> bool {
        self.piece(id)
            .is_some_and(|p| p.len() > 2 && p.starts_with('<') && p.ends_with('>'))
    }

    /// Joins the pieces of `ids`, word marks as spaces.
    fn render(
        &self,
        ids: impl IntoIterator<Item = usize>,
    ) -> BunsenResult<String> {
        let mut text = String::new();
        for id in ids {
            let piece = self.piece(id).ok_or_else(|| {
                BunsenError::invalid_resource(format!(
                    "token id {id} is outside the {}-piece vocabulary",
                    self.len()
                ))
            })?;
            text.extend(piece.chars().map(|c| if c == WORD_MARK { ' ' } else { c }));
        }
        Ok(text)
    }

    /// The transcript `ids` spell: special pieces dropped, word marks as
    /// spaces, and the leading space of the first word trimmed.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if an id is outside the vocabulary.
    pub fn transcript(
        &self,
        ids: &[usize],
    ) -> BunsenResult<String> {
        let text = self.render(ids.iter().copied().filter(|&id| !self.is_special(id)))?;
        Ok(text.trim().to_string())
    }
}

impl Detokenizer for ParakeetVocabulary {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        let ids = ids
            .iter()
            .map(|&id| {
                usize::try_from(id).map_err(|_| {
                    BunsenError::invalid_resource(format!("token id {id} is negative"))
                })
            })
            .collect::<BunsenResult<Vec<_>>>()?;
        self.render(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> ParakeetVocabulary {
        ParakeetVocabulary::new(
            [
                "<unk>",
                "<|endoftext|>",
                "\u{2581}the",
                "\u{2581}mo",
                "on",
                ".",
                "<",
            ]
            .map(String::from)
            .to_vec(),
        )
    }

    #[test]
    fn test_transcript_and_detokenize() {
        let vocab = toy();
        assert_eq!(vocab.len(), 7);
        assert!(vocab.is_special(0) && vocab.is_special(1));
        assert!(!vocab.is_special(6), "a lone `<` is a piece, not a special");
        assert_eq!(vocab.transcript(&[1, 2, 3, 4, 5, 0]).unwrap(), "the moon.");
        assert_eq!(vocab.detokenize(&[2, 1, 6]).unwrap(), " the<|endoftext|><");
        assert!(vocab.transcript(&[7]).is_err());
        assert!(vocab.detokenize(&[-1]).is_err());
    }

    #[test]
    fn test_readers_agree() {
        let nemo = r#"{"labels": ["<unk>", "▁a", "b"], "other": 1}"#;
        let tok = r#"{"model": {"type": "BPE", "vocab": {"b": 2, "<unk>": 0, "▁a": 1}}}"#;
        let a = ParakeetVocabulary::from_nemo_config(nemo).unwrap();
        let b = ParakeetVocabulary::from_tokenizer_json(tok).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.transcript(&[1, 2]).unwrap(), "ab");

        assert!(ParakeetVocabulary::from_nemo_config("{}").is_err());
        let gap = r#"{"model": {"vocab": {"a": 0, "b": 2}}}"#;
        assert!(ParakeetVocabulary::from_tokenizer_json(gap).is_err());
    }
}
