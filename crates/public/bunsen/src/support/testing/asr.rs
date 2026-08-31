//! Speech-recognition test helpers: token ids to text, and word error rate.
//!
//! bunsen's Whisper emits **token ids**; judging them against a transcript
//! needs text, and text needs a vocabulary. Encoding is the hard half of a BPE
//! tokenizer — the merge table, the pre-tokenizer regex, the special-token
//! handling — but *decoding* is a lookup and a concatenation, which is all a
//! transcription test needs. So this is a decoder, not a tokenizer, and it is
//! deliberately not the general-purpose one.

use std::path::Path;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Magic bytes at the head of a decode table.
const MAGIC: &[u8; 8] = b"BWVOCAB1";

/// Whisper's special tokens start here in the multilingual vocabulary.
///
/// `<|endoftext|>` is the first; everything at or above it is a control token
/// (language, task, timestamps) rather than text.
pub const WHISPER_FIRST_SPECIAL: usize = 50257;

/// The decode half of a byte-level BPE vocabulary: id to raw bytes.
///
/// Built by `tools/gen_speech_fixtures.py` from `openai-whisper`'s own
/// tokenizer, so the byte-level mapping is already undone and an entry is
/// literal UTF-8 (or a fragment of a character that spans several tokens).
#[derive(Debug, Clone)]
pub struct BpeDecodeTable {
    /// Concatenated token bytes.
    blob: Vec<u8>,
    /// `[start, end)` into `blob`, indexed by token id.
    spans: Vec<(u32, u32)>,
}

impl BpeDecodeTable {
    /// Reads a table written by `tools/gen_speech_fixtures.py`.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the file is not a decode table, or is
    /// truncated.
    pub fn load(path: impl AsRef<Path>) -> BunsenResult<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(BunsenError::external)?;
        Self::from_bytes(&bytes)
    }

    /// Parses a decode table from its serialized form.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the header is wrong or the body is
    /// truncated.
    pub fn from_bytes(bytes: &[u8]) -> BunsenResult<Self> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err(BunsenError::Invalid(
                "not a bunsen BPE decode table".to_string(),
            ));
        }

        let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        let mut blob = Vec::with_capacity(bytes.len().saturating_sub(12 + count));
        let mut spans = Vec::with_capacity(count);
        let mut at = 12;

        for id in 0..count {
            let len = *bytes.get(at).ok_or_else(|| {
                BunsenError::Invalid(format!("decode table truncated at token {id}"))
            })? as usize;
            at += 1;

            let end = at + len;
            if end > bytes.len() {
                return Err(BunsenError::Invalid(format!(
                    "decode table truncated inside token {id}"
                )));
            }

            let start = blob.len() as u32;
            blob.extend_from_slice(&bytes[at..end]);
            spans.push((start, blob.len() as u32));
            at = end;
        }

        Ok(Self { blob, spans })
    }

    /// The number of ids the table covers.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The raw bytes for one id, or `None` if it is out of range.
    pub fn bytes_of(
        &self,
        id: i64,
    ) -> Option<&[u8]> {
        let id = usize::try_from(id).ok()?;
        let &(start, end) = self.spans.get(id)?;
        Some(&self.blob[start as usize..end as usize])
    }

    /// Decodes ids to text, dropping ids at or above `first_special`.
    ///
    /// Tokens are concatenated **before** the UTF-8 decode, because a
    /// multi-byte character can span two tokens; decoding each separately
    /// would replace both halves. An id the table does not cover contributes
    /// nothing.
    ///
    /// # Arguments
    /// * `ids` - the token ids to decode.
    /// * `first_special` - the lowest control-token id, e.g.
    ///   [`WHISPER_FIRST_SPECIAL`]. Pass `usize::MAX` to keep everything.
    pub fn decode(
        &self,
        ids: &[i64],
        first_special: usize,
    ) -> String {
        let mut raw = Vec::new();
        for &id in ids {
            if usize::try_from(id).is_ok_and(|id| id >= first_special) {
                continue;
            }
            if let Some(b) = self.bytes_of(id) {
                raw.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&raw).into_owned()
    }
}

/// Normalizes a transcript for comparison: lowercase, punctuation stripped,
/// whitespace collapsed.
///
/// This is deliberately much smaller than Whisper's own `EnglishTextNormalizer`
/// — it does **not** reconcile spelled-out numbers with digits, or expand
/// contractions. A threshold set against this normalizer is therefore counting
/// some formatting differences as errors; see the callers' notes.
pub fn normalize_transcript(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word error rate: edit distance over words, divided by the reference length.
///
/// `(substitutions + insertions + deletions) / reference_words`. Not bounded
/// by 1 — a hypothesis longer than the reference can exceed it.
///
/// Returns `0.0` when both are empty, and `1.0` when only the reference is.
///
/// # Arguments
/// * `hypothesis` - normalized words produced by the model.
/// * `reference` - normalized words from the transcript.
pub fn word_error_rate(
    hypothesis: &[String],
    reference: &[String],
) -> f64 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }

    // Two-row Levenshtein: the full matrix is O(n*m) memory for no benefit.
    let mut prev: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut cur = vec![0usize; hypothesis.len() + 1];

    for (r, refr) in reference.iter().enumerate() {
        cur[0] = r + 1;
        for (h, hyp) in hypothesis.iter().enumerate() {
            let cost = usize::from(refr != hyp);
            cur[h + 1] = (prev[h] + cost).min(prev[h + 1] + 1).min(cur[h] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    prev[hypothesis.len()] as f64 / reference.len() as f64
}

/// Word error rate between two raw strings, normalized by
/// [`normalize_transcript`].
pub fn text_error_rate(
    hypothesis: &str,
    reference: &str,
) -> f64 {
    word_error_rate(
        &normalize_transcript(hypothesis),
        &normalize_transcript(reference),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        normalize_transcript(s)
    }

    #[test]
    fn test_normalize_strips_case_and_punctuation() {
        assert_eq!(
            words("  The Moon?? Why -- choose  this,  as our goal! "),
            ["the", "moon", "why", "choose", "this", "as", "our", "goal"],
        );
        // Apostrophes are kept: "we're" and "were" are different words.
        assert_eq!(words("we're"), ["we're"]);
    }

    #[test]
    fn test_wer_counts_each_edit_once() {
        let refr = words("we choose to go to the moon");

        assert_eq!(word_error_rate(&refr, &refr), 0.0);

        // One substitution out of seven.
        let sub = words("we choose to go to the sun");
        assert!((word_error_rate(&sub, &refr) - 1.0 / 7.0).abs() < 1e-12);

        // One deletion out of seven.
        let del = words("we choose to go to moon");
        assert!((word_error_rate(&del, &refr) - 1.0 / 7.0).abs() < 1e-12);

        // One insertion out of seven.
        let ins = words("we choose to go to the big moon");
        assert!((word_error_rate(&ins, &refr) - 1.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_wer_edge_cases() {
        assert_eq!(word_error_rate(&[], &[]), 0.0);
        assert_eq!(word_error_rate(&words("anything"), &[]), 1.0);
        // Nothing recognized at all is a full miss.
        assert_eq!(word_error_rate(&[], &words("we choose to go")), 1.0);
    }

    #[test]
    fn test_decode_table_rejects_junk() {
        assert!(BpeDecodeTable::from_bytes(b"nope").is_err());
        assert!(BpeDecodeTable::from_bytes(b"BWVOCAB1\x01\x00\x00\x00").is_err());
    }

    /// A character split across two tokens must survive the join.
    #[test]
    fn test_decode_joins_before_utf8() {
        // "é" is 0xC3 0xA9, here split across ids 0 and 1.
        let mut blob = Vec::from(*MAGIC);
        blob.extend_from_slice(&2u32.to_le_bytes());
        blob.extend_from_slice(&[1, 0xC3, 1, 0xA9]);

        let table = BpeDecodeTable::from_bytes(&blob).unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(table.decode(&[0, 1], usize::MAX), "é");
    }
}
