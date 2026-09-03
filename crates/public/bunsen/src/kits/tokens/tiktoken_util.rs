//! # The `.tiktoken` rank file.
//!
//! Whisper's base vocabulary ships as a `tiktoken` rank file: one
//! `<base64 bytes> <rank>` pair per line, ranks contiguous from zero.
//! [`TiktokenRanks`] parses one. Nothing here knows about special tokens —
//! they are not in the file; see [`tokens`](whisper_token_layout).
//!
//! The parser is here, rather than borrowed from a tokenizer crate, for one
//! reason. `multilingual.tiktoken` ends with the line `= 50256`: base64 of
//! nothing, and Whisper's rank 50256 is a genuinely empty token. Python's
//! `base64.b64decode` accepts bare padding and returns `b""`; a strict
//! decoder rejects it, and then the file cannot be loaded at all. This one
//! reads an all-padding field as the empty span, as the reference loader
//! does. It is also what keeps text decoding free of any `std`-only I/O in
//! its dependency.

use std::{
    collections::HashMap,
    path::Path,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The base vocabulary of a `.tiktoken` file: rank to bytes.
///
/// Ranks are the token ids the model emits for text; they run from zero
/// without gaps. An entry is raw bytes, not text — a multi-byte character can
/// span several ranks, and one entry can be empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiktokenRanks {
    /// Indexed by rank.
    spans: Vec<Vec<u8>>,
}

impl TiktokenRanks {
    /// Parses the text of a `.tiktoken` file.
    ///
    /// Blank lines are skipped, as the reference loader skips them. Every
    /// other line is `<base64> <rank>`, and the ranks must be exactly
    /// `0..len` in any order.
    ///
    /// # Errors
    /// [`BunsenError::ParseError`] for a malformed line, a rank that repeats
    /// or is missing, or an empty file.
    pub fn parse(text: &str) -> BunsenResult<Self> {
        let mut entries: Vec<(usize, Vec<u8>)> = Vec::new();

        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let number = i + 1;

            let mut fields = line.split_whitespace();
            let (Some(b64), Some(rank), None) = (fields.next(), fields.next(), fields.next())
            else {
                return Err(BunsenError::ParseError(format!(
                    "line {number}: expected `<base64> <rank>`, got {line:?}"
                )));
            };

            let span = lenient_decode_base64(b64).ok_or_else(|| {
                BunsenError::ParseError(format!("line {number}: invalid base64 {b64:?}"))
            })?;
            let rank: usize = rank.parse().map_err(|_| {
                BunsenError::ParseError(format!("line {number}: invalid rank {rank:?}"))
            })?;

            entries.push((rank, span));
        }

        if entries.is_empty() {
            return Err(BunsenError::ParseError(
                "tiktoken file has no ranks".to_string(),
            ));
        }

        let count = entries.len();
        let mut spans: Vec<Option<Vec<u8>>> = vec![None; count];
        for (rank, span) in entries {
            let slot = spans.get_mut(rank).ok_or_else(|| {
                BunsenError::ParseError(format!(
                    "rank {rank} is beyond the {count} entries: ranks must be contiguous from 0"
                ))
            })?;
            if slot.is_some() {
                return Err(BunsenError::ParseError(format!(
                    "rank {rank} appears more than once"
                )));
            }
            *slot = Some(span);
        }

        // A rank beyond the count was caught above, so with as many entries
        // as slots every slot is filled.
        let spans = spans
            .into_iter()
            .map(|span| span.expect("every rank is present"))
            .collect();

        Ok(Self { spans })
    }

    /// Reads and parses a `.tiktoken` file.
    ///
    /// # Errors
    /// [`BunsenError::External`] if the file cannot be read, or whatever
    /// [`parse`](Self::parse) reports.
    pub fn load(path: impl AsRef<Path>) -> BunsenResult<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(BunsenError::external)?;
        Self::parse(&text)
    }

    /// The number of ranks; the first special id in a layout built on this
    /// vocabulary.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether there are no ranks. Never true for a parsed file.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The bytes of one rank, or `None` past the end.
    pub fn get(
        &self,
        rank: usize,
    ) -> Option<&[u8]> {
        self.spans.get(rank).map(Vec::as_slice)
    }

    /// Every rank's bytes, in rank order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.spans.iter().map(Vec::as_slice)
    }

    /// The table itself, indexed by rank.
    pub fn into_spans(self) -> Vec<Vec<u8>> {
        self.spans
    }
}

/// Decodes standard-alphabet base64, leniently.
///
/// Padding is optional, and an all-padding field decodes to nothing — that is
/// how `multilingual.tiktoken` spells its empty token. A symbol outside the
/// alphabet, or a length no encoding can produce, is `None`.
fn lenient_decode_base64(field: &str) -> Option<Vec<u8>> {
    let data = field.trim_end_matches('=');
    if data.len() % 4 == 1 {
        return None;
    }

    let mut out = Vec::with_capacity(data.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0;

    for c in data.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };

        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }

    Some(out)
}

/// The symbols upstream's `non_speech_tokens` suppresses when they are a
/// single token, with or without a leading space.
pub const SYMBOLS: &[&str] = &[
    "\"",
    "#",
    "(",
    ")",
    "*",
    "+",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
    "\u{300c}",
    "\u{300d}",
    "\u{300e}",
    "\u{300f}",
    "<<",
    ">>",
    "<<<",
    ">>>",
    "--",
    "---",
    "-(",
    "-[",
    "('",
    "(\"",
    "((",
    "))",
    "(((",
    ")))",
    "[[",
    "]]",
    "{{",
    "}}",
    "\u{266a}\u{266a}",
    "\u{266a}\u{266a}\u{266a}",
];
/// The music symbols, suppressed by their first token whatever it is.
pub const MISCELLANEOUS: &[&str] = &[
    "\u{2669}", "\u{266a}", "\u{266b}", "\u{266c}", "\u{266d}", "\u{266e}", "\u{266f}",
];

/// The rank file as `{ bytes -> id }`.
pub fn lookup(ranks: &TiktokenRanks) -> HashMap<&[u8], i64> {
    ranks
        .iter()
        .enumerate()
        .map(|(id, bytes)| (bytes, id as i64))
        .collect()
}

/// The id of the longest prefix of `bytes` that is a token: what byte-level
/// BPE emits first for a string that is not itself a token.
fn first_token(
    table: &HashMap<&[u8], i64>,
    bytes: &[u8],
) -> Option<i64> {
    (1..=bytes.len())
        .rev()
        .find_map(|n| table.get(&bytes[..n]).copied())
}

/// Upstream's `Tokenizer.non_speech_tokens`, from the rank file alone:
/// the ids that would make a transcript say `[APPLAUSE]` or draw a music
/// note, plus the leading `-` and `'` that would start a word with one.
pub fn non_speech_tokens(ranks: &TiktokenRanks) -> Vec<i64> {
    let table = lookup(ranks);
    let mut ids: Vec<i64> = [" -", " '"]
        .iter()
        .filter_map(|s| table.get(s.as_bytes()).copied())
        .collect();

    for symbol in SYMBOLS {
        for candidate in [symbol.to_string(), format!(" {symbol}")] {
            if let Some(&id) = table.get(candidate.as_bytes()) {
                ids.push(id);
            }
        }
    }
    for symbol in MISCELLANEOUS {
        for candidate in [symbol.to_string(), format!(" {symbol}")] {
            if let Some(id) = first_token(&table, candidate.as_bytes()) {
                ids.push(id);
            }
        }
    }

    ids.sort_unstable();
    ids.dedup();
    ids
}

/// The id of the single-space token, which opens a blank transcript.
pub fn blank_token(ranks: &TiktokenRanks) -> Option<i64> {
    lookup(ranks).get(b" ".as_slice()).copied()
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;

    use super::*;
    use crate::{
        kits::speech::whisper::{
            LogitFilter,
            SuppressBlank,
        },
        support::testing::CpuBackend,
    };

    #[test]
    fn test_decode_base64() {
        assert_eq!(lenient_decode_base64("IQ=="), Some(b"!".to_vec()));
        assert_eq!(
            lenient_decode_base64("IQ"),
            Some(b"!".to_vec()),
            "padding is optional"
        );
        assert_eq!(lenient_decode_base64("SGVsbG8="), Some(b"Hello".to_vec()));
        assert_eq!(lenient_decode_base64("IGdhemVk"), Some(b" gazed".to_vec()));
        assert_eq!(
            lenient_decode_base64("IPCfjg=="),
            Some(b" \xf0\x9f\x8e".to_vec())
        );
        assert_eq!(lenient_decode_base64("+/8="), Some(vec![0xfb, 0xff]));

        // The empty token, both as the file spells it and as nothing at all.
        assert_eq!(lenient_decode_base64("="), Some(Vec::new()));
        assert_eq!(lenient_decode_base64("=="), Some(Vec::new()));
        assert_eq!(lenient_decode_base64(""), Some(Vec::new()));

        assert_eq!(lenient_decode_base64("I%=="), None, "outside the alphabet");
        assert_eq!(
            lenient_decode_base64("I"),
            None,
            "no encoding is one symbol long"
        );
        assert_eq!(
            lenient_decode_base64("IQ =="),
            None,
            "whitespace is not skipped"
        );
    }

    /// The shape of the real file: base64, a space, a rank; the empty token
    /// as bare padding; blank lines tolerated.
    #[test]
    fn test_parse() {
        let ranks = TiktokenRanks::parse("IQ== 0\nIg== 1\n\n= 2\n").unwrap();

        assert_eq!(ranks.len(), 3);
        assert!(!ranks.is_empty());
        assert_eq!(ranks.get(0), Some(&b"!"[..]));
        assert_eq!(ranks.get(1), Some(&b"\""[..]));
        assert_eq!(ranks.get(2), Some(&b""[..]), "the empty token is present");
        assert_eq!(ranks.get(3), None);
        assert_eq!(ranks.iter().count(), 3);
        assert_eq!(
            ranks.into_spans(),
            vec![b"!".to_vec(), b"\"".to_vec(), Vec::new()]
        );
    }

    /// Ranks may arrive in any order; they still index the table.
    #[test]
    fn test_parse_orders_by_rank() {
        let ranks = TiktokenRanks::parse("Ig== 1\nIQ== 0\n").unwrap();
        assert_eq!(ranks.get(0), Some(&b"!"[..]));
        assert_eq!(ranks.get(1), Some(&b"\""[..]));
    }

    #[test]
    fn test_parse_rejects_malformed_input() {
        for (text, why) in [
            ("", "empty"),
            ("\n\n", "only blank lines"),
            ("IQ==", "no rank"),
            ("IQ== 0 extra", "three fields"),
            ("IQ== x", "rank not a number"),
            ("IQ== -1", "negative rank"),
            ("I%== 0", "bad base64"),
            ("IQ== 0\nIg== 0", "duplicate rank"),
            ("IQ== 0\nIg== 2", "gap in the ranks"),
            ("IQ== 1", "does not start at zero"),
        ] {
            let err = TiktokenRanks::parse(text).expect_err(why);
            assert!(matches!(err, BunsenError::ParseError(_)), "{why}: {err:?}");
        }
    }

    #[test]
    fn test_load_reports_a_missing_file() {
        let err = TiktokenRanks::load("/nonexistent/whisper.tiktoken").unwrap_err();
        assert!(matches!(err, BunsenError::External(_)), "{err:?}");
    }

    fn logits<B: Backend>(
        rows: &[&[f32]],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let vocab = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::from_data(TensorData::new(flat, [rows.len(), vocab]), device)
    }

    fn to_rows<B: Backend>(t: Tensor<B, 2>) -> Vec<Vec<f32>> {
        let [rows, vocab] = t.dims();
        let flat = t.to_data().convert::<f32>().to_vec::<f32>().unwrap();
        flat.chunks(vocab).map(|c| c.to_vec()).collect::<Vec<_>>()[..rows].to_vec()
    }

    /// Only at the first sampled position: once anything has been sampled,
    /// the blank and the stop token are allowed again.
    #[test]
    fn test_suppress_blank() {
        type B = CpuBackend;
        let device = Default::default();

        let filter = SuppressBlank::new(2, 4);
        let first = to_rows(LogitFilter::<B>::apply(
            &filter,
            logits(&[&[0.0, 1.0, 2.0, 3.0, 4.0]], &device),
            &[vec![9, 9]],
            2,
        ));
        assert_eq!(
            first[0],
            vec![0.0, 1.0, f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY]
        );

        let later = to_rows(LogitFilter::<B>::apply(
            &filter,
            logits(&[&[0.0, 1.0, 2.0, 3.0, 4.0]], &device),
            &[vec![9, 9, 1]],
            2,
        ));
        assert_eq!(later[0], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }
}
